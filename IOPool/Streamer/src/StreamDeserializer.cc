/**
 * StreamDeserializer.cc
 *
 * Utility class for deserializing streamer message objects
 * into framework objects (e.g. ProductRegistry and EventPrincipal)
 */

#include "IOPool/Streamer/interface/StreamDeserializer.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Streamer/interface/StreamedProducts.h"

#include "zlib.h"

using namespace std;

namespace edm
{

  /**
   * Creates a translator instance for the specified product registry.
   */
  StreamDeserializer::StreamDeserializer():
    processConfiguration_()
  { }

  /**
   * Deserializes the specified init message into a SendJobHeader object
   * (which is related to the product registry).
   */
  std::auto_ptr<SendJobHeader>
  StreamDeserializer::deserializeRegistry(InitMsgView const& initView)
  {
    if(initView.code() != Header::INIT)
      throw cms::Exception("StreamTranslation","Registry deserialization error")
        << "received wrong message type: expected INIT, got "
        << initView.code() << "\n";

    TClass* desc = getTClass(typeid(SendJobHeader));

    TBuffer xbuf(TBuffer::kRead, initView.descLength(),
                 (char*)initView.descData(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc));

    if(sd.get()==0) 
      {
        throw cms::Exception("StreamTranslation","Registry deserialization error")
          << "Could not read the initial product registry list\n";
      }

    return sd;  
  }

  /**
   * Deserializes the specified event message into an EventPrincipal object.
   */
  std::auto_ptr<EventPrincipal>
  StreamDeserializer::deserializeEvent(EventMsgView const& eventView,
                                     const ProductRegistry& productRegistry)
  {
    if(eventView.code() != Header::EVENT)
      throw cms::Exception("StreamTranslation","Event deserialization error")
        << "received wrong message type: expected EVENT, got "
        << eventView.code() << "\n";
    FDEBUG(9) << "Decode event: "
         << eventView.event() << " "
         << eventView.run() << " "
         << eventView.size() << " "
         << eventView.eventLength() << " "
         << eventView.eventData()
         << endl;
    // uncompress if we need to
    // 78 was a dummy value (for no uncompressed) - should be 0 for uncompressed
    // need to get rid of this when 090 MTCC streamers are gotten rid of
    unsigned long origsize = eventView.reserved();
    ///unsigned char dest[7008*1000];
    std::vector<unsigned char> dest;
    unsigned long dest_size = 7008*1000; //(should be >= eventView.reserved() )
    if(eventView.reserved() != 78 && eventView.reserved() != 0)
    {
      dest_size = eventView.reserved();
      dest.resize(dest_size);
      int ret = uncompress(&dest[0], &dest_size, (unsigned char*)eventView.eventData(),
                          eventView.eventLength()); // do not need compression level
      //cout<<"unCompress Return value: "<<ret<< " Okay = " << Z_OK << endl;
      if(ret == Z_OK) {
        // check the length against original uncompressed length
        FDEBUG(10) << " original size = " << origsize << " final size = " 
                   << dest_size << endl;
        if(origsize != dest_size) {
          cerr << "deserializeEvent: Problem with uncompress, original size = "
               << origsize << " uncompress size = " << dest_size << endl;
          // we throw an error and return without event! null pointer
          throw cms::Exception("StreamTranslation","Deserialization error")
            << "mismatch event lengths should be" << origsize << " got "
            << dest_size << "\n";
          // do I need to return here?
          return std::auto_ptr<EventPrincipal>();
        }
      }
      else
      {
        // we throw an error and return without event! null pointer
        cerr << "deserializeEvent: Problem with uncompress, return value = "
             << ret << endl;
        throw cms::Exception("StreamTranslation","Deserialization error")
            << "Error code = " << ret << "\n ";
        // do I need to return here?
        return std::auto_ptr<EventPrincipal>();
      }
    }
    else // not compressed
    {
      // we need to copy anyway the buffer as we are using dest in xbuf
      dest_size = eventView.eventLength();
      dest.resize(dest_size);
      unsigned char* pos = (unsigned char*) &dest[0];
      unsigned char* from = (unsigned char*) eventView.eventData();
      copy(from,from+dest_size,pos);
    }
    TBuffer xbuf(TBuffer::kRead, dest_size,
                 (char*) &dest[0],kFALSE);
    //TBuffer xbuf(TBuffer::kRead, eventView.eventLength(),
    //             (char*) eventView.eventData(),kFALSE);
    RootDebug tracer(10,10);
    TClass* tc = getTClass(typeid(SendEvent));
    auto_ptr<SendEvent> sd((SendEvent*)xbuf.ReadObjectAny(tc));
    if(sd.get()==0)
      {
        throw cms::Exception("StreamTranslation","Event deserialization error")
          << "got a null event from input stream\n";
      }

    FDEBUG(5) << "Got event: " << sd->id_ << " " << sd->prods_.size() << endl;
    auto_ptr<EventPrincipal> ep(new EventPrincipal(sd->id_,
                                                   sd->time_,
                                                   productRegistry,
						   processConfiguration_));
    // Add processConfiguration_ to the process history.
    ep->addToProcessHistory();
    // no process name list handling

    SendProds::iterator spi(sd->prods_.begin()),spe(sd->prods_.end());
    for(;spi!=spe;++spi)
      {
        FDEBUG(10) << "check prodpair" << endl;
        if(spi->prov()==0)
          throw cms::Exception("StreamTranslation","EmptyProvenance");
        if(spi->desc()==0)
          throw cms::Exception("StreamTranslation","EmptyDesc");
        FDEBUG(5) << "Prov:"
             << " " << spi->desc()->className()
             << " " << spi->desc()->productInstanceName()
             << " " << spi->desc()->productID()
             << " " << spi->prov()->productID_
             << endl;

        if(spi->prod()==0)
          {
            FDEBUG(10) << "Product is null" << endl;
            continue;
            throw cms::Exception("StreamTranslation","EmptyProduct");
          }

        auto_ptr<EDProduct>
          aprod(const_cast<EDProduct*>(spi->prod()));
        auto_ptr<BranchEntryDescription>
          aedesc(const_cast<BranchEntryDescription*>(spi->prov()));
        auto_ptr<BranchDescription>
          adesc(const_cast<BranchDescription*>(spi->desc()));

        auto_ptr<Provenance> aprov(new Provenance);
        aprov->event   = *(aedesc.get());
        aprov->product = *(adesc.get());
        if(aprov->isPresent()) {
          FDEBUG(10) << "addgroup next " << aprov->productID() << endl;
          FDEBUG(10) << "addgroup next " << aprov->event.productID_ << endl;
          ep->addGroup(auto_ptr<Group>(new Group(aprod,aprov)));
          FDEBUG(10) << "addgroup done" << endl;
        } else {
          FDEBUG(10) << "addgroup empty next " << aprov->productID() << endl;
          FDEBUG(10) << "addgroup empty next " << aprov->event.productID_ 
                                               << endl;
          ep->addGroup(auto_ptr<Group>(new Group(aprov, false)));
          FDEBUG(10) << "addgroup empty done" << endl;
        }
        spi->clear();
      }

    FDEBUG(10) << "Size = " << ep->numEDProducts() << endl;

    return ep;     
  }

}
