#include "DataFormats/Common/interface/Wrapper.h"
#include "IOPool/Streamer/src/StreamerFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "FWCore/Framework/interface/EventPrincipal.h"


//#include "IOPool/Streamer/src/StreamerUtilities.h"

#include "TClass.h"

#include <algorithm>
#include <iterator>

using namespace std;
using namespace edm;

namespace edmtestp
{  
/**
    void loadextrastuff()
    {
      static bool loaded = false;
      if(loaded==false)
        {
          loadExtraClasses();
          loaded=true;
        }
    }

    template <class T>
    TClass* getTClassFor()
    {
      static TClass* ans = 0;
      loadextrastuff();
      if(!ans) {
        if((ans = getTClass(typeid(T)))==0) {
          throw cms::Exception("gettclass")
            << "Could not get the TClass for "
            << typeid(T).name() << "\n";
        }
      }
      return ans;
    } 
**/
  // ----------------------------------

  StreamerFileReader::StreamerFileReader(edm::ParameterSet const& pset,
					       edm::InputSourceDescription const& desc):
    edm::InputSource(pset, desc),
    filename_(pset.getParameter<string>("fileName")),
    stream_reader_(new StreamerInputFile(filename_.c_str()))
    //tc_(),
    //desc_()
  {
    std::auto_ptr<edm::SendJobHeader> p = readHeader(stream_reader_);
    SendDescs & descs = p->descs_;
    mergeWithRegistry(descs, productRegistry());

    // jbk - the next line should not be needed
    declareStreamers(descs);
    buildClassCache(descs);
    edm::loadExtraClasses();
  }

  StreamerFileReader::~StreamerFileReader()
  {
      delete stream_reader_;
  }

  std::auto_ptr<edm::EventPrincipal> StreamerFileReader::read()
  {
    return edm::readEvent(productRegistry(), stream_reader_);
  }

  /****************
  std::auto_ptr<SendJobHeader> StreamerFileReader::readHeader()
  {

  desc_ = edm::getTClassFor<SendJobHeader>();
 
  InitMsgView* header = (InitMsgView*) stream_reader_->startMessage();
  
  if(header->code() != 0) //INIT Msg
      throw cms::Exception("readHeader","StreamerFileReader")
        << "received wrong message type: expected INIT, got "
        << header->code() << "\n";

  TBuffer xbuf(TBuffer::kRead, header->descLength(),(char*)header->descData(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc_));

    if(sd.get()==0) {
        throw cms::Exception("HeaderDecode","DecodeProductList")
          << "Could not read the initial product registry list\n";
    }

    return sd;
  }
 std::auto_ptr<EventPrincipal> StreamerFileReader::readEvent(const ProductRegistry& pr)
 {

    if (! stream_reader_->next() ) 
        return std::auto_ptr<edm::EventPrincipal>();

    EventMsgView* eview = (EventMsgView*) stream_reader_->currentRecord();
      
    cout << "Decide event: "
              << eview->event() << " "
              << eview->run() << " "
              << eview->size() << " "
              << eview->eventLength() << " "
              << eview->eventData()
              << endl;

    tc_ = edm::getTClassFor<SendEvent>();
    TBuffer xbuf(TBuffer::kRead, eview->eventLength(),(char*) eview->eventData(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendEvent> sd((SendEvent*)xbuf.ReadObjectAny(tc_)); 
    if(sd.get()==0)
      {
        throw cms::Exception("EventInput","Read")
          << "got a null event from input stream\n";
      } 

    FDEBUG(5) << "Got event: " << sd->id_ << " " << sd->prods_.size() << endl;
    auto_ptr<EventPrincipal> ep(new EventPrincipal(sd->id_,
                                                   sd->time_,
                                                   pr));
    // no process name list handling

    SendProds::iterator spi(sd->prods_.begin()),spe(sd->prods_.end());
    for(;spi!=spe;++spi)
      {
        FDEBUG(10) << "check prodpair" << endl;
        if(spi->prov()==0)
          throw cms::Exception("NoData","EmptyProvenance");
        if(spi->desc()==0)
          throw cms::Exception("NoData","EmptyDesc");

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
            throw cms::Exception("NoData","EmptyProduct");
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
          FDEBUG(10) << "addgroup empty next " << aprov->event.productID_ << endl;
          ep->addGroup(auto_ptr<Group>(new Group(aprov, false)));
          FDEBUG(10) << "addgroup empty done" << endl;
	}
        spi->clear();
      }

    FDEBUG(10) << "Size = " << ep->numEDProducts() << endl;

    return ep;
 } 
  *********/

  void StreamerFileReader::mergeWithRegistry(const SendDescs& descs,
                         ProductRegistry& reg)
  {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());
  
    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct
  
    FDEBUG(6) << "mergeWithRegistry: Product List: " << endl;
    for(; i != e; ++i) {
        reg.copyProduct(*i);
        FDEBUG(6) << "StreamInput prod = " << i->className() << endl;
    }

    // not needed any more
    // fillStreamers(*pr_);
  }

  void StreamerFileReader::declareStreamers(const SendDescs& descs)
  {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        string real_name = edm::wrappedClassName(i->className());
        FDEBUG(6) << "declare: " << real_name << endl;
        edm::loadCap(real_name);
    }
  }

  void StreamerFileReader::buildClassCache(const SendDescs& descs)
  {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        string real_name = edm::wrappedClassName(i->className());
        FDEBUG(6) << "BuildReadData: " << real_name << endl;
        edm::doBuildRealData(real_name);
    }
  }




}


