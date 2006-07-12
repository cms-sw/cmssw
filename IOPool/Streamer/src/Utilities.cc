
#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Framework/interface/Event.h"

#include <typeinfo>
#include <iostream>

using namespace std;

namespace edm
{

  // this code is not design to be accessed in multiple threads

  namespace
  {
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
  }

int serializeRegistry(Selections const& prods, InitMsgBuilder& init_message)
  {

    //FDEBUG(6) << "StreamOutput: serializeRegistry" << endl;
    TClass* prog_reg = getTClass(typeid(SendJobHeader));
    SendJobHeader sd;

    Selections::const_iterator i(prods.begin()),e(prods.end());

    FDEBUG(9) << "Product List: " << endl;
    cout << "Product List: " << endl;

    for(;i!=e;++i)  
      {
        sd.descs_.push_back(**i);
        FDEBUG(9) << "StreamOutput got product = " << (*i)->className()
                  << endl;
        cout << "StreamOutput got product = " << (*i)->className() <<endl;
      }

    int sufficiently_large_size = 100 * 1000;     

    TBuffer rootbuf(TBuffer::kWrite,sufficiently_large_size,
                               init_message.dataAddress(),kFALSE);

    RootDebug tracer(10,10);

    int bres = rootbuf.WriteObjectAny((char*)&sd,prog_reg);

    switch(bres)
      {
      case 0: // failure
        {
          throw cms::Exception("Output","SerializationReg")
            << "EventStreamOutput module could not serialize event\n";
          break;
        }
      case 1: // succcess
        break;
      case 2: // truncated result
        {
          throw cms::Exception("Output","SerializationReg")
            << "EventStreamOutput module attempted to serialize the registry\n"
            << "that is to big for the allocated buffers\n";
          break;
        }
      default: // unknown
        {
          throw cms::Exception("Output","SerializationReg")
            << "EventStreamOutput module got an unknown error code\n"
            << " while attempting to serialize registry\n";
          break;
        }
      }

    cout<<"Returning rootbuf.Length()"<<rootbuf.Length()<<endl;
    return  rootbuf.Length();
}

int serializeEvent(EventPrincipal const& e, 
                   Selections const* selections, 
                   EventMsgBuilder& msg, int maxEventSize)
  {
    std::list<Provenance> provenances; // Use list so push_back does not invalidate iterators.
    // all provenance data needs to be transferred, including the
    // indirect stuff referenced from the product provenance structure.
    SendEvent se(e.id(),e.time());

    Selections::const_iterator i(selections->begin()),ie(selections->end());
    // Loop over EDProducts, fill the provenance, and write.

    cout<<"Loop over EDProducts, fill the provenance, and write"<<endl;

    for(; i != ie; ++i) {
      BranchDescription const& desc = **i;
      ProductID const& id = desc.productID();

      if (id == ProductID()) {
        throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
          << "EventStreamOutput::serialize: invalid ProductID supplied in productRegistry\n";
      }
      EventPrincipal::SharedGroupPtr const group = e.getGroup(id);
      if (group.get() == 0) {
        std::string const& name = desc.className();
        std::string const className = wrappedClassName(name);
        TClass *cp = gROOT->GetClass(className.c_str());
	if (cp == 0) {
          throw edm::Exception(errors::ProductNotFound,"NoMatch")
            << "TypeID::className: No dictionary for class " << className << '\n'
            << "Add an entry for this class\n"
            << "to the appropriate 'classes_def.xml' and 'classes.h' files." << '\n';
	}


        EDProduct *p = static_cast<EDProduct *>(cp->New());
        se.prods_.push_back(ProdPair(p, &group->provenance()));
      } else {
        se.prods_.push_back(ProdPair(group->product(), &group->provenance()));
      }
     }

       // Provenance prov(desc);
       // prov.event.status = BranchEntryDescription::CreatorNotRun;
       // prov.event.productID_ = id;
       // provenances.push_back(prov);
       // Provenance & provenance = *provenances.rbegin();
       // se.prods_.push_back(ProdPair(p, &provenance));
      //} else {
      //  se.prods_.push_back(ProdPair(group->product(), &group->provenance()));
      //}
    //}

#if 0
    FDEBUG(11) << "-----Dump start" << endl;
    for(SendProds::iterator pii=se.prods_.begin();pii!=se.prods_.end();++pii)
      std::cout << "Prov:"
	   << " " << pii->desc()->className()
	   << " " << pii->desc()->productID_
	   << endl;      
    FDEBUG(11) << "-----Dump end" << endl;
#endif

    TBuffer rootbuf(TBuffer::kWrite,maxEventSize,msg.eventAddr(),kFALSE);
    RootDebug tracer(10,10);


    TClass* tc = getTClassFor<SendEvent>();
    //TClass* tc = getTClass(typeid(SendEvent));
    int bres = rootbuf.WriteObjectAny(&se,tc);
   switch(bres)
      {
      case 0: // failure
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module could not serialize event: "
	    << e.id();
	  break;
	}
      case 1: // succcess
	break;
      case 2: // truncated result
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module attempted to serialize an event\n"
	    << "that is to big for the allocated buffers: "
	    << e.id();
	  break;
	}
    default: // unknown
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module got an unknown error code\n"
	    << " while attempting to serialize event: "
	    << e.id();
	  break;
	}
      }
     
    return rootbuf.Length();
}

std::auto_ptr<SendJobHeader> readHeader(StreamerInputFile* stream_reader ) 

{

  TClass* desc = getTClassFor<SendJobHeader>();

  InitMsgView* header = (InitMsgView*) stream_reader->startMessage();

  if(header->code() != 0) //INIT Msg
      throw cms::Exception("readHeader","StreamerFileReader")
        << "received wrong message type: expected INIT, got "
        << header->code() << "\n";

  TBuffer xbuf(TBuffer::kRead, header->descLength(),(char*)header->descData(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc));

    if(sd.get()==0) {
        throw cms::Exception("HeaderDecode","DecodeProductList")
          << "Could not read the initial product registry list\n";
    }

    return sd;
}


std::auto_ptr<EventPrincipal> readEvent(const ProductRegistry& pr,
                                        StreamerInputFile* stream_reader)
{
    if (! stream_reader->next() ) 
        return std::auto_ptr<edm::EventPrincipal>();

    EventMsgView* eview = (EventMsgView*) stream_reader->currentRecord();
      
    cout << "Decide event: "
              << eview->event() << " "
              << eview->run() << " "
              << eview->size() << " "
              << eview->eventLength() << " "
              << eview->eventData()
              << endl;

    TClass* tc = getTClassFor<SendEvent>();
    TBuffer xbuf(TBuffer::kRead, eview->eventLength(),(char*) eview->eventData(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendEvent> sd((SendEvent*)xbuf.ReadObjectAny(tc)); 
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




















  // getTClassFor<SendJobHeader>()
  // getTClassFor<SendEvent>()

  // ---------------------------------------

  JobHeaderDecoder::JobHeaderDecoder():
    desc_(getTClassFor<SendJobHeader>()),
    buf_(TBuffer::kRead)
  {
  }

  JobHeaderDecoder::~JobHeaderDecoder() { }

  std::auto_ptr<SendJobHeader>
  JobHeaderDecoder::decodeJobHeader(const InitMsg& msg)
  {
    FDEBUG(6) << "StreamInput: decodeRegistry" << endl;

    if(msg.getCode()!=MsgCode::INIT)
      throw cms::Exception("HeaderDecode","EventStreamerInput")
	<< "received wrong message type: expected INIT, got "
	<< msg.getCode() << "\n";

    // This "SetBuffer" stuff does not appear to work or I don't understand
    // what needs to be done to actually make it go. (JBK)
    //buf_.SetBuffer((char*)msg.data(),msg.getDataSize(),kFALSE);
    TBuffer xbuf(TBuffer::kRead,msg.getDataSize(),(char*)msg.data(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc_));

    if(sd.get()==0) {
	throw cms::Exception("HeaderDecode","DecodeProductList")
	  << "Could not read the initial product registry list\n";
    }

    return sd;
  }

  bool registryIsSubset(const SendJobHeader& sd,
			const ProductRegistry& reg)
  {
    bool rc = true;
    SendDescs::const_iterator i(sd.descs_.begin()),e(sd.descs_.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    FDEBUG(6) << "registryIsSubset: Product List: " << endl;
    for(;i!=e; ++i) {
	typedef edm::ProductRegistry::ProductList plist;
	// the new products must be contained in the old registry
	// form a branchkey from the *i branchdescription,
	// use the productlist from the product registry to locate
	// the branchkey.  If not found, then error in this state
	BranchKey key(*i);
	if(reg.productList().find(key)==reg.productList().end()) {
	  rc = false;
	  break;
#if 0
	  throw cms::Exception("InconsistentRegistry","EventStreamer")
	    << "A new product registry was received during the "
	    << "running state with entries that were not present "
	    << "in the original registry.\n"
	    << "The new type is " << i->className() << "\n";
#endif
	  FDEBUG(6) << "Inconsistent Registry: new type is "
		    << i->className() << "\n";
	}
    }

    return rc;
  }

  void mergeWithRegistry(const SendDescs& descs,
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

  void declareStreamers(const SendDescs& descs)
  {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
	//pi->init();
	string real_name = wrappedClassName(i->className());
	FDEBUG(6) << "declare: " << real_name << endl;
	edm::loadCap(real_name);
    }
  }

  void buildClassCache(const SendDescs& descs)
  {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
	//pi->init();
	string real_name = wrappedClassName(i->className());
	FDEBUG(6) << "BuildReadData: " << real_name << endl;
	edm::doBuildRealData(real_name);
    }
  }

  // ---------------------------------------

  EventDecoder::EventDecoder():
    desc_(getTClassFor<SendEvent>()),
    buf_(TBuffer::kRead)
  {
  }

  EventDecoder::~EventDecoder() { }

  std::auto_ptr<EventPrincipal>
  EventDecoder::decodeEvent(const EventMsg& msg, const ProductRegistry& pr)
  {
    FDEBUG(5) << "Decide event: "
	      << msg.getEventNumber() << " "
	      << msg.getRunNumber() << " "
	      << msg.getTotalSegs() << " "
	      << msg.getWhichSeg() << " "
	      << msg.msgSize() << " "
	      << msg.getDataSize() << " "
	      << msg.data()
	      << endl;

    // This "SetBuffer" stuff does not appear to work or I don't understand
    // what needs to be done to actually make it go. (JBK)
    //buf_.SetBuffer((char*)msg.data(),msg.getDataSize(),kFALSE);
    TBuffer xbuf(TBuffer::kRead,msg.getDataSize(),(char*)msg.data(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendEvent> sd((SendEvent*)xbuf.ReadObjectAny(desc_));

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
	  // << " " << spi->prod()->id()
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

  std::auto_ptr<SendJobHeader> readHeaderFromStream(ifstream& ist)
  {
    JobHeaderDecoder decoder;
    vector<char> regdata(1000*1000);

    int len;
    ist.read((char*)&len,sizeof(int));
    regdata.resize(len);
    ist.read(&regdata[0],len);

    if(!ist)
      throw cms::Exception("ReadHeader","getRegFromFile")
	<< "Could not read the registry information from the test\n"
	<< "event stream file \n";

    edm::InitMsg msg(&regdata[0],len);
    std::auto_ptr<SendJobHeader> p = decoder.decodeJobHeader(msg);
    return p;
  }

  edm::ProductRegistry getRegFromFile(const std::string& filename)
  {
    edm::ProductRegistry pr;
    ifstream ist(filename.c_str(),ios_base::binary | ios_base::in);

    if(!ist)
      {
	throw cms::Exception("ReadRegistry","getRegFromFile")
	  << "cannot open file " << filename;
      }

    std::auto_ptr<SendJobHeader> p = readHeaderFromStream(ist);
    mergeWithRegistry(p->descs_,pr);
    return pr;
  }

  int EventReader::readMessage(Buf& here)
  {
    int len=0;
    ist_->read((char*)&len,sizeof(int));

    if(!*ist_ || len==0) return 0;

    here.resize(len);
    ist_->read(&here[0],len);
    return len;
  }

  std::auto_ptr<EventPrincipal> EventReader::read(const ProductRegistry& prods)
  {
    int len = readMessage(b_);
    //cout << "readMessage done len=" << len << " " << (void*)len << endl;
    if(len==0)
	return std::auto_ptr<edm::EventPrincipal>();

    edm::EventMsg msg(&b_[0],len);
    //cout << "turned into EventMsg" << endl;
    return decoder_.decodeEvent(msg,prods);

  }


}
