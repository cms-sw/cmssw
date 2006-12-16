
#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/StreamTranslator.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Common/interface/Wrapper.h"

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

  bool registryIsSubset(const SendJobHeader& sd,
			const SendJobHeader& ref)
  {
    bool rc = true;
    SendDescs::const_iterator i(sd.descs_.begin()),e(sd.descs_.end());

    FDEBUG(6) << "registryIsSubset: Product List: " << endl;
    for(;i!=e; ++i) {
	// the new products must be contained in the old registry
	// form a branchkey from the *i branchdescription,
	// use the productlist from the product registry to locate
	// the branchkey.  If not found, then error in this state
	BranchKey key(*i);
        // look for matching in ref
	FDEBUG(9) << "Looking for " << i->className() << "\n";
        SendDescs::const_iterator iref(ref.descs_.begin()),eref(ref.descs_.end());
        bool found = false;
        for(;iref!=eref; ++iref) {
	  FDEBUG(9) << "testing against " << iref->className() << "\n";
          BranchKey refkey(*iref);
          if(key == refkey) {
            found = true;
	    FDEBUG(9) << "found!" << "\n";
            break;
          }
        }
	if(!found) {
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

    //int len;
    //ist.read((char*)&len,sizeof(int));
    //regdata.resize(len);
    //ist.read(&regdata[0],len);
    ist.read(&regdata[0], sizeof(HeaderView));

    if (ist.eof() || (unsigned int)ist.gcount() < sizeof(HeaderView)  )
    {
          throw cms::Exception("ReadHeader","getRegFromFile")
                << "No file exists or Empty file encountered:\n";
    }

    HeaderView head(&regdata[0]);
    uint32 code = head.code();
    if (code != Header::INIT) /** Not an init message should return ******/
    {
      throw cms::Exception("ReadHeader","getRegFromFile")
                << "Expecting an init Message at start of file\n";
    }

    uint32 headerSize = head.size();
    //Bring the pointer at start of Start Message/start of file
    ist.seekg(0, ios::beg);
    ist.read(&regdata[0], headerSize);

    //if(!ist)
    //  throw cms::Exception("ReadHeader","getRegFromFile")
    //	<< "Could not read the registry information from the test\n"
    //	<< "event stream file \n";

    //edm::InitMsg msg(&regdata[0],len);
    //std::auto_ptr<SendJobHeader> p = decoder.decodeJobHeader(msg);
    InitMsgView initView(&regdata[0]);
    std::auto_ptr<SendJobHeader> p = StreamTranslator::deserializeRegistry(initView);
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
    if(len==0)
	return std::auto_ptr<edm::EventPrincipal>();

    edm::EventMsg msg(&b_[0],len);
    return decoder_.decodeEvent(msg,prods);

  }


}
