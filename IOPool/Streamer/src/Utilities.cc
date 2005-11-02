
#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

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
      if(!ans)
	{
	  if((ans = getTClass(typeid(T))))
	    {
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
    
    buf_.SetBuffer((char*)msg.data(),msg.dataSize(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendJobHeader> sd((SendJobHeader*)buf_.ReadObjectAny(desc_));

    if(sd.get()==0)
      {
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
    for(;i!=e;++i)
      {
	typedef edm::ProductRegistry::ProductList plist;
	// the new products must be contained in the old registry
	// form a branchkey from the *i branchdescription,
	// use the productlist from the product registry to locate
	// the branchkey.  If not found, then error in this state
	BranchKey key(*i);
	if(reg.productList().find(key)==reg.productList().end())
	  {
	    rc = false;
	    break;
#if 0
	    throw cms::Exception("InconsistentRegistry","EventStreamer")
	      << "A new product registry was received during the "
	      << "running state with entries that were not present "
	      << "in the original registry.\n"
	      << "The new type is " << i->fullClassName_ << "\n";
#endif
	    FDEBUG(6) << "Inconsistent Registry: new type is "
		      << i->fullClassName_ << "\n";
	  }
      }

    return rc;
  }

  void mergeWithRegistry(const SendJobHeader& sd,
			 ProductRegistry& reg)
  {
    SendDescs::const_iterator i(sd.descs_.begin()),e(sd.descs_.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    FDEBUG(6) << "mergeWithRegistry: Product List: " << endl;
    for(;i!=e;++i)
      {
	reg.copyProduct(*i);
	FDEBUG(6) << "StreamInput prod = " << i->fullClassName_ << endl;
      }

    // not needed any more
    // fillStreamers(*pr_);
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
    buf_.SetBuffer((char*)msg.data(),msg.dataSize(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendEvent> sd((SendEvent*)buf_.ReadObjectAny(desc_));

    if(sd.get()==0)
      {
	throw cms::Exception("EventInput","Read")
	  << "got a null event from input stream\n";
      }

    FDEBUG(5) << "Got event: " << sd->id_ << endl;

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
	if(spi->prod()==0)
	  throw cms::Exception("NoData","EmptyProduct");
	if(spi->desc()==0)
	  throw cms::Exception("NoData","EmptyDesc");

	FDEBUG(5) << "Prov:"
	     << " " << spi->desc()->fullClassName_
	  // << " " << spi->prod()->id()
	     << " " << spi->desc()->productID_
	     << " " << spi->prov()->productID_
	     << endl;

	auto_ptr<EDProduct> 
	  aprod(const_cast<EDProduct*>(spi->prod()));
	auto_ptr<BranchEntryDescription> 
	  aedesc(const_cast<BranchEntryDescription*>(spi->prov()));
	auto_ptr<BranchDescription> 
	  adesc(const_cast<BranchDescription*>(spi->desc()));

	auto_ptr<Provenance> aprov(new Provenance);
	aprov->event   = *(aedesc.get());
	aprov->product = *(adesc.get());
	
	FDEBUG(10) << "addgroup next" << endl;
	ep->addGroup(auto_ptr<Group>(new Group(aprod,aprov)));
	FDEBUG(10) << "addgroup done" << endl;
	spi->clear();
      }

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
    mergeWithRegistry(*p,pr);
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
