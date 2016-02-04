
#include "IOPool/Streamer/interface/Utilities.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/Messages.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "FWCore/Utilities/interface/Exception.h"

//#include <typeinfo>
#include <fstream>

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
  JobHeaderDecoder::decodeJobHeader(InitMsg const& msg)
  {
    FDEBUG(6) << "StreamInput: decodeRegistry" << std::endl;

    if(msg.getCode()!=MsgCode::INIT)
      throw cms::Exception("HeaderDecode","EventStreamerInput")
	<< "received wrong message type: expected INIT, got "
	<< msg.getCode() << "\n";

    // This "SetBuffer" stuff does not appear to work or I don't understand
    // what needs to be done to actually make it go. (JBK)
    //buf_.SetBuffer((char*)msg.data(),msg.getDataSize(),kFALSE);
    TBufferFile xbuf(TBuffer::kRead,msg.getDataSize(),(char*)msg.data(),kFALSE);
    RootDebug tracer(10,10);
    std::auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc_));

    if(sd.get()==0) {
	throw cms::Exception("HeaderDecode","DecodeProductList")
	  << "Could not read the initial product registry list\n";
    }

    return sd;
  }

  bool registryIsSubset(SendJobHeader const& sd,
			ProductRegistry const& reg)
  {
    bool rc = true;
    SendDescs::const_iterator i(sd.descs().begin()),e(sd.descs().end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    FDEBUG(6) << "registryIsSubset: Product List: " << std::endl;
    for(;i != e; ++i) {
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

  bool registryIsSubset(SendJobHeader const& sd,
			SendJobHeader const& ref)
  {
    bool rc = true;
    SendDescs::const_iterator i(sd.descs().begin()),e(sd.descs().end());

    FDEBUG(6) << "registryIsSubset: Product List: " << std::endl;
    for(;i != e; ++i) {
	// the new products must be contained in the old registry
	// form a branchkey from the *i branchdescription,
	// use the productlist from the product registry to locate
	// the branchkey.  If not found, then error in this state
	BranchKey key(*i);
        // look for matching in ref
	FDEBUG(9) << "Looking for " << i->className() << "\n";
        SendDescs::const_iterator iref(ref.descs().begin()),eref(ref.descs().end());
        bool found = false;
        for(; iref != eref; ++iref) {
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

  std::auto_ptr<SendJobHeader> readHeaderFromStream(ifstream& ist)
  {
    JobHeaderDecoder decoder;
    std::vector<char> regdata(1000*1000);

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
    ist.seekg(0, std::ios::beg);
    ist.read(&regdata[0], headerSize);

    //if(!ist)
    //  throw cms::Exception("ReadHeader","getRegFromFile")
    //	<< "Could not read the registry information from the test\n"
    //	<< "event stream file \n";

    //edm::InitMsg msg(&regdata[0],len);
    //std::auto_ptr<SendJobHeader> p = decoder.decodeJobHeader(msg);
    InitMsgView initView(&regdata[0]);
    std::auto_ptr<SendJobHeader> p = StreamerInputSource::deserializeRegistry(initView);
    return p;
  }

  edm::ProductRegistry getRegFromFile(std::string const& filename)
  {
    edm::ProductRegistry pr;
    ifstream ist(filename.c_str(), std::ios_base::binary | std::ios_base::in);

    if(!ist)
      {
	throw cms::Exception("ReadRegistry","getRegFromFile")
	  << "cannot open file " << filename;
      }

    std::auto_ptr<SendJobHeader> p = readHeaderFromStream(ist);
    StreamerInputSource::mergeIntoRegistry(*p, pr, false);
    return pr;
  }

}
