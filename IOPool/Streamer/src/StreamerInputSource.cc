#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h" 

#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"


#include "zlib.h"


#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <string>
#include <iostream>

namespace edm {
  namespace {
    const int init_size = 1024*1024;
  }

  std::string StreamerInputSource::processName_;
  unsigned int StreamerInputSource::protocolVersion_;


  StreamerInputSource::StreamerInputSource(
                    ParameterSet const& pset,
                    InputSourceDescription const& desc):
    InputSource(pset, desc),
    holder_(),
    lbp_(),
    rp_(),
    processHistoryID_(),
    tc_(getTClass(typeid(SendEvent))),
    dest_(init_size),
    xbuf_(TBuffer::kRead, init_size)
  {
  }

  StreamerInputSource::~StreamerInputSource() {}

  // ---------------------------------------

  void
  StreamerInputSource::mergeIntoRegistry(SendDescs const& descs,
			 ProductRegistry& reg) {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    FDEBUG(6) << "mergeIntoRegistry: Product List: " << std::endl;
    for(; i != e; ++i) {
	reg.copyProduct(*i);
	FDEBUG(6) << "StreamInput prod = " << i->className() << std::endl;
    }

    // not needed any more
    // fillStreamers(*pr_);
  }

  void
  StreamerInputSource::mergeWithRegistry(SendDescs const& descs) {

    mergeIntoRegistry(descs, productRegistryUpdate());

    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    std::string processName;

    //process name is already stored in deserializer, if for Protocol version 4 and above.
    //From the INIT Message itself.
    
    if ( protocolVersion_ > 3) {
           processName = processName_;
    } else { 
    	if (i != e) {
       		processName = i->processName();
	}
    	for (; i != e; ++i) {
		if(processName != i->processName()) {
	   	throw cms::Exception("MultipleProcessNames")
	      		<< "at least two different process names ('"
	      		<< processName
	      		<< "', '"
	      		<< i->processName()
	      		<< "' found in JobHeader. We can only support one.";
		}
    	}
    }

    FDEBUG(10) << "StreamerInputSource::mergeWithRegistry :"<<processName<<std::endl; 

    edm::ProcessHistory ph;
    ph.reserve(1);
    ph.push_back(edm::ProcessConfiguration(processName, ParameterSetID(), ReleaseVersion(), PassID()));
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph);
    setProcessHistoryID(ph.id());

  }

  void
  StreamerInputSource::declareStreamers(SendDescs const& descs) {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        std::string const real_name = wrappedClassName(i->className());
        FDEBUG(6) << "declare: " << real_name << std::endl;
        loadCap(real_name);
    }
  }


  void
  StreamerInputSource::buildClassCache(SendDescs const& descs) { 
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        std::string const real_name = wrappedClassName(i->className());
        FDEBUG(6) << "BuildReadData: " << real_name << std::endl;
        doBuildRealData(real_name);
    }
  }

  void 
  StreamerInputSource::saveTriggerNames(InitMsgView const* header) {

    ParameterSet trigger_pset;
    std::vector<std::string> paths;
    header->hltTriggerNames(paths);
    trigger_pset.addParameter<Strings>("@trigger_paths", paths);
    pset::Registry* psetRegistry = pset::Registry::instance();
    psetRegistry->insertMapped(trigger_pset);
  }

  boost::shared_ptr<RunPrincipal>
  StreamerInputSource::readRun_() {
    if (holder_.get() == 0) holder_ = read();
    if (holder_.get() == 0) {
      return boost::shared_ptr<RunPrincipal>();
    }
    return rp_;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  StreamerInputSource::readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp) {
    if (holder_.get() == 0) holder_ = read();
    if (holder_.get() == 0 || rp->run() != rp_->run()) {
      return boost::shared_ptr<LuminosityBlockPrincipal>();
    }
    return lbp_;
  }

  std::auto_ptr<EventPrincipal>
  StreamerInputSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    if (holder_.get() == 0) holder_ = read();
    if (holder_.get() == 0 ||
	lbp->runNumber() != rp_->run() ||
        lbp->luminosityBlock() != lbp_->luminosityBlock()) {
      return std::auto_ptr<EventPrincipal>(0);
    }
    return holder_;
  }

  /**
   * Deserializes the specified init message into a SendJobHeader object
   * (which is related to the product registry).
   */
  std::auto_ptr<SendJobHeader>
  StreamerInputSource::deserializeRegistry(InitMsgView const& initView)
  {
    if(initView.code() != Header::INIT)
      throw cms::Exception("StreamTranslation","Registry deserialization error")
        << "received wrong message type: expected INIT, got "
        << initView.code() << "\n";

    //Get the process name and store if for Protocol version 4 and above.
    if (initView.protocolVersion() > 3) {

         processName_ = initView.processName();
         protocolVersion_ = initView.protocolVersion();

         FDEBUG(10) << "StreamerInputSource::deserializeRegistry processName = "<< processName_<< std::endl;
         FDEBUG(10) << "StreamerInputSource::deserializeRegistry protocolVersion_= "<< protocolVersion_<< std::endl;
    }
    
    TClass* desc = getTClass(typeid(SendJobHeader));

    TBuffer xbuf(TBuffer::kRead, initView.descLength(),
                 (char*)initView.descData(),kFALSE);
    RootDebug tracer(10,10);
    std::auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc));

    if(sd.get()==0) {
        throw cms::Exception("StreamTranslation","Registry deserialization error")
          << "Could not read the initial product registry list\n";
    }

    return sd;  
  }

  /**
   * Deserializes the specified event message into an EventPrincipal object.
   */
  std::auto_ptr<EventPrincipal>
  StreamerInputSource::deserializeEvent(EventMsgView const& eventView)
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
         << std::endl;
    // uncompress if we need to
    // 78 was a dummy value (for no uncompressed) - should be 0 for uncompressed
    // need to get rid of this when 090 MTCC streamers are gotten rid of
    unsigned long origsize = eventView.reserved();
    unsigned long dest_size; //(should be >= eventView.reserved() )

    if(eventView.reserved() != 78 && eventView.reserved() != 0)
    {
      // compressed
      dest_size = uncompressBuffer((unsigned char*)eventView.eventData(),
                                   eventView.eventLength(), dest_, origsize);
    }
    else // not compressed
    {
      // we need to copy anyway the buffer as we are using dest in xbuf
      dest_size = eventView.eventLength();
      dest_.resize(dest_size);
      unsigned char* pos = (unsigned char*) &dest_[0];
      unsigned char* from = (unsigned char*) eventView.eventData();
      std::copy(from,from+dest_size,pos);
    }
    //TBuffer xbuf(TBuffer::kRead, dest_size,
    //             (char*) &dest[0],kFALSE);
    //TBuffer xbuf(TBuffer::kRead, eventView.eventLength(),
    //             (char*) eventView.eventData(),kFALSE);
    xbuf_.Reset();
    xbuf_.SetBuffer(&dest_[0],dest_size,kFALSE);
    RootDebug tracer(10,10);

    std::auto_ptr<SendEvent> sd((SendEvent*)xbuf_.ReadObjectAny(tc_));

    if(sd.get()==0) {
        throw cms::Exception("StreamTranslation","Event deserialization error")
          << "got a null event from input stream\n";
    }

    FDEBUG(5) << "Got event: " << sd->id_ << " " << sd->prods_.size() << std::endl;
    if(!rp_ || rp_->run() != sd->id_.run()) {
	rp_ = boost::shared_ptr<RunPrincipal>(
          new RunPrincipal(sd->id_.run(),
			   sd->time_,
			   Timestamp::invalidTimestamp(),
			   productRegistry(),
			   processConfiguration()));
        lbp_.reset();
    }
    if(!lbp_ || lbp_->luminosityBlock() != eventView.lumi()) {
      lbp_ = boost::shared_ptr<LuminosityBlockPrincipal>(
        new LuminosityBlockPrincipal(eventView.lumi(),
				     sd->time_,
				     Timestamp::invalidTimestamp(),
				     productRegistry(),
				     rp_,
				     processConfiguration()));
    }
    std::auto_ptr<EventPrincipal> ep(new EventPrincipal(sd->id_,
                                                   sd->time_,
                                                   productRegistry(),
                                                   lbp_,
                                                   processConfiguration(),
                                                   true,
						   EventAuxiliary::Any,
						   EventPrincipal::invalidBunchXing,
						   EventPrincipal::invalidStoreNumber,
						   processHistoryID_));
    // no process name list handling

    SendProds::iterator spi(sd->prods_.begin()),spe(sd->prods_.end());
    for(; spi != spe; ++spi) {
        FDEBUG(10) << "check prodpair" << std::endl;
        if(spi->prov() == 0)
          throw cms::Exception("StreamTranslation","EmptyProvenance");
        if(spi->desc() == 0)
          throw cms::Exception("StreamTranslation","EmptyDesc");
        FDEBUG(5) << "Prov:"
             << " " << spi->desc()->className()
             << " " << spi->desc()->productInstanceName()
             << " " << spi->desc()->productID()
             << " " << spi->prov()->productID_
             << std::endl;

        if(spi->prod()==0) {
            FDEBUG(10) << "Product is null" << std::endl;
            continue;
            throw cms::Exception("StreamTranslation","EmptyProduct");
        }

        std::auto_ptr<EDProduct>
          aprod(const_cast<EDProduct*>(spi->prod()));
        std::auto_ptr<BranchEntryDescription>
          aedesc(const_cast<BranchEntryDescription*>(spi->prov()));
        std::auto_ptr<BranchDescription>
          adesc(const_cast<BranchDescription*>(spi->desc()));

        std::auto_ptr<Provenance> aprov(new Provenance(*(adesc.get()), *(aedesc.get())));
        if(aprov->isPresent()) {
          FDEBUG(10) << "addgroup next " << aprov->productID() << std::endl;
          FDEBUG(10) << "addgroup next " << aprov->event().productID_ << std::endl;
          ep->addGroup(aprod, aprov);
          FDEBUG(10) << "addgroup done" << std::endl;
        } else {
          FDEBUG(10) << "addgroup empty next " << aprov->productID() << std::endl;
          FDEBUG(10) << "addgroup empty next " << aprov->event().productID_ 
                                               << std::endl;
          ep->addGroup(aprov);
          FDEBUG(10) << "addgroup empty done" << std::endl;
        }
        spi->clear();
    }

    FDEBUG(10) << "Size = " << ep->numEDProducts() << std::endl;

    return ep;     
  }

  /**
   * Uncompresses the data in the specified input buffer into the
   * specified output buffer.  The inputSize should be set to the size
   * of the compressed data in the inputBuffer.  The expectedFullSize should
   * be set to the original size of the data (before compression).
   * Returns the actual size of the uncompressed data.
   * Errors are reported by throwing exceptions.
   */
  unsigned int
  StreamerInputSource::uncompressBuffer(unsigned char *inputBuffer,
				       unsigned int inputSize,
				       std::vector<unsigned char> &outputBuffer,
				       unsigned int expectedFullSize)
  {
    unsigned long origSize = expectedFullSize;
    unsigned long uncompressedSize = expectedFullSize;
    FDEBUG(1) << "Uncompress: original size = " << origSize
              << ", compressed size = " << inputSize
              << std::endl;
    outputBuffer.resize(origSize);
    int ret = uncompress(&outputBuffer[0], &uncompressedSize,
                         inputBuffer, inputSize); // do not need compression level
    //std::cout << "unCompress Return value: " << ret << " Okay = " << Z_OK << std::endl;
    if(ret == Z_OK) {
        // check the length against original uncompressed length
        FDEBUG(10) << " original size = " << origSize << " final size = " 
                   << uncompressedSize << std::endl;
        if(origSize != uncompressedSize) {
            std::cerr << "deserializeEvent: Problem with uncompress, original size = "
                 << origSize << " uncompress size = " << uncompressedSize << std::endl;
            // we throw an error and return without event! null pointer
            throw cms::Exception("StreamDeserialization","Uncompression error")
              << "mismatch event lengths should be" << origSize << " got "
              << uncompressedSize << "\n";
        }
    } else {
        // we throw an error and return without event! null pointer
        std::cerr << "deserializeEvent: Problem with uncompress, return value = "
             << ret << std::endl;
        throw cms::Exception("StreamDeserialization","Uncompression error")
            << "Error code = " << ret << "\n ";
    }

    return (unsigned int) uncompressedSize;
  }
}
