#ifndef IOPool_Streamer_RecoEventOutputModuleForFU_h
#define IOPool_Streamer_RecoEventOutputModuleForFU_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

#include <sstream>
#include <iomanip>
#include "boost/filesystem.hpp"

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/plugins/FastMonitoringService.h"


namespace evf {
  template<typename Consumer>
  class RecoEventOutputModuleForFU : public edm::StreamerOutputModuleBase {
    
    /** Consumers are supposed to provide
	void doOutputHeader(InitMsgBuilder const& init_message)
	void doOutputEvent(EventMsgBuilder const& msg)
	void start()
	void stop()
	static void fillDescription(ParameterSetDescription&)
    **/
    
  public:
    explicit RecoEventOutputModuleForFU(edm::ParameterSet const& ps);  
    virtual ~RecoEventOutputModuleForFU();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
  private:
    virtual void start() const;
    virtual void stop() const;
    virtual void doOutputHeader(InitMsgBuilder const& init_message) const;
    virtual void doOutputEvent(EventMsgBuilder const& msg) const;
    //    virtual void beginRun(edm::RunPrincipal const&);
    virtual void beginLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*);
    virtual void endLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*);

    void initializeStreams() {
      // find run dir
      boost::filesystem::path runDirectory(
					   edm::Service<evf::EvFDaqDirector>()->findCurrentRunDir());
      datapath_ = runDirectory.string();
      edm::LogInfo("RecoEventOutputModuleForFU") << "Writing .dat files to "
						 << datapath_;
      // create open dir if not already there
      boost::filesystem::path openPath = runDirectory;
      openPath /= "open";
      // do these dirs need to be created?
      bool foundOpenDir = false;
      if (boost::filesystem::is_directory(openPath))
	foundOpenDir = true;
      if (!foundOpenDir) {
	edm::LogInfo("RecoEventOutputModuleForFU") << "<open> FU dir not found. Creating...";
	boost::filesystem::create_directories(openPath);
      }
    }

  private:
    std::auto_ptr<Consumer> c_;
    std::string stream_label_;
    std::string events_base_filename_;
    std::string datapath_;
    boost::filesystem::path openDatFilePath_;
    IntJ processed_;
    mutable IntJ accepted_;
    IntJ errorEvents_; 
    IntJ retCodeMask_; 
    StringJ filelist_;
    StringJ inputFiles_;
    boost::shared_ptr<FastMonitor> jsonMonitor_;
    evf::FastMonitoringService *fms_;
    DataPointDefinition outJsonDef_;


  }; //end-of-class-def

  template<typename Consumer>
  RecoEventOutputModuleForFU<Consumer>::RecoEventOutputModuleForFU(edm::ParameterSet const& ps) :
    edm::StreamerOutputModuleBase(ps),
    c_(new Consumer(ps)),
    stream_label_(ps.getParameter<std::string>("@module_label")),
    processed_(0),
    accepted_(0),
    errorEvents_(0),
    retCodeMask_(0),
    filelist_(),
    inputFiles_()
  {
    initializeStreams();
    fms_ = (evf::FastMonitoringService *)(edm::Service<evf::MicroStateService>().operator->());
    
    processed_.setName("Processed");
    accepted_.setName("Accepted");
    errorEvents_.setName("ErrorEvents");
    retCodeMask_.setName("ReturnCodeMask");
    filelist_.setName("Filelist");
    inputFiles_.setName("InputFiles");

    outJsonDef_.setDefaultGroup("data");
    outJsonDef_.addLegendItem("Processed","integer",DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("Accepted","integer",DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("ErrorEvents","integer",DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("ReturnCodeMask","integer",DataPointDefinition::BINARYOR);
    outJsonDef_.addLegendItem("Filelist","string",DataPointDefinition::MERGE);
    outJsonDef_.addLegendItem("InputFiles","string",DataPointDefinition::CAT);
    std::stringstream ss;
    ss << edm::Service<evf::EvFDaqDirector>()->baseRunDir() << "/" << "output_" << getpid() << ".jsd";
    std::string outJsonDefName = ss.str();

    edm::Service<evf::EvFDaqDirector>()->lockInitLock();
    struct stat   fstat;
    if (stat (outJsonDefName.c_str(), &fstat) != 0) { //file does not exist
      edm::LogInfo("RecoEventOutputModuleForFU") << " writing output definition file " << outJsonDefName;
      std::string content;
      JSONSerializer::serialize(&outJsonDef_,content);
      FileIO::writeStringToFile(outJsonDefName, content);
    }
    edm::Service<evf::EvFDaqDirector>()->unlockInitLock();

    jsonMonitor_.reset(new FastMonitor(&outJsonDef_,true));
    jsonMonitor_->setDefPath(outJsonDefName);
    jsonMonitor_->registerGlobalMonitorable(&processed_,false);
    jsonMonitor_->registerGlobalMonitorable(&accepted_,false);
    jsonMonitor_->registerGlobalMonitorable(&errorEvents_,false);
    jsonMonitor_->registerGlobalMonitorable(&retCodeMask_,false);
    jsonMonitor_->registerGlobalMonitorable(&filelist_,false);
    jsonMonitor_->registerGlobalMonitorable(&inputFiles_,false);
    jsonMonitor_->commit(nullptr);
  }
  
  template<typename Consumer>
  RecoEventOutputModuleForFU<Consumer>::~RecoEventOutputModuleForFU() {}

  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::start() const
  {
    const std::string initFileName = edm::Service<evf::EvFDaqDirector>()->getInitFilePath(stream_label_);
    edm::LogInfo("RecoEventOutputModuleForFU") << "RecoEventOutputModuleForFU start() method, initializing streams. init stream: " 
	                                       << initFileName;
    c_->setInitMessageFile(initFileName);
    c_->start();
  }
  
  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::stop() const
  {
    c_->stop();
  }

  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::doOutputHeader(InitMsgBuilder const& init_message) const
  {
    c_->doOutputHeader(init_message);
  }
   
  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::doOutputEvent(EventMsgBuilder const& msg) const {
	accepted_.value()++;
    c_->doOutputEvent(msg); // You can't use msg in RecoEventOutputModuleForFU after this point
  }

  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::StreamerOutputModuleBase::fillDescription(desc);
    Consumer::fillDescription(desc);
    descriptions.add("streamerOutput", desc);
  }

  template<typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::beginLuminosityBlock(edm::LuminosityBlockPrincipal const &ls, edm::ModuleCallingContext const*)
  {
    //edm::LogInfo("RecoEventOutputModuleForFU") << "begin lumi";
    openDatFilePath_ = edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(ls.luminosityBlock(),stream_label_);
    c_->setOutputFile(openDatFilePath_.string());
    filelist_ = openDatFilePath_.filename().string();
  }

  template<typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::endLuminosityBlock(edm::LuminosityBlockPrincipal const &ls, edm::ModuleCallingContext const*)
  {
    //edm::LogInfo("RecoEventOutputModuleForFU") << "end lumi";
    c_->closeOutputFile();
    processed_.value() = fms_->getEventsProcessedForLumi(ls.luminosityBlock());
    if(processed_.value()!=0){
      int b;
      // move dat file to one level up - this is VERRRRRY inefficient, come up with a smarter idea

      FILE *des = edm::Service<evf::EvFDaqDirector>()->maybeCreateAndLockFileHeadForStream(ls.luminosityBlock(),stream_label_);
      FILE *src = fopen(openDatFilePath_.string().c_str(),"r");
      if(des != 0 && src !=0){
	while((b=fgetc(src))!= EOF){
	  fputc((unsigned char)b,des);
	}
      }

      edm::Service<evf::EvFDaqDirector>()->unlockAndCloseMergeStream();
      fclose(src);
    }
    //remove file
    remove(openDatFilePath_.string().c_str());

    // output jsn file
    if(processed_.value()!=0){
	jsonMonitor_->snap(ls.luminosityBlock());
	const std::string outputJsonNameStream =
	  edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(ls.luminosityBlock(),stream_label_);
	jsonMonitor_->outputFullJSON(outputJsonNameStream,ls.luminosityBlock());
    }

    // reset monitoring params
    accepted_.value() = 0;
    filelist_ = "";
  }

} // end of namespace-edm

#endif
