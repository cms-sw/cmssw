#ifndef IOPool_Streamer_RecoEventOutputModuleForFU_h
#define IOPool_Streamer_RecoEventOutputModuleForFU_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

#include <sstream>
#include <iomanip>
#include "boost/filesystem.hpp"

#include "../interface/JsonMonitorable.h"
#include "../interface/FastMonitor.h"
#include "../interface/JSONSerializer.h"

#include "FastMonitoringService.h"

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
      smpath_ = runDirectory.string();
      edm::LogInfo("RecoEventOutputModuleForFU") << "Writing .dat files to "
						 << smpath_;
      // create open dir if not already there
      boost::filesystem::path openPath = runDirectory;
      openPath /= "open";
      // do these dirs need to be created?
      bool foundOpenDir = false;
      if (boost::filesystem::is_directory(openPath))
	foundOpenDir = true;
      if (!foundOpenDir) {
	std::cout << "<open> FU dir not found. Creating..."
		  << std::endl;
	boost::filesystem::create_directories(openPath);
      }
    }

  private:
    std::auto_ptr<Consumer> c_;
    std::string stream_label_;
    std::string events_base_filename_;
    std::string baseDir_;
    std::string smpath_;
    std::string jsonDefPath_;
    boost::filesystem::path openDatFilePath_;
    IntJ processed_;
    mutable IntJ accepted_;
    StringJ filelist_;
    boost::shared_ptr<FastMonitor> jsonMonitor_;
    evf::FastMonitoringService *fms_;


  }; //end-of-class-def

  template<typename Consumer>
  RecoEventOutputModuleForFU<Consumer>::RecoEventOutputModuleForFU(edm::ParameterSet const& ps) :
    edm::StreamerOutputModuleBase(ps),
    c_(new Consumer(ps)),
    stream_label_(ps.getParameter<std::string>("@module_label")),
    baseDir_(ps.getUntrackedParameter<std::string>("baseDir","")),
    processed_(0),
    accepted_(0),
    filelist_("")
  {
    initializeStreams();
    
    fms_ = (evf::FastMonitoringService *)(edm::Service<evf::MicroStateService>().operator->());
    jsonDefPath_ = fms_->getOutputDefPath();
    
    processed_.setName("Processed");
    accepted_.setName("Accepted");
    filelist_.setName("Filelist");
    vector<JsonMonitorable*> monParams;
    monParams.push_back(&processed_);
    monParams.push_back(&accepted_);
    monParams.push_back(&filelist_);
    
    jsonMonitor_.reset(new FastMonitor(monParams, jsonDefPath_));
  }
  
  template<typename Consumer>
  RecoEventOutputModuleForFU<Consumer>::~RecoEventOutputModuleForFU() {}

  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::start() const
  {
    std::cout << "RecoEventOutputModuleForFU: start() method " << std::endl;
    
    const std::string initFileName = edm::Service<evf::EvFDaqDirector>()->getInitFilePath(stream_label_);
    
    std::cout << "RecoEventOutputModuleForFU, initializing streams. init stream: " 
	      << initFileName << std::endl;

    c_->setInitMessageFile(initFileName);
    c_->start();
  }
  
  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::stop() const {
    c_->stop();
  }

  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::doOutputHeader(InitMsgBuilder const& init_message) const {
    c_->doOutputHeader(init_message);
  }
   
//______________________________________________________________________________
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
    desc.addUntracked<std::string>("baseDir", "")
        ->setComment("Top level output directory");
    descriptions.add("streamerOutput", desc);
  }

//   template<typename Consumer>
//   void RecoEventOutputModuleForFU<Consumer>::beginRun(edm::RunPrincipal const &run){


//   }

  template<typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::beginLuminosityBlock(edm::LuminosityBlockPrincipal const &ls, edm::ModuleCallingContext const*){
    std::cout << "RecoEventOutputModuleForFU : begin lumi " << std::endl;
	openDatFilePath_ = edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(ls.luminosityBlock(),stream_label_);
	c_->setOutputFile(openDatFilePath_.string());
	filelist_ = openDatFilePath_.filename().string();
  }

  template<typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::endLuminosityBlock(edm::LuminosityBlockPrincipal const &ls, edm::ModuleCallingContext const*){
    std::cout << "RecoEventOutputModuleForFU : end lumi " << std::endl;
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
	jsonMonitor_->snap(false, "");
	const std::string outputJsonNameStream =
	  edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(ls.luminosityBlock(),stream_label_);
	jsonMonitor_->outputFullHistoDataPoint(outputJsonNameStream);
    }

    // reset monitoring params
    accepted_.value() = 0;
    filelist_ = "";
  }

} // end of namespace-edm

#endif
