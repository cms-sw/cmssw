#ifndef IOPool_Streamer_RecoEventOutputModuleForFU_h
#define IOPool_Streamer_RecoEventOutputModuleForFU_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

#include <sstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <zlib.h>

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"


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
    //virtual void beginRun(edm::RunPrincipal const&, edm::ModuleCallingContext const*);
    virtual void beginJob();
    virtual void beginLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*);
    virtual void endLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*);

  private:
    std::auto_ptr<Consumer> c_;
    std::string stream_label_;
    boost::filesystem::path openDatFilePath_;
    boost::filesystem::path openDatChecksumFilePath_;
    IntJ processed_;
    mutable IntJ accepted_;
    IntJ errorEvents_; 
    IntJ retCodeMask_; 
    StringJ filelist_;
    IntJ filesize_; 
    StringJ inputFiles_;
    IntJ fileAdler32_; 
    StringJ transferDestination_; 
    boost::shared_ptr<FastMonitor> jsonMonitor_;
    evf::FastMonitoringService *fms_;
    DataPointDefinition outJsonDef_;
    unsigned char* outBuf_=0;
    bool readAdler32Check_=false;


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
    filesize_(0),
    inputFiles_(),
    fileAdler32_(1),
    transferDestination_(),
    outBuf_(new unsigned char[1024*1024])
  {
    std::string baseRunDir = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
    readAdler32Check_ =  edm::Service<evf::EvFDaqDirector>()->outputAdler32Recheck();
    LogDebug("RecoEventOutputModuleForFU") << "writing .dat files to -: " << baseRunDir;
    // create open dir if not already there
    edm::Service<evf::EvFDaqDirector>()->createRunOpendirMaybe();

    //replace hltOutoputA with stream if the HLT menu uses this convention
    std::string testPrefix="hltOutput";
    if (stream_label_.find(testPrefix)==0) 
      stream_label_=std::string("stream")+stream_label_.substr(testPrefix.size());

    if (stream_label_.find("_")!=std::string::npos) {
      throw cms::Exception("RecoEventOutputModuleForFU")
        << "Underscore character is reserved can not be used for stream names in FFF, but was detected in stream name -: " << stream_label_;
    }


    std::string stream_label_lo = stream_label_;
    boost::algorithm::to_lower(stream_label_lo);
    auto streampos = stream_label_lo.rfind("stream");
    if (streampos !=0 && streampos!=std::string::npos)
      throw cms::Exception("RecoEventOutputModuleForFU")
        << "stream (case-insensitive) sequence was found in stream suffix. This is reserved and can not be used for names in FFF based HLT, but was detected in stream name";

    fms_ = (evf::FastMonitoringService *)(edm::Service<evf::MicroStateService>().operator->());
    
    processed_.setName("Processed");
    accepted_.setName("Accepted");
    errorEvents_.setName("ErrorEvents");
    retCodeMask_.setName("ReturnCodeMask");
    filelist_.setName("Filelist");
    filesize_.setName("Filesize");
    inputFiles_.setName("InputFiles");
    fileAdler32_.setName("FileAdler32");
    transferDestination_.setName("TransferDestination");

    outJsonDef_.setDefaultGroup("data");
    outJsonDef_.addLegendItem("Processed","integer",DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("Accepted","integer",DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("ErrorEvents","integer",DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("ReturnCodeMask","integer",DataPointDefinition::BINARYOR);
    outJsonDef_.addLegendItem("Filelist","string",DataPointDefinition::MERGE);
    outJsonDef_.addLegendItem("Filesize","integer",DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("InputFiles","string",DataPointDefinition::CAT);
    outJsonDef_.addLegendItem("FileAdler32","integer",DataPointDefinition::ADLER32);
    outJsonDef_.addLegendItem("TransferDestination","string",DataPointDefinition::SAME);
    std::stringstream tmpss,ss;
    tmpss << baseRunDir << "/open/" << "output_" << getpid() << ".jsd";
    ss << baseRunDir << "/" << "output_" << getpid() << ".jsd";
    std::string outTmpJsonDefName = tmpss.str();
    std::string outJsonDefName = ss.str();

    edm::Service<evf::EvFDaqDirector>()->lockInitLock();
    struct stat   fstat;
    if (stat (outJsonDefName.c_str(), &fstat) != 0) { //file does not exist
      LogDebug("RecoEventOutputModuleForFU") << "writing output definition file -: " << outJsonDefName;
      std::string content;
      JSONSerializer::serialize(&outJsonDef_,content);
      FileIO::writeStringToFile(outTmpJsonDefName, content);
      boost::filesystem::rename(outTmpJsonDefName,outJsonDefName);
    }
    edm::Service<evf::EvFDaqDirector>()->unlockInitLock();

    jsonMonitor_.reset(new FastMonitor(&outJsonDef_,true));
    jsonMonitor_->setDefPath(outJsonDefName);
    jsonMonitor_->registerGlobalMonitorable(&processed_,false);
    jsonMonitor_->registerGlobalMonitorable(&accepted_,false);
    jsonMonitor_->registerGlobalMonitorable(&errorEvents_,false);
    jsonMonitor_->registerGlobalMonitorable(&retCodeMask_,false);
    jsonMonitor_->registerGlobalMonitorable(&filelist_,false);
    jsonMonitor_->registerGlobalMonitorable(&filesize_,false);
    jsonMonitor_->registerGlobalMonitorable(&inputFiles_,false);
    jsonMonitor_->registerGlobalMonitorable(&fileAdler32_,false);
    jsonMonitor_->registerGlobalMonitorable(&transferDestination_,false);
    jsonMonitor_->commit(nullptr);

  }
  
  template<typename Consumer>
  RecoEventOutputModuleForFU<Consumer>::~RecoEventOutputModuleForFU() {}

  template<typename Consumer>
  void
  RecoEventOutputModuleForFU<Consumer>::start() const
  {
    const std::string openInitFileName = edm::Service<evf::EvFDaqDirector>()->getOpenInitFilePath(stream_label_);
    edm::LogInfo("RecoEventOutputModuleForFU") << "start() method, initializing streams. init stream -: "  
	                                       << openInitFileName;
    c_->setInitMessageFile(openInitFileName);
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

    const std::string openIniFileName = edm::Service<evf::EvFDaqDirector>()->getOpenInitFilePath(stream_label_);
    struct stat istat;
    stat(openIniFileName.c_str(), &istat);
    //read back file to check integrity of what was written
    off_t readInput=0;
    uint32_t adlera=1,adlerb=0;
    FILE *src = fopen(openIniFileName.c_str(),"r");
    while (readInput<istat.st_size)
    {
      size_t toRead=  readInput+1024*1024 < istat.st_size ? 1024*1024 : istat.st_size-readInput;
      fread(outBuf_,toRead,1,src);
      cms::Adler32((const char*)outBuf_,toRead,adlera,adlerb);
      readInput+=toRead;
    }
    fclose(src);
    uint32_t adler32c = (adlerb << 16) | adlera;
    if (adler32c != c_->get_adler32_ini()) {
      throw cms::Exception("RecoEventOutputModuleForFU") << "Checksum mismatch of ini file -: " << openIniFileName
                           << " expected:" << c_->get_adler32_ini() << " obtained:" << adler32c;
    }
    else {
      edm::LogWarning("RecoEventOutputModuleForFU") << "Ini file checksum -: "<< stream_label_ << " " << adler32c;
      boost::filesystem::rename(openIniFileName,edm::Service<evf::EvFDaqDirector>()->getInitFilePath(stream_label_));
    }
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
  void RecoEventOutputModuleForFU<Consumer>::beginJob()
  {
    //get stream transfer destination
    transferDestination_ = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations(stream_label_);
  }


  template<typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::beginLuminosityBlock(edm::LuminosityBlockPrincipal const &ls, edm::ModuleCallingContext const*)
  {
    //edm::LogInfo("RecoEventOutputModuleForFU") << "begin lumi";
    openDatFilePath_ = edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(ls.luminosityBlock(),stream_label_);
    openDatChecksumFilePath_ = edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(ls.luminosityBlock(),stream_label_);
    c_->setOutputFile(openDatFilePath_.string());
    filelist_ = openDatFilePath_.filename().string();
  }

  template<typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::endLuminosityBlock(edm::LuminosityBlockPrincipal const &ls, edm::ModuleCallingContext const*)
  {
    //edm::LogInfo("RecoEventOutputModuleForFU") << "end lumi";
    long filesize=0;
    fileAdler32_.value() = c_->get_adler32();
    c_->closeOutputFile();
    bool abortFlag = false;
    processed_.value() = fms_->getEventsProcessedForLumi(ls.luminosityBlock(),&abortFlag);

    if (abortFlag) {
        edm::LogInfo("RecoEventOutputModuleForFU") << "output suppressed";
        return;
    }
    
    if(processed_.value()!=0) {

      //lock
      FILE *des = edm::Service<evf::EvFDaqDirector>()->maybeCreateAndLockFileHeadForStream(ls.luminosityBlock(),stream_label_);

      std::string deschecksum = edm::Service<evf::EvFDaqDirector>()->getMergedDatChecksumFilePath(ls.luminosityBlock(), stream_label_);

      struct stat istat;
      FILE * cf = NULL;
      uint32_t mergedAdler32=1;
      //get adler32 accumulated checksum for the merged file
      if (!stat(deschecksum.c_str(), &istat)) {
          if (istat.st_size) {
            cf = fopen(deschecksum.c_str(),"r");
            if (!cf) throw cms::Exception("RecoEventOutputModuleForFU") << "Unable to open checksum file -: " << deschecksum.c_str();
            fscanf(cf,"%u",&mergedAdler32);
            fclose(cf);
          }
          else edm::LogWarning("RecoEventOutputModuleForFU") << "Checksum file size is empty -: "<< deschecksum.c_str();
      }

      FILE *src = fopen(openDatFilePath_.string().c_str(),"r");

      stat(openDatFilePath_.string().c_str(), &istat);
      off_t readInput=0;
      uint32_t adlera=1;
      uint32_t adlerb=0;
      while (readInput<istat.st_size) {
          size_t toRead=  readInput+1024*1024 < istat.st_size ? 1024*1024 : istat.st_size-readInput;
          fread(outBuf_,toRead,1,src);
          fwrite(outBuf_,toRead,1,des);
          if (readAdler32Check_)
            cms::Adler32((const char*)outBuf_,toRead,adlera,adlerb);
          readInput+=toRead;
          filesize+=toRead;
      }

      //if(des != 0 && src !=0){
      //	while((b=fgetc(src))!= EOF){
      //	  fputc((unsigned char)b,des);
      //    filesize++;
      //	}
      //}

      //write new string representation of the checksum value
      cf = fopen(deschecksum.c_str(),"w");
      if (!cf) throw cms::Exception("RecoEventOutputModuleForFU") << "Unable to open or rewind checksum file for writing -:" << deschecksum.c_str();

      //write adler32 combine to checksum file 
      mergedAdler32 = adler32_combine(mergedAdler32,fileAdler32_.value(),filesize);

      fprintf(cf,"%u",mergedAdler32);
      fclose(cf);

      edm::Service<evf::EvFDaqDirector>()->unlockAndCloseMergeStream();
      fclose(src);

      if (readAdler32Check_ && ((adlerb << 16) | adlera) != fileAdler32_.value()) {

        throw cms::Exception("RecoEventOutputModuleForFU") << "Adler32 checksum mismatch after reading file -: " 
                                                           << openDatFilePath_.string() <<" in LS " << ls.luminosityBlock() << std::endl;
      }

    } else {
      //return if not in empty lumisectio mode
      if (!edm::Service<evf::EvFDaqDirector>()->emptyLumisectionMode())
        return;
      filelist_ = "";
      fileAdler32_.value()=-1;
    }

    //remove file
    remove(openDatFilePath_.string().c_str());
    filesize_=filesize;

    jsonMonitor_->snap(ls.luminosityBlock());
    const std::string outputJsonNameStream =
      edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(ls.luminosityBlock(),stream_label_);
    jsonMonitor_->outputFullJSON(outputJsonNameStream,ls.luminosityBlock());

    // reset monitoring params
    accepted_.value() = 0;
    filelist_ = "";
  }

} // end of namespace-edm

#endif
