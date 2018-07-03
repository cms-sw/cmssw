// Original Author:  Anne-Marie Magnan
//         Created:  2010/02/25
//

#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"

//
// Class declaration
//
namespace sistrip {

  class SpyExtractRunModule : public edm::EDAnalyzer
  {
  public:

    explicit SpyExtractRunModule(const edm::ParameterSet&);
    ~SpyExtractRunModule() override;

  private:

    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override;

    //check when the current run changes
    const bool updateRun(const uint32_t aRun);

    //name of the output file containing the run number
    //get it from the input file
    std::string fileName_;

    //tag of spydata run number collection
    edm::InputTag runTag_;
    edm::EDGetTokenT<uint32_t> runToken_;

    //cache of the current and previous run number
    uint32_t currentRun_;
    uint32_t previousRun_;

    //error counter for number of times the run number changes
    uint32_t errCounter_;

  };
}//namespace

using edm::LogError;
using edm::LogWarning;
using edm::LogInfo;
//
// Constructors and destructor
//
namespace sistrip {

  SpyExtractRunModule::SpyExtractRunModule(const edm::ParameterSet& iConfig)
    : fileName_(iConfig.getParameter<std::string>("OutputTextFile")),
      runTag_(iConfig.getParameter<edm::InputTag>("RunNumberTag")),
      currentRun_(0),
      previousRun_(0),
      errCounter_(0)
  {
    runToken_ = consumes<uint32_t>(runTag_);
  }


  SpyExtractRunModule::~SpyExtractRunModule() {

  }

  void SpyExtractRunModule::beginJob()
  {
    currentRun_ = 0;
    previousRun_ = 0;
    errCounter_ = 0;

  }

  void SpyExtractRunModule::analyze(const edm::Event& aEvt, const edm::EventSetup& aSetup)
  {

    static bool lFirstEvent = true;
    edm::Handle<uint32_t> lRun;
    aEvt.getByToken( runToken_, lRun ); 

    const bool isUpdated = updateRun(*lRun);

    if (isUpdated && !lFirstEvent){
      edm::LogError("SpyExtractRunModule") << " -- Run number changed for event : " << aEvt.id().event() 
					   << " (id().run() = " << aEvt.id().run()
					   << ") from " << previousRun_ << " to " << currentRun_
					   << std::endl;
    }


    lFirstEvent = false;

  }


  void SpyExtractRunModule::endJob() {

    //save global run number in text file in local directory
    //output loginfo with number of errors
    //or throw exception ?


    if (errCounter_ == 1){
      edm::LogInfo("SiStripSpyExtractRun") << " -- Writting run number " << currentRun_ 
					   << " into file " << fileName_
					   << std::endl;
      std::ofstream lOutFile;
      lOutFile.open(fileName_.c_str(),std::ios::out);
      if (!lOutFile.is_open()) {
	edm::LogError("SiStripSpyExtractRun")  << " -- Cannot open file : " << fileName_ << " for writting run number " 
					       << currentRun_
					       << std::endl;
      }
      else {
	lOutFile << currentRun_ << std::endl;
	lOutFile.close();
      }

    }
    else {
      edm::LogError("SiStripSpyExtractRun")  << " -- Number of times the run number changed in this job = " << errCounter_
					     << ", currentRun = " << currentRun_ 
					     << ", previousRun = " << previousRun_
					     << std::endl;
    }


  }

  const bool SpyExtractRunModule::updateRun(const uint32_t aRun) {
    if (aRun != currentRun_){
      previousRun_ = currentRun_;
      currentRun_ = aRun;
      errCounter_++;
      return true;
    }
    return false;

  }

}//namespace

#include "FWCore/Framework/interface/MakerMacros.h"
typedef sistrip::SpyExtractRunModule SiStripSpyExtractRunModule;
DEFINE_FWK_MODULE(SiStripSpyExtractRunModule);
