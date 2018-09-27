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
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"

//
// Class declaration
//
namespace sistrip {

  class SpyIdentifyRunsModule : public edm::EDAnalyzer
  {
  public:

    explicit SpyIdentifyRunsModule(const edm::ParameterSet&);
    ~SpyIdentifyRunsModule() override;

  private:

    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override;

    void writeRunInFile(const unsigned int aRunNumber);
 
    //name of the output file containing the run numbers
    //of spy runs
    std::string fileName_;
    std::ofstream outFile_;

    //tag of spydata source collection
    edm::InputTag srcTag_;
    edm::EDGetTokenT<FEDRawDataCollection> srcToken_;
    uint32_t prevRun_;

  };
}//namespace

using edm::LogError;
using edm::LogWarning;
using edm::LogInfo;
//
// Constructors and destructor
//
namespace sistrip {

  SpyIdentifyRunsModule::SpyIdentifyRunsModule(const edm::ParameterSet& iConfig)
    : fileName_(iConfig.getParameter<std::string>("OutputTextFile")),
      srcTag_(iConfig.getParameter<edm::InputTag>("InputProductLabel")),
      prevRun_(0)
  {
    srcToken_ = consumes<FEDRawDataCollection>(srcTag_);
  }


  SpyIdentifyRunsModule::~SpyIdentifyRunsModule() {

  }

  void SpyIdentifyRunsModule::beginJob()
  {
    outFile_.open(fileName_.c_str(),std::ios::out);
    if (!outFile_.is_open()) {
      edm::LogError("SiStripSpyIdentifyRuns")  << " -- Cannot open file : " << fileName_ << " for writting." 
					       << std::endl;
      edm::LogInfo("SiStripSpyIdentifyRuns")  << " *** SPY RUNS *** "<< std::endl;

    }
    else {
      outFile_ << " *** SPY RUNS *** " << std::endl;
    }
  }

  void SpyIdentifyRunsModule::analyze(const edm::Event& aEvt, const edm::EventSetup& aSetup)
  {

    //static bool lFirstEvent = true;
    //if (!lFirstEvent) return;
    uint32_t lRunNum = aEvt.id().run();
    if (lRunNum == prevRun_) return;

    edm::Handle<FEDRawDataCollection> lHandle;
    aEvt.getByToken( srcToken_, lHandle ); 
    const FEDRawDataCollection& buffers = *lHandle;

    for (unsigned int iFed(FEDNumbering::MINSiStripFEDID);
	 iFed <= FEDNumbering::MAXSiStripFEDID;
	 iFed++)
      {

	//retrieve FED raw data for given FED 
	const FEDRawData& input = buffers.FEDData( static_cast<int>(iFed) );
	//check on FEDRawData pointer and size
	if ( !input.data() ||!input.size() ) continue;
          
	//construct FEDBuffer
	std::unique_ptr<sistrip::FEDSpyBuffer> buffer;
	try {
	  buffer.reset(new sistrip::FEDSpyBuffer(input.data(),input.size()));
	} catch (const cms::Exception& e) { 
	  edm::LogWarning("SiStripSpyIdentifyRuns")
	    << "Exception caught when creating FEDSpyBuffer object for FED " << iFed << ": " << e.what();
	  //if (!(buffer->readoutMode() == READOUT_MODE_SPY)) break;
	  std::string lErrStr = e.what();
	  if (lErrStr.find("Buffer is not from spy channel")!=lErrStr.npos) break;
	  else {
	    writeRunInFile(lRunNum);
	    break;
	  }
	} // end of buffer reset try.
        edm::LogWarning("SiStripSpyIdentifyRuns")
	  << " -- this is a spy file, run " << lRunNum << std::endl;
	writeRunInFile(lRunNum);
	break;
      }
    //lFirstEvent = false;
    prevRun_ = lRunNum;

  }

  void SpyIdentifyRunsModule::writeRunInFile(const unsigned int aRunNumber){
    if (!outFile_.is_open()) {
      edm::LogInfo("SiStripSpyIdentifyRuns") << aRunNumber
					     << std::endl;
    }
    else {
      outFile_ << aRunNumber  << std::endl;
    }
  }

  void SpyIdentifyRunsModule::endJob() {

    //save global run number in text file in local directory
    //output loginfo with number of errors
    //or throw exception ?
    if (outFile_.is_open()) outFile_.close();

  }


}//namespace

#include "FWCore/Framework/interface/MakerMacros.h"
typedef sistrip::SpyIdentifyRunsModule SiStripSpyIdentifyRuns;
DEFINE_FWK_MODULE(SiStripSpyIdentifyRuns);
