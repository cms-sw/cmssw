#include "DQMOffline/CalibTracker/plugins/SiStripPedestalsDQMService.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <string>
#include <sstream>
#include <cctype>
#include <time.h>
#include <boost/cstdint.hpp>

using namespace std;

SiStripPedestalsDQMService::SiStripPedestalsDQMService(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  // SiStripCondObjBuilderBase<SiStripPedestals>::SiStripCondObjBuilderBase(iConfig),
  SiStripBaseServiceFromDQM<SiStripPedestals>::SiStripBaseServiceFromDQM(iConfig),
  iConfig_(iConfig),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat")))
{
  obj_ = 0;
  edm::LogInfo("SiStripPedestalsDQMService") <<  "[SiStripPedestalsDQMService::SiStripPedestalsDQMService]";
}

SiStripPedestalsDQMService::~SiStripPedestalsDQMService()
{
  edm::LogInfo("SiStripPedestalsDQMService") <<  "[SiStripPedestalsDQMService::~SiStripPedestalsDQMService]";
}

void SiStripPedestalsDQMService::readPedestals()
{
  std::cout << "SiStripPedestalsDQMService::readPedestals" << std::endl;

  openRequestedFile();

  std::cout << "[readBadComponents]: opened requested file" << std::endl;

  obj_= new SiStripPedestals;

  SiStripDetInfoFileReader reader(fp_.fullPath());

  // dqmStore_->cd(iConfig_.getUntrackedParameter<std::string>("ME_DIR"));
  dqmStore_->cd();

  uint32_t stripsPerApv = 128;

  // Get the full list of monitoring elements
  // const std::vector<MonitorElement*>& MEs = dqmStore_->getAllContents(iConfig_.getUntrackedParameter<std::string>("ME_DIR","DQMData"));

  // Take a copy of the vector
  std::vector<MonitorElement*> MEs = dqmStore_->getAllContents(iConfig_.getUntrackedParameter<std::string>("ME_DIR","DQMData"));
  // Remove all but the MEs we are using
  std::vector<MonitorElement*>::iterator newEnd = remove_if(MEs.begin(), MEs.end(), StringNotMatch("PedsPerStrip__det__"));
  MEs.erase(newEnd, MEs.end());


  // The histograms are one per DetId, loop on all the DetIds and extract the corresponding histogram
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo> DetInfos  = reader.getAllData();
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>::const_iterator it = DetInfos.begin(); it != DetInfos.end(); ++it) {


    SiStripPedestals::InputVector theSiStripVector;

    // Take the path for each DetId and build the complete path + histogram name


    // MonitorElement * mE = getModuleHistogram(it->first, "PedsPerStrip");


    MonitorElement * mE = 0;
    std::string MEname("PedsPerStrip__det__"+boost::lexical_cast<string>(it->first));
    for( std::vector<MonitorElement*>::const_iterator MEit = MEs.begin();
         MEit != MEs.end(); ++MEit ) {
      if( (*MEit)->getName() == MEname ) {
        mE = *MEit;
        break;
      }
    }

    // find( MEs.begin(), MEs.end(), "PedsPerStrip__det__"+boost::lexical_cast<string>(it->first), findMEbyName() );
    // MonitorElement * mE = *(find( MEs.begin(), MEs.end(), findMEbyName("PedsPerStrip__det__"+boost::lexical_cast<string>(it->first)) ));

    if( mE != 0 ) {
      TH1F* histo = mE->getTH1F();

      if( histo != 0 ) {

        // Read the pedestals from the histograms
        uint32_t nBinsX = histo->GetXaxis()->GetNbins();

        if( nBinsX != stripsPerApv*(it->second.nApvs) ) {
          std::cout << "ERROR: number of bin = " << nBinsX << " != number of strips = " << stripsPerApv*(it->second.nApvs) << std::endl;
        }

        // std::cout << "Bin 0 = " << histo->GetBinContent(0) << std::endl;

        // TH1 bins start from 1, 0 is the underflow, nBinsX+1 the overflow.
        for( uint32_t iBin = 1; iBin <= nBinsX; ++iBin ) {
          // encode the pedestal value and put it in the vector (push_back)
          obj_->setData( histo->GetBinContent(iBin), theSiStripVector );
        }
      }
      else {
        std::cout << "ERROR: histo = " << histo << std::endl;
      }
    }
    else {
      std::cout << "ERROR: ME = " << mE << std::endl;
    }
    // If the ME was absent fill the vector with 0
    if( theSiStripVector.empty() ) {
      for(unsigned short j=0; j<128*it->second.nApvs; ++j){
        obj_->setData(0, theSiStripVector);
      }
    }

    if ( ! obj_->put(it->first, theSiStripVector) )
      edm::LogError("SiStripPedestalsFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
  dqmStore_->cd();
}
