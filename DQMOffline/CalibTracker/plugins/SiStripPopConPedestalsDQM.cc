#include "DQMOffline/CalibTracker/plugins/SiStripPopConSourceHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMStoreReader.h"

/**
  @class SiStripPopConPedestalsHandlerFromDQM
  @author M. De Mattia, S. Dutta, D. Giordano

  @popcon::PopConSourceHandler to extract pedestal values the DQM as bad and write in the database.
*/
class SiStripPopConPedestalsHandlerFromDQM : public SiStripPopConSourceHandler<SiStripPedestals>, private SiStripDQMStoreReader
{
public:
  explicit SiStripPopConPedestalsHandlerFromDQM(const edm::ParameterSet& iConfig);
  virtual ~SiStripPopConPedestalsHandlerFromDQM();
  // interface methods: implemented in template
  SiStripPedestals* getObj() const;
private:
  edm::FileInPath fp_;
  std::string MEDir_;
};

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripPopConPedestalsHandlerFromDQM::SiStripPopConPedestalsHandlerFromDQM(const edm::ParameterSet& iConfig)
  : SiStripPopConSourceHandler<SiStripPedestals>(iConfig)
  , SiStripDQMStoreReader(iConfig)
  , fp_{iConfig.getUntrackedParameter<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))}
  , MEDir_{iConfig.getUntrackedParameter<std::string>("ME_DIR", "DQMData")}
{
  edm::LogInfo("SiStripPedestalsDQMService") <<  "[SiStripPedestalsDQMService::SiStripPedestalsDQMService]";
}

SiStripPopConPedestalsHandlerFromDQM::~SiStripPopConPedestalsHandlerFromDQM()
{
  edm::LogInfo("SiStripPedestalsDQMService") <<  "[SiStripPedestalsDQMService::~SiStripPedestalsDQMService]";
}

SiStripPedestals* SiStripPopConPedestalsHandlerFromDQM::getObj() const
{
  std::cout << "SiStripPedestalsDQMService::readPedestals" << std::endl;

  openRequestedFile();

  std::cout << "[readBadComponents]: opened requested file" << std::endl;

  std::unique_ptr<SiStripPedestals> obj{new SiStripPedestals{}};

  SiStripDetInfoFileReader reader(fp_.fullPath());

  // dqmStore_->cd(iConfig_.getUntrackedParameter<std::string>("ME_DIR"));
  dqmStore_->cd();

  uint32_t stripsPerApv = 128;

  // Get the full list of monitoring elements
  // const std::vector<MonitorElement*>& MEs = dqmStore_->getAllContents(iConfig_.getUntrackedParameter<std::string>("ME_DIR","DQMData"));

  // Take a copy of the vector
  std::vector<MonitorElement*> MEs = dqmStore_->getAllContents(MEDir_);
  // Remove all but the MEs we are using
  std::vector<MonitorElement*>::iterator newEnd = remove_if(MEs.begin(), MEs.end(), StringNotMatch("PedsPerStrip__det__"));
  MEs.erase(newEnd, MEs.end());

  // The histograms are one per DetId, loop on all the DetIds and extract the corresponding histogram
  for ( const auto& detInfo : reader.getAllData() ) {
    SiStripPedestals::InputVector theSiStripVector;

    // Take the path for each DetId and build the complete path + histogram name

    // MonitorElement * mE = getModuleHistogram(detInfo.first, "PedsPerStrip");
    const MonitorElement* mE{nullptr};
    std::string MEname("PedsPerStrip__det__"+std::to_string(detInfo.first));
    for ( const MonitorElement* ime : MEs ) {
      if( ime->getName() == MEname ) {
        mE = ime;
        break;
      }
    }

    // find( MEs.begin(), MEs.end(), "PedsPerStrip__det__"+boost::lexical_cast<std::string>(detInfo.first), findMEbyName() );
    // MonitorElement * mE = *(find( MEs.begin(), MEs.end(), findMEbyName("PedsPerStrip__det__"+boost::lexical_cast<std::string>(detInfo.first)) ));

    if ( mE ) {
      TH1F* histo = mE->getTH1F();
      if ( histo ) {
        // Read the pedestals from the histograms
        uint32_t nBinsX = histo->GetXaxis()->GetNbins();

        if ( nBinsX != stripsPerApv*(detInfo.second.nApvs) ) {
          std::cout << "ERROR: number of bin = " << nBinsX << " != number of strips = " << stripsPerApv*(detInfo.second.nApvs) << std::endl;
        }

        // std::cout << "Bin 0 = " << histo->GetBinContent(0) << std::endl;
        // TH1 bins start from 1, 0 is the underflow, nBinsX+1 the overflow.
        for ( uint32_t iBin = 1; iBin <= nBinsX; ++iBin ) {
          // encode the pedestal value and put it in the vector (push_back)
          obj->setData( histo->GetBinContent(iBin), theSiStripVector );
        }
      } else {
        std::cout << "ERROR: histo = " << histo << std::endl;
      }
    } else {
      std::cout << "ERROR: ME = " << mE << std::endl;
    }
    // If the ME was absent fill the vector with 0
    if ( theSiStripVector.empty() ) {
      for(unsigned short j=0; j<128*detInfo.second.nApvs; ++j){
        obj->setData(0, theSiStripVector);
      }
    }

    if ( ! obj->put(detInfo.first, theSiStripVector) )
      edm::LogError("SiStripPedestalsFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
  dqmStore_->cd();

  return obj.release();
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
using SiStripPopConPedestalsDQM = popcon::PopConAnalyzer<SiStripPopConPedestalsHandlerFromDQM>;
DEFINE_FWK_MODULE(SiStripPopConPedestalsDQM);
