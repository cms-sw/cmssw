#include "DQMOffline/CalibTracker/plugins/SiStripPopConSourceHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMStoreReader.h"

/**
  @class SiStripNoisesDQMService
  @author M. De Mattia, S. Dutta, D. Giordano

  @popcon::PopConSourceHandler to extract noise values the DQM as bad and write in the database.
*/
class SiStripPopConNoisesHandlerFromDQM : public SiStripPopConSourceHandler<SiStripNoises>, private SiStripDQMStoreReader
{
public:
  explicit SiStripPopConNoisesHandlerFromDQM(const edm::ParameterSet& iConfig);
  virtual ~SiStripPopConNoisesHandlerFromDQM();
  // interface methods: implemented in template
  SiStripNoises* getObj();
private:
  edm::FileInPath fp_;
  std::string MEDir_;
};

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripPopConNoisesHandlerFromDQM::SiStripPopConNoisesHandlerFromDQM(const edm::ParameterSet& iConfig)
  : SiStripPopConSourceHandler<SiStripNoises>(iConfig)
  , SiStripDQMStoreReader(iConfig)
  , fp_{iConfig.getUntrackedParameter<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))}
  , MEDir_{iConfig.getUntrackedParameter<std::string>("ME_DIR", "DQMData")}
{
  edm::LogInfo("SiStripNoisesDQMService") <<  "[SiStripNoisesDQMService::SiStripNoisesDQMService]";
}

SiStripPopConNoisesHandlerFromDQM::~SiStripPopConNoisesHandlerFromDQM()
{
  edm::LogInfo("SiStripNoisesDQMService") <<  "[SiStripNoisesDQMService::~SiStripNoisesDQMService]";
}

SiStripNoises* SiStripPopConNoisesHandlerFromDQM::getObj()
{
  std::cout << "SiStripNoisesDQMService::readNoises" << std::endl;

  openRequestedFile();

  std::cout << "[readBadComponents]: opened requested file" << std::endl;

  std::unique_ptr<SiStripNoises> obj{new SiStripNoises{}};

  SiStripDetInfoFileReader reader(fp_.fullPath());

  // dqmStore_->cd(iConfig_.getUntrackedParameter<std::string>("ME_DIR"));
  dqmStore_->cd();

  uint32_t stripsPerApv = 128;

  // Get the full list of monitoring elements
  // const std::vector<MonitorElement*>& MEs = dqmStore_->getAllContents(iConfig_.getUntrackedParameter<std::string>("ME_DIR","DQMData"));

  // Take a copy of the vector
  std::vector<MonitorElement*> MEs = dqmStore_->getAllContents(MEDir_);
  // Remove all but the MEs we are using
  std::vector<MonitorElement*>::iterator newEnd = remove_if(MEs.begin(), MEs.end(), StringNotMatch("CMSubNoisePerStrip__det__"));
  MEs.erase(newEnd, MEs.end());

  // The histograms are one per DetId, loop on all the DetIds and extract the corresponding histogram
  for ( const auto& detInfo : reader.getAllData() ) {

    SiStripNoises::InputVector theSiStripVector;

    // Take the path for each DetId and build the complete path + histogram name


    // MonitorElement * mE = getModuleHistogram(detInfo.first, "PedsPerStrip");
    MonitorElement * mE{nullptr};
    std::string MEname("CMSubNoisePerStrip__det__"+std::to_string(detInfo.first));
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
      if( histo != 0 ) {
        // Read the noise from the histograms
        uint32_t nBinsX = histo->GetXaxis()->GetNbins();

        if( nBinsX != stripsPerApv*(detInfo.second.nApvs) ) {
          std::cout << "ERROR: number of bin = " << nBinsX << " != number of strips = " << stripsPerApv*(detInfo.second.nApvs) << std::endl;
        }

        // std::cout << "Bin 0 = " << histo->GetBinContent(0) << std::endl;
        // TH1 bins start from 1, 0 is the underflow, nBinsX+1 the overflow.
        for( uint32_t iBin = 1; iBin <= nBinsX; ++iBin ) {
          // encode the pedestal value and put it in the vector (push_back)
          obj->setData( histo->GetBinContent(iBin), theSiStripVector );
        }
      } else {
        std::cout << "ERROR: histo = " << histo << std::endl;
      }
    } else {
      std::cout << "ERROR: ME = " << mE << std::endl;
    }
    // If the ME was absent fill the vector with 50 (we want a high noise to avoid these modules being considered good by mistake)
    if( theSiStripVector.empty() ) {
      for(unsigned short j=0; j<128*detInfo.second.nApvs; ++j){
        obj->setData(50, theSiStripVector);
      }
    }

    if ( ! obj->put(detInfo.first, theSiStripVector) )
      edm::LogError("SiStripNoisesFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
  dqmStore_->cd();

  return obj.release();
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
using SiStripPopConNoisesDQM = popcon::PopConAnalyzer<SiStripPopConNoisesHandlerFromDQM>;
DEFINE_FWK_MODULE(SiStripPopConNoisesDQM);
