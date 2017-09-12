#include "DQMOffline/CalibTracker/plugins/SiStripDQMPopConSourceHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
/**
  @class SiStripNoisesDQMService
  @author M. De Mattia, S. Dutta, D. Giordano

  @popcon::PopConSourceHandler to extract noise values the DQM as bad and write in the database.
*/
class SiStripPopConNoisesHandlerFromDQM : public SiStripDQMPopConSourceHandler<SiStripNoises>
{
public:
  explicit SiStripPopConNoisesHandlerFromDQM(const edm::ParameterSet& iConfig);
  ~SiStripPopConNoisesHandlerFromDQM() override;
  // interface methods: implemented in template
  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) override;
  SiStripNoises* getObj() const override;
private:
  edm::FileInPath fp_;
  std::string MEDir_;
  SiStripNoises m_obj;
};

#include "DQMServices/Core/interface/MonitorElement.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripPopConNoisesHandlerFromDQM::SiStripPopConNoisesHandlerFromDQM(const edm::ParameterSet& iConfig)
  : SiStripDQMPopConSourceHandler<SiStripNoises>(iConfig)
  , fp_{iConfig.getUntrackedParameter<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))}
  , MEDir_{iConfig.getUntrackedParameter<std::string>("ME_DIR", "DQMData")}
{
  edm::LogInfo("SiStripNoisesDQMService") <<  "[SiStripNoisesDQMService::SiStripNoisesDQMService]";
}

SiStripPopConNoisesHandlerFromDQM::~SiStripPopConNoisesHandlerFromDQM()
{
  edm::LogInfo("SiStripNoisesDQMService") <<  "[SiStripNoisesDQMService::~SiStripNoisesDQMService]";
}

void SiStripPopConNoisesHandlerFromDQM::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter& getter)
{
  std::cout << "SiStripNoisesDQMService::readNoises" << std::endl;

  m_obj = SiStripNoises();

  SiStripDetInfoFileReader reader(fp_.fullPath());

  // getter.cd(iConfig_.getUntrackedParameter<std::string>("ME_DIR"));
  getter.cd();

  uint32_t stripsPerApv = 128;

  // Get the full list of monitoring elements
  // const std::vector<MonitorElement*>& MEs = getter.getAllContents(iConfig_.getUntrackedParameter<std::string>("ME_DIR","DQMData"));

  // Take a copy of the vector
  std::vector<MonitorElement*> MEs = getter.getAllContents(MEDir_);
  // Remove all but the MEs we are using
  MEs.erase(std::remove_if(MEs.begin(), MEs.end(),
        [] ( const MonitorElement* ME ) -> bool {
          return std::string::npos == ME->getName().find("CMSubNoisePerStrip__det__");
        }), MEs.end());

  // The histograms are one per DetId, loop on all the DetIds and extract the corresponding histogram
  for ( const auto& detInfo : reader.getAllData() ) {

    SiStripNoises::InputVector theSiStripVector;

    // Take the path for each DetId and build the complete path + histogram name


    // MonitorElement * mE = getModuleHistogram(detInfo.first, "PedsPerStrip");
    const MonitorElement * mE{nullptr};
    std::string MEname("CMSubNoisePerStrip__det__"+std::to_string(detInfo.first));
    for ( const MonitorElement* ime : MEs ) {
      if( ime->getName() == MEname ) {
        mE = ime;
        break;
      }
    }

    // find( MEs.begin(), MEs.end(), "PedsPerStrip__det__"+std::to_string(detInfo.first), findMEbyName() );
    // MonitorElement * mE = *(find( MEs.begin(), MEs.end(), findMEbyName("PedsPerStrip__det__"+std::to_string(detInfo.first)) ));
    if ( mE ) {
      TH1F* histo = mE->getTH1F();
      if( histo != nullptr ) {
        // Read the noise from the histograms
        uint32_t nBinsX = histo->GetXaxis()->GetNbins();

        if( nBinsX != stripsPerApv*(detInfo.second.nApvs) ) {
          std::cout << "ERROR: number of bin = " << nBinsX << " != number of strips = " << stripsPerApv*(detInfo.second.nApvs) << std::endl;
        }

        // std::cout << "Bin 0 = " << histo->GetBinContent(0) << std::endl;
        // TH1 bins start from 1, 0 is the underflow, nBinsX+1 the overflow.
        for( uint32_t iBin = 1; iBin <= nBinsX; ++iBin ) {
          // encode the pedestal value and put it in the vector (push_back)
          m_obj.setData( histo->GetBinContent(iBin), theSiStripVector );
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
        m_obj.setData(50, theSiStripVector);
      }
    }

    if ( ! m_obj.put(detInfo.first, theSiStripVector) )
      edm::LogError("SiStripNoisesFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
  getter.cd();
}

SiStripNoises* SiStripPopConNoisesHandlerFromDQM::getObj() const
{
  return new SiStripNoises(m_obj);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/CalibTracker/plugins/SiStripPopConDQMEDHarvester.h"
using SiStripPopConNoisesDQM = SiStripPopConDQMEDHarvester<SiStripPopConNoisesHandlerFromDQM>;
DEFINE_FWK_MODULE(SiStripPopConNoisesDQM);
