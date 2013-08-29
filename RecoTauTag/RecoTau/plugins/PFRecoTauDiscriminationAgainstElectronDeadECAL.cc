
/** \class PFRecoTauDiscriminationAgainstElectronDeadECAL
 *
 * Flag tau candidates reconstructed near dead ECAL channels,
 * in order to reduce e -> tau fakes not rejected by anti-e MVA discriminator
 *
 * The motivation for this flag is this presentation:
 *   https://indico.cern.ch/getFile.py/access?contribId=0&resId=0&materialId=slides&confId=177223
 *
 * \authors Lauri Andreas Wendland,
 *          Christian Veelken
 *
 *
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include <TMath.h>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronDeadECAL : public PFTauDiscriminationProducerBase  
{
 public:
  explicit PFRecoTauDiscriminationAgainstElectronDeadECAL(const edm::ParameterSet& cfg)
    : PFTauDiscriminationProducerBase(cfg),
      moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      isFirstEvent_(true)
  {
    minStatus_ = cfg.getParameter<uint32_t>("minStatus");
    dR_ = cfg.getParameter<double>("dR");
  }
  ~PFRecoTauDiscriminationAgainstElectronDeadECAL() {}

  void beginEvent(const edm::Event& evt, const edm::EventSetup& es)
  {
    updateBadTowers(es);
  }

  double discriminate(const PFTauRef& pfTau)
  {
    //std::cout << "<PFRecoTauDiscriminationAgainstElectronDeadECAL::discriminate>:" << std::endl;
    //std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
    //std::cout << " #badTowers = " << badTowers_.size() << std::endl;
    double discriminator = 1.;
    for ( std::vector<towerInfo>::const_iterator badTower = badTowers_.begin();
	  badTower != badTowers_.end(); ++badTower ) {
      if ( deltaR(badTower->eta_, badTower->phi_, pfTau->eta(), pfTau->phi()) < dR_ ) discriminator = 0.;
    }
    //std::cout << "--> discriminator = " << discriminator << std::endl;
    return discriminator;
  }

 private:
  void updateBadTowers(const edm::EventSetup& es) 
  {
    // NOTE: modified version of SUSY CAF code
    //         UserCode/SusyCAF/plugins/SusyCAF_EcalDeadChannels.cc
    const uint32_t channelStatusId = es.get<EcalChannelStatusRcd>().cacheIdentifier();
    const uint32_t caloGeometryId  = es.get<CaloGeometryRecord>().cacheIdentifier();
    const uint32_t idealGeometryId = es.get<IdealGeometryRecord>().cacheIdentifier();
    
    if ( !isFirstEvent_ && channelStatusId == channelStatusId_cache_ && caloGeometryId == caloGeometryId_cache_ && idealGeometryId == idealGeometryId_cache_  ) return;

    edm::ESHandle<EcalChannelStatus> channelStatus;    
    es.get<EcalChannelStatusRcd>().get(channelStatus);
    channelStatusId_cache_ = channelStatusId;  

    edm::ESHandle<CaloGeometry> caloGeometry;         
    es.get<CaloGeometryRecord>().get(caloGeometry);
    caloGeometryId_cache_ = caloGeometryId;  

    edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap;
    es.get<IdealGeometryRecord>().get(ttMap);
    idealGeometryId_cache_ = idealGeometryId;

    std::map<uint32_t,unsigned> nBadCrystals, maxStatus;
    std::map<uint32_t,double> sumEta, sumPhi;
    
    loopXtals<EBDetId>(nBadCrystals, maxStatus, sumEta, sumPhi, channelStatus.product(), caloGeometry.product(), ttMap.product());
    loopXtals<EEDetId>(nBadCrystals, maxStatus, sumEta, sumPhi, channelStatus.product(), caloGeometry.product(), ttMap.product());
    
    badTowers_.clear();
    for ( std::map<uint32_t, unsigned>::const_iterator it = nBadCrystals.begin(); 
	  it != nBadCrystals.end(); ++it ) {
      uint32_t key = it->first;
      badTowers_.push_back(towerInfo(key, it->second, maxStatus[key], sumEta[key]/it->second, sumPhi[key]/it->second));
    }

    isFirstEvent_ = false;
  }
  
  template <class Id>
  void loopXtals(std::map<uint32_t, unsigned>& nBadCrystals,
		 std::map<uint32_t, unsigned>& maxStatus,
		 std::map<uint32_t, double>& sumEta,
		 std::map<uint32_t, double>& sumPhi ,
		 const EcalChannelStatus* channelStatus,
		 const CaloGeometry* caloGeometry,
		 const EcalTrigTowerConstituentsMap* ttMap) const 
  {
    // NOTE: modified version of SUSY CAF code
    //         UserCode/SusyCAF/plugins/SusyCAF_EcalDeadChannels.cc
    for ( int i = 0; i < Id::kSizeForDenseIndexing; ++i ) {
      Id id = Id::unhashIndex(i);  
      if ( id == Id(0) ) continue;
      EcalChannelStatusMap::const_iterator it = channelStatus->getMap().find(id.rawId());
      unsigned status = ( it == channelStatus->end() ) ? 
	0 : (it->getStatusCode() & statusMask_);
      if ( status >= minStatus_ ) {
	const GlobalPoint& point = caloGeometry->getPosition(id);
	uint32_t key = ttMap->towerOf(id);
	maxStatus[key] = TMath::Max(status, maxStatus[key]);
	++nBadCrystals[key];
	sumEta[key] += point.eta();
	sumPhi[key] += point.phi();
      }
    }
  }

  struct towerInfo 
  {
    towerInfo(uint32_t id, unsigned nBad, unsigned maxStatus, double eta, double phi)
      : id_(id), 
	nBad_(nBad), 
	maxStatus_(maxStatus), 
	eta_(eta), 
	phi_(phi) 
    {}
    uint32_t id_;
    unsigned nBad_;
    unsigned maxStatus_;
    double eta_;
    double phi_;
  };
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PolarLorentzVector;

  std::string moduleLabel_;
  unsigned minStatus_;
  double dR_;

  std::vector<towerInfo> badTowers_;
  static const uint16_t statusMask_ = 0x1F;

  uint32_t channelStatusId_cache_;
  uint32_t caloGeometryId_cache_;
  uint32_t idealGeometryId_cache_;
  bool isFirstEvent_;
};

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronDeadECAL);
