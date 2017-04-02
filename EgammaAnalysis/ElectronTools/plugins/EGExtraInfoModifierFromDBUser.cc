#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include <vdt/vdtMath.h>

namespace {
  const edm::InputTag empty_tag;
}

#include <unordered_map>

class EGExtraInfoModifierFromDBUser : public ModifyObjectValueBase {
public:
  typedef edm::EDGetTokenT<edm::ValueMap<float> > ValMapFloatToken;
  typedef edm::EDGetTokenT<edm::ValueMap<int> > ValMapIntToken;
  typedef std::pair<edm::InputTag, ValMapFloatToken> ValMapFloatTagTokenPair;
  typedef std::pair<edm::InputTag, ValMapIntToken> ValMapIntTagTokenPair;

  struct electron_config {
    edm::InputTag electron_src;
    edm::EDGetTokenT<edm::View<pat::Electron> > tok_electron_src;
    std::unordered_map<std::string, ValMapFloatTagTokenPair> tag_float_token_map;
    std::unordered_map<std::string, ValMapIntTagTokenPair> tag_int_token_map;

    std::vector<std::string> condnames_ecalonly_mean;
    std::vector<std::string> condnames_ecalonly_sigma;
    std::vector<std::string> condnames_ecaltrk_mean;
    std::vector<std::string> condnames_ecaltrk_sigma;
  };

  struct photon_config {
    edm::InputTag photon_src;
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    std::unordered_map<std::string, ValMapFloatTagTokenPair> tag_float_token_map;
    std::unordered_map<std::string, ValMapIntTagTokenPair> tag_int_token_map;

    std::vector<std::string> condnames_ecalonly_mean;
    std::vector<std::string> condnames_ecalonly_sigma;
  };

  EGExtraInfoModifierFromDBUser(const edm::ParameterSet& conf);
  ~EGExtraInfoModifierFromDBUser();
    
  void setEvent(const edm::Event&) override final;
  void setEventContent(const edm::EventSetup&) override final;
  void setConsumes(edm::ConsumesCollector&) override final;
  
  void modifyObject(reco::GsfElectron&) const override final;
  void modifyObject(reco::Photon&) const override final;
  
  // just calls reco versions
  void modifyObject(pat::Electron&) const override final; 
  void modifyObject(pat::Photon&) const override final;

private:
  electron_config e_conf;
  photon_config   ph_conf;
  std::unordered_map<unsigned,edm::Ptr<reco::GsfElectron> > eles_by_oop; // indexed by original object ptr
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > ele_vmaps;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<int> > > ele_int_vmaps;
  std::unordered_map<unsigned,edm::Ptr<reco::Photon> > phos_by_oop;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > pho_vmaps;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<int> > > pho_int_vmaps;

  edm::Handle<edm::SortedCollection<EcalRecHit> > ecalRecHitsEB_;
  edm::Handle<edm::SortedCollection<EcalRecHit> > ecalRecHitsEE_;
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit> > ecalRecHitEBToken_;
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit> > ecalRecHitEEToken_;
  
  float rhoValue_;
  edm::InputTag rhoTag_;
  edm::EDGetTokenT<double> rhoToken_;

  edm::InputTag ecalRecHitEBTag_;
  edm::InputTag ecalRecHitEETag_;

  const edm::EventSetup* iSetup_;

  std::vector<const GBRForestD*> ph_forestH_mean_;
  std::vector<const GBRForestD*> ph_forestH_sigma_; 
  std::vector<const GBRForestD*> e_forestH_mean_;
  std::vector<const GBRForestD*> e_forestH_sigma_;

  const CaloTopology *topology_;

  double lowEnergy_ECALonlyThr_; // 300
  double lowEnergy_ECALTRKThr_;  // 50
  double highEnergy_ECALTRKThr_; // 200
  double eOverP_ECALTRKThr_;     // 0.025
  double epDiffSig_ECALTRKThr_;  // 15
  double epSig_ECALTRKThr_;      // 10
  
  
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromDBUser,
		  "EGExtraInfoModifierFromDBUser");

EGExtraInfoModifierFromDBUser::EGExtraInfoModifierFromDBUser(const edm::ParameterSet& conf) :
  ModifyObjectValueBase(conf) {

  lowEnergy_ECALonlyThr_ = conf.getParameter<double>("lowEnergy_ECALonlyThr");
  lowEnergy_ECALTRKThr_ = conf.getParameter<double>("lowEnergy_ECALTRKThr");
  highEnergy_ECALTRKThr_ = conf.getParameter<double>("highEnergy_ECALTRKThr");
  eOverP_ECALTRKThr_ = conf.getParameter<double>("eOverP_ECALTRKThr");
  epDiffSig_ECALTRKThr_ = conf.getParameter<double>("epDiffSig_ECALTRKThr");
  epSig_ECALTRKThr_ = conf.getParameter<double>("epSig_ECALTRKThr");

  rhoTag_ = conf.getParameter<edm::InputTag>("rhoCollection");

  ecalRecHitEBTag_ = conf.getParameter<edm::InputTag>("ecalrechitsEB");  
  ecalRecHitEETag_ = conf.getParameter<edm::InputTag>("ecalrechitsEE");

  constexpr char electronSrc[] =  "electronSrc";
  constexpr char photonSrc[] =  "photonSrc";
    
  if(conf.exists("electron_config")) {
    const edm::ParameterSet& electrons = conf.getParameter<edm::ParameterSet>("electron_config");
    if( electrons.exists(electronSrc) ) 
      e_conf.electron_src = electrons.getParameter<edm::InputTag>(electronSrc);
    
    std::vector<std::string> intValueMaps;
    if ( electrons.existsAs<std::vector<std::string> >("intValueMaps")) 
      intValueMaps = electrons.getParameter<std::vector<std::string> >("intValueMaps");

    const std::vector<std::string> parameters = electrons.getParameterNames();
    for( const std::string& name : parameters ) {
      if( std::string(electronSrc) == name ) 
	continue;
      if( electrons.existsAs<edm::InputTag>(name)) {
	for (auto vmp : intValueMaps) {
	  if (name == vmp) {
	    e_conf.tag_int_token_map[name] = ValMapIntTagTokenPair(electrons.getParameter<edm::InputTag>(name), ValMapIntToken());
	    break;
	  } 
	}
	e_conf.tag_float_token_map[name] = ValMapFloatTagTokenPair(electrons.getParameter<edm::InputTag>(name), ValMapFloatToken());
      }
    }
    
    e_conf.condnames_ecalonly_mean  = electrons.getParameter<std::vector<std::string> >("regressionKey_ecalonly");
    e_conf.condnames_ecalonly_sigma = electrons.getParameter<std::vector<std::string> >("uncertaintyKey_ecalonly");
    e_conf.condnames_ecaltrk_mean   = electrons.getParameter<std::vector<std::string> >("regressionKey_ecaltrk");
    e_conf.condnames_ecaltrk_sigma  = electrons.getParameter<std::vector<std::string> >("uncertaintyKey_ecaltrk");

    unsigned int encor = e_conf.condnames_ecalonly_mean.size();
    e_forestH_mean_.reserve(2*encor);
    e_forestH_sigma_.reserve(2*encor);

  }
  
  if( conf.exists("photon_config") ) { 
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");

    if( photons.exists(photonSrc) ) 
      ph_conf.photon_src = photons.getParameter<edm::InputTag>(photonSrc);

    std::vector<std::string> intValueMaps;
    if ( photons.existsAs<std::vector<std::string> >("intValueMaps")) 
      intValueMaps = photons.getParameter<std::vector<std::string> >("intValueMaps");

    const std::vector<std::string> parameters = photons.getParameterNames();
    for( const std::string& name : parameters ) {
      if( std::string(photonSrc) == name ) 
	continue;
      if( photons.existsAs<edm::InputTag>(name)) {
	for (auto vmp : intValueMaps) {
	  if (name == vmp) {
	    ph_conf.tag_int_token_map[name] = ValMapIntTagTokenPair(photons.getParameter<edm::InputTag>(name), ValMapIntToken());
	    break;
	  } 
	}
	ph_conf.tag_float_token_map[name] = ValMapFloatTagTokenPair(photons.getParameter<edm::InputTag>(name), ValMapFloatToken());
      }
    }

    ph_conf.condnames_ecalonly_mean  = photons.getParameter<std::vector<std::string>>("regressionKey_ecalonly");
    ph_conf.condnames_ecalonly_sigma = photons.getParameter<std::vector<std::string>>("uncertaintyKey_ecalonly");

    unsigned int ncor = ph_conf.condnames_ecalonly_mean.size();
    ph_forestH_mean_.reserve(ncor);
    ph_forestH_sigma_.reserve(ncor);

  }


}

namespace {
  template<typename T>
  inline void get_product(const edm::Event& evt,
                          const edm::EDGetTokenT<edm::ValueMap<T> >& tok,
                          std::unordered_map<unsigned, edm::Handle<edm::ValueMap<T> > >& map) {
    evt.getByToken(tok,map[tok.index()]);
  }
}

EGExtraInfoModifierFromDBUser::~EGExtraInfoModifierFromDBUser() {}

void EGExtraInfoModifierFromDBUser::setEvent(const edm::Event& evt) {

  //  std::cout << evt.id().event() << " " << evt.id().luminosityBlock() << " " << evt.id().run() << std::endl;

  eles_by_oop.clear();
  phos_by_oop.clear();  
  ele_vmaps.clear();
  ele_int_vmaps.clear();
  pho_vmaps.clear();
  pho_int_vmaps.clear();
  
  if( !e_conf.tok_electron_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Electron> > eles;
    evt.getByToken(e_conf.tok_electron_src, eles);
    
    for( unsigned i = 0; i < eles->size(); ++i ) {
      edm::Ptr<pat::Electron> ptr = eles->ptrAt(i);
      eles_by_oop[ptr->originalObjectRef().key()] = ptr;
    }    
  }

  for (std::unordered_map<std::string, ValMapFloatTagTokenPair>::iterator imap = e_conf.tag_float_token_map.begin(); 
       imap != e_conf.tag_float_token_map.end(); 
       imap++) {
    get_product(evt, imap->second.second, ele_vmaps);
  }

  for (std::unordered_map<std::string, ValMapIntTagTokenPair>::iterator imap = e_conf.tag_int_token_map.begin(); 
       imap != e_conf.tag_int_token_map.end(); 
       imap++) {
    get_product(evt, imap->second.second, ele_int_vmaps);
  }
  
  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Photon> > phos;
    evt.getByToken(ph_conf.tok_photon_src,phos);
  
    for( unsigned i = 0; i < phos->size(); ++i ) {
      edm::Ptr<pat::Photon> ptr = phos->ptrAt(i);
      phos_by_oop[ptr->originalObjectRef().key()] = ptr;
    }
  }
   

  for (std::unordered_map<std::string, ValMapFloatTagTokenPair>::iterator imap = ph_conf.tag_float_token_map.begin(); 
       imap != ph_conf.tag_float_token_map.end(); 
       imap++) {
    get_product(evt, imap->second.second, pho_vmaps);
  }

  for (std::unordered_map<std::string, ValMapIntTagTokenPair>::iterator imap = ph_conf.tag_int_token_map.begin(); 
       imap != ph_conf.tag_int_token_map.end(); 
       imap++) {
    get_product(evt, imap->second.second, pho_int_vmaps);
  }
  
  edm::Handle<double> rhoH;
  evt.getByToken(rhoToken_, rhoH);
  rhoValue_ = *rhoH;
 
  evt.getByToken( ecalRecHitEBToken_, ecalRecHitsEB_ );
  evt.getByToken( ecalRecHitEEToken_, ecalRecHitsEE_ );
  
}

void EGExtraInfoModifierFromDBUser::setEventContent(const edm::EventSetup& evs) {

  iSetup_ = &evs;

  edm::ESHandle<GBRForestD> forestDEH;
  
  const std::vector<std::string> ph_condnames_ecalonly_mean  = ph_conf.condnames_ecalonly_mean;
  const std::vector<std::string> ph_condnames_ecalonly_sigma = ph_conf.condnames_ecalonly_sigma;

  unsigned int ncor = ph_condnames_ecalonly_mean.size();
  for (unsigned int icor=0; icor<ncor; ++icor) {
    evs.get<GBRDWrapperRcd>().get(ph_condnames_ecalonly_mean[icor], forestDEH);
    ph_forestH_mean_[icor] = forestDEH.product();
    evs.get<GBRDWrapperRcd>().get(ph_condnames_ecalonly_sigma[icor], forestDEH);
    ph_forestH_sigma_[icor] = forestDEH.product();
  }

  const std::vector<std::string> e_condnames_ecalonly_mean  = e_conf.condnames_ecalonly_mean;
  const std::vector<std::string> e_condnames_ecalonly_sigma = e_conf.condnames_ecalonly_sigma;
  const std::vector<std::string> e_condnames_ecaltrk_mean  = e_conf.condnames_ecaltrk_mean;
  const std::vector<std::string> e_condnames_ecaltrk_sigma = e_conf.condnames_ecaltrk_sigma;

  unsigned int encor = e_condnames_ecalonly_mean.size();  
  for (unsigned int icor=0; icor<encor; ++icor) {
    evs.get<GBRDWrapperRcd>().get(e_condnames_ecalonly_mean[icor], forestDEH);
    e_forestH_mean_[icor] = forestDEH.product();
    evs.get<GBRDWrapperRcd>().get(e_condnames_ecalonly_sigma[icor], forestDEH);
    e_forestH_sigma_[icor] = forestDEH.product();
  } 
  for (unsigned int icor=0; icor<encor; ++icor) {
    evs.get<GBRDWrapperRcd>().get(e_condnames_ecaltrk_mean[icor], forestDEH);
    e_forestH_mean_[icor+encor] = forestDEH.product();
    evs.get<GBRDWrapperRcd>().get(e_condnames_ecaltrk_sigma[icor], forestDEH);
    e_forestH_sigma_[icor+encor] = forestDEH.product();
  }
  
  edm::ESHandle<CaloTopology> pTopology;
  evs.get<CaloTopologyRecord>().get(pTopology);
  topology_ = pTopology.product();
  
}

namespace {
  template<typename T, typename U, typename V>
  inline void make_consumes(T& tag,U& tok,V& sume) { 
    if(!(empty_tag == tag)) 
      tok = sume.template consumes<edm::ValueMap<float> >(tag); 
  }

  template<typename T, typename U, typename V>
  inline void make_int_consumes(T& tag,U& tok,V& sume) { 
    if(!(empty_tag == tag)) 
      tok = sume.template consumes<edm::ValueMap<int> >(tag); 
  }
}

void EGExtraInfoModifierFromDBUser::setConsumes(edm::ConsumesCollector& sumes) {
 
  rhoToken_ = sumes.consumes<double>(rhoTag_);
  ecalRecHitEBToken_ = sumes.consumes<edm::SortedCollection<EcalRecHit> > ( ecalRecHitEBTag_ );
  ecalRecHitEEToken_ = sumes.consumes<edm::SortedCollection<EcalRecHit> > ( ecalRecHitEETag_ );

  //setup electrons
  if(!(empty_tag == e_conf.electron_src))
    e_conf.tok_electron_src = sumes.consumes<edm::View<pat::Electron> >(e_conf.electron_src);  

  for ( std::unordered_map<std::string, ValMapFloatTagTokenPair>::iterator imap = e_conf.tag_float_token_map.begin(); 
	imap != e_conf.tag_float_token_map.end(); 
	imap++) {
    make_consumes(imap->second.first, imap->second.second, sumes);
  }  

  for ( std::unordered_map<std::string, ValMapIntTagTokenPair>::iterator imap = e_conf.tag_int_token_map.begin(); 
	imap != e_conf.tag_int_token_map.end(); 
	imap++) {
    make_int_consumes(imap->second.first, imap->second.second, sumes);
  }  
  
  // setup photons 
  if(!(empty_tag == ph_conf.photon_src)) 
    ph_conf.tok_photon_src = sumes.consumes<edm::View<pat::Photon> >(ph_conf.photon_src);

  for ( std::unordered_map<std::string, ValMapFloatTagTokenPair>::iterator imap = ph_conf.tag_float_token_map.begin(); 
	imap != ph_conf.tag_float_token_map.end(); 
	imap++) {
    make_consumes(imap->second.first, imap->second.second, sumes);
  }  

  for ( std::unordered_map<std::string, ValMapIntTagTokenPair>::iterator imap = ph_conf.tag_int_token_map.begin(); 
	imap != ph_conf.tag_int_token_map.end(); 
	imap++) {
    make_int_consumes(imap->second.first, imap->second.second, sumes);
  }  
}

namespace {
  template<typename T, typename U, typename V, typename Z>
  inline void assignValue(const T& ptr, const U& tok, const V& map, Z& value) {
    if( !tok.isUninitialized() ) value = map.find(tok.index())->second->get(ptr.id(),ptr.key());
  }
}

void EGExtraInfoModifierFromDBUser::modifyObject(reco::GsfElectron& ele) const {

  // regression calculation needs no additional valuemaps

  const reco::SuperClusterRef& the_sc = ele.superCluster();
  const edm::Ptr<reco::CaloCluster>& theseed = the_sc->seed();
  const int numberOfClusters =  the_sc->clusters().size();
  const bool missing_clusters = !the_sc->clusters()[numberOfClusters-1].isAvailable();
  
  if( missing_clusters ) return ; // do not apply corrections in case of missing info (slimmed MiniAOD electrons)

  const bool iseb = ele.isEB();  
  edm::Handle<edm::SortedCollection<EcalRecHit> > ecalRecHits;
  if (iseb) ecalRecHits = ecalRecHitsEB_ ;
  else      ecalRecHits = ecalRecHitsEE_ ;
  if ( !ecalRecHits.isValid() ) return;
  
  int nSaturatedXtals  = 0;
  std::vector< std::pair<DetId, float> > hitsAndFractions = theseed->hitsAndFractions();
  for (auto hitFractionPair : hitsAndFractions) {    
    auto ecalRecHit = ecalRecHits->find(hitFractionPair.first);
    if (ecalRecHit == ecalRecHits->end()) continue;
    if (ecalRecHit->checkFlag(EcalRecHit::Flags::kSaturated)) nSaturatedXtals++;
  }

  std::array<float, 32> eval;  
  const double raw_energy = the_sc->rawEnergy(); 
  const double raw_es_energy = the_sc->preshowerEnergy();
  const auto& full5x5_ess = ele.full5x5_showerShape();

  float e5x5Inverse = full5x5_ess.e5x5 != 0. ? 1./full5x5_ess.e5x5 : 0.;

  eval[0]  = raw_energy;
  eval[1]  = the_sc->etaWidth();
  eval[2]  = the_sc->phiWidth(); 
  eval[3]  = the_sc->seed()->energy()/raw_energy;
  eval[4]  = full5x5_ess.e5x5/raw_energy;
  eval[5]  = ele.hcalOverEcalBc();
  eval[6]  = rhoValue_;
  eval[7]  = theseed->eta() - the_sc->position().Eta();
  eval[8]  = reco::deltaPhi( theseed->phi(),the_sc->position().Phi());
  eval[9]  = full5x5_ess.r9;
  eval[10]  = full5x5_ess.sigmaIetaIeta;
  eval[11]  = full5x5_ess.sigmaIetaIphi;
  eval[12]  = full5x5_ess.sigmaIphiIphi;
  eval[13]  = full5x5_ess.eMax*e5x5Inverse;
  eval[14]  = full5x5_ess.e2nd*e5x5Inverse;
  eval[15]  = full5x5_ess.eTop*e5x5Inverse;
  eval[16]  = full5x5_ess.eBottom*e5x5Inverse;
  eval[17]  = full5x5_ess.eLeft*e5x5Inverse;
  eval[18]  = full5x5_ess.eRight*e5x5Inverse;
  eval[19]  = EcalClusterToolsT<true>::e2x5Max(*theseed, &*ecalRecHits, topology_)*e5x5Inverse;
  eval[20]  = EcalClusterToolsT<true>::e2x5Left(*theseed, &*ecalRecHits, topology_)*e5x5Inverse;
  eval[21]  = EcalClusterToolsT<true>::e2x5Right(*theseed, &*ecalRecHits, topology_)*e5x5Inverse;
  eval[22]  = EcalClusterToolsT<true>::e2x5Top(*theseed, &*ecalRecHits, topology_)*e5x5Inverse;
  eval[23]  = EcalClusterToolsT<true>::e2x5Bottom(*theseed, &*ecalRecHits, topology_)*e5x5Inverse;
  eval[24]  = nSaturatedXtals;
  eval[25]  = std::max(0,numberOfClusters);
      
  // calculate coordinate variables
  EcalClusterLocal _ecalLocal;
  if (iseb) {

    float dummy;
    int ieta;
    int iphi;
    _ecalLocal.localCoordsEB(*theseed, *iSetup_, dummy, dummy, ieta, iphi, dummy, dummy);
    eval[26] = ieta;
    eval[27] = iphi;
    int signieta = ieta > 0 ? +1 : -1;
    eval[28] = (ieta-signieta)%5;
    eval[29] = (iphi-1)%2;
    eval[30] = (abs(ieta)<=25)*((ieta-signieta)) + (abs(ieta)>25)*((ieta-26*signieta)%20);  
    eval[31] = (iphi-1)%20;

  } else {

    float dummy;
    int ix;
    int iy;
    _ecalLocal.localCoordsEE(*theseed, *iSetup_, dummy, dummy, ix, iy, dummy, dummy);
    eval[26] = ix;
    eval[27] = iy;
    eval[28] = raw_es_energy/raw_energy;

  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  constexpr double meanlimlow  = -1.0;
  constexpr double meanlimhigh = 3.0;
  constexpr double meanoffset  = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
  constexpr double meanscale   = 0.5*(meanlimhigh-meanlimlow);
  
  constexpr double sigmalimlow  = 0.0002;
  constexpr double sigmalimhigh = 0.5;
  constexpr double sigmaoffset  = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
  constexpr double sigmascale   = 0.5*(sigmalimhigh-sigmalimlow);  

  
  size_t coridx = 0;
  float raw_pt = raw_energy/cosh(the_sc->eta());

  if (iseb && raw_pt < lowEnergy_ECALonlyThr_)
    coridx = 0;
  else if (iseb && raw_pt >= lowEnergy_ECALonlyThr_)
    coridx = 1;
  else if (!iseb && raw_pt < lowEnergy_ECALonlyThr_)
    coridx = 2;
  else if (!iseb && raw_pt >= lowEnergy_ECALonlyThr_)
    coridx = 3;
  
  //these are the actual BDT responses
  double rawmean = e_forestH_mean_[coridx]->GetResponse(eval.data());
  double rawsigma = e_forestH_sigma_[coridx]->GetResponse(eval.data());
  
  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);

  // std::cout << coridx << " " << raw_pt << " " << the_sc->eta() << " " << the_sc->position().Eta() << "    ";
  // for (int i =0 ; i<32; i++) {
  //   if (!iseb && i>28) continue;
  //   std::cout << eval[i] << " ";
  // }
  // std::cout << mean << " " << sigma << " ";
  // std::cout << std::endl;

  // Correct the energy. A negative energy means that the correction went
  // outside the boundaries of the training. In this case uses raw.
  // The resolution estimation, on the other hand should be ok.
  if (mean < 0.) mean = 1.0;
    
  const double ecor = mean*(raw_energy + raw_es_energy);
  const double sigmacor = sigma*ecor;
  
  ele.setCorrectedEcalEnergy(ecor);
  ele.setCorrectedEcalEnergyError(sigmacor);

  double combinedEnergy = ecor;
  double combinedEnergyError = sigmacor;

  auto el_track = ele.gsfTrack();

  const float trkMomentum = el_track->pMode();
  const float trkEta      = el_track->etaMode();
  const float trkPhi      = el_track->phiMode();
  
  float ptMode       = el_track->ptMode();
  float ptModeErrror = el_track->ptModeError();
  float etaModeError = el_track->etaModeError();
  float pModeError   = sqrt(ptModeErrror*ptModeErrror*cosh(trkEta)*cosh(trkEta) + ptMode*ptMode*sinh(trkEta)*sinh(trkEta)*etaModeError*etaModeError);
  
  const float trkMomentumError = pModeError;
  const float eOverP = (raw_energy+raw_es_energy)*mean/trkMomentum;
  const float fbrem = ele.fbrem();

  // E-p combination
  if (ecor < highEnergy_ECALTRKThr_ &&
      eOverP > eOverP_ECALTRKThr_ && 
      std::abs(ecor - trkMomentum) < epDiffSig_ECALTRKThr_*std::sqrt(pModeError*pModeError+sigmacor*sigmacor) && 
      trkMomentumError < epSig_ECALTRKThr_*trkMomentum) { 

    raw_pt = ecor/cosh(trkEta);
    if (iseb && raw_pt < lowEnergy_ECALTRKThr_)
      coridx = 4;
    else if (iseb && raw_pt >= lowEnergy_ECALTRKThr_)
      coridx = 5;
    else if (!iseb && raw_pt < lowEnergy_ECALTRKThr_)
      coridx = 6;
    else if (!iseb && raw_pt >= lowEnergy_ECALTRKThr_)
      coridx = 7;
  
    eval[0] = ecor;
    eval[1] = sigma/mean;
    eval[2] = trkMomentumError/trkMomentum;
    eval[3] = eOverP;
    eval[4] = ele.ecalDrivenSeed();
    eval[5] = full5x5_ess.r9;
    eval[6] = fbrem;
    eval[7] = trkEta; 
    eval[8] = trkPhi; 
    
    float rawcomb = ( ecor*trkMomentumError*trkMomentumError + trkMomentum*(raw_energy+raw_es_energy)*(raw_energy+raw_es_energy)*sigma*sigma ) / ( trkMomentumError*trkMomentumError + (raw_energy + raw_es_energy)*(raw_energy + raw_es_energy)*sigma*sigma );

    //these are the actual BDT responses
    double rawmean_trk = e_forestH_mean_[coridx]->GetResponse(eval.data());
    double rawsigma_trk = e_forestH_sigma_[coridx]->GetResponse(eval.data());
    
    //apply transformation to limited output range (matching the training)
    double mean_trk = meanoffset + meanscale*vdt::fast_sin(rawmean_trk);
    double sigma_trk = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma_trk);

    // std::cout << coridx << " " << rawcomb << " " << raw_pt << " " << ecor << " " << trkEta << "    ";
    // for (int i =0 ; i<9; i++) {
    //   std::cout << eval[i] << " ";
    // }
    // std::cout << mean_trk << " " << sigma_trk << std::endl;
    
    // Final correction
    // A negative energy means that the correction went
    // outside the boundaries of the training. In this case uses raw.
    // The resolution estimation, on the other hand should be ok.
    if (mean_trk < 0.) mean_trk = 1.0;

    combinedEnergy = mean_trk*rawcomb;
    combinedEnergyError = sigma_trk*rawcomb;
  }

  math::XYZTLorentzVector oldFourMomentum = ele.p4();
  math::XYZTLorentzVector newFourMomentum = math::XYZTLorentzVector(oldFourMomentum.x()*combinedEnergy/oldFourMomentum.t(),
								    oldFourMomentum.y()*combinedEnergy/oldFourMomentum.t(),
								    oldFourMomentum.z()*combinedEnergy/oldFourMomentum.t(),
								    combinedEnergy);
 
  ele.correctMomentum(newFourMomentum, ele.trackMomentumError(), combinedEnergyError);
}

void EGExtraInfoModifierFromDBUser::modifyObject(pat::Electron& ele) const {
  modifyObject(static_cast<reco::GsfElectron&>(ele));
}

void EGExtraInfoModifierFromDBUser::modifyObject(reco::Photon& pho) const {
  // regression calculation needs no additional valuemaps
  

  const reco::SuperClusterRef& the_sc = pho.superCluster();
  const edm::Ptr<reco::CaloCluster>& theseed = the_sc->seed();  
  const int numberOfClusters =  the_sc->clusters().size();
  const bool missing_clusters = !the_sc->clusters()[numberOfClusters-1].isAvailable();

  if( missing_clusters ) return ; // do not apply corrections in case of missing info (slimmed MiniAOD electrons)

  const bool iseb = pho.isEB();  
  edm::Handle<edm::SortedCollection<EcalRecHit> > ecalRecHits;
  if (iseb) ecalRecHits = ecalRecHitsEB_ ;
  else      ecalRecHits = ecalRecHitsEE_ ;

  int nSaturatedXtals  = 0;
  std::vector< std::pair<DetId, float> > hitsAndFractions = theseed->hitsAndFractions();
  for (auto hitFractionPair : hitsAndFractions) {
    auto ecalRecHit = ecalRecHits->find(hitFractionPair.first);
    if (ecalRecHit == ecalRecHits->end()) continue;
    if (ecalRecHit->checkFlag(EcalRecHit::Flags::kSaturated)) nSaturatedXtals++;
  }
  
  std::array<float, 32> eval;  
  const double raw_energy = the_sc->rawEnergy(); 
  const double raw_es_energy = the_sc->preshowerEnergy();
  const auto& full5x5_pss = pho.full5x5_showerShapeVariables();

  float e5x5Inverse = full5x5_pss.e5x5 != 0. ? 1./full5x5_pss.e5x5 : 0.;

  eval[0]  = raw_energy;
  eval[1]  = the_sc->etaWidth();
  eval[2]  = the_sc->phiWidth(); 
  eval[3]  = the_sc->seed()->energy()/raw_energy;
  eval[4]  = full5x5_pss.e5x5/raw_energy;
  eval[5]  = pho.hadronicOverEm();
  eval[6]  = rhoValue_;
  eval[7]  = theseed->eta() - the_sc->position().Eta();
  eval[8]  = reco::deltaPhi( theseed->phi(),the_sc->position().Phi());
  eval[9]  = pho.full5x5_r9();
  eval[10]  = full5x5_pss.sigmaIetaIeta;
  eval[11]  = full5x5_pss.sigmaIetaIphi;
  eval[12]  = full5x5_pss.sigmaIphiIphi;
  eval[13]  = full5x5_pss.maxEnergyXtal*e5x5Inverse;
  eval[14]  = full5x5_pss.e2nd*e5x5Inverse;
  eval[15]  = full5x5_pss.eTop*e5x5Inverse;
  eval[16]  = full5x5_pss.eBottom*e5x5Inverse;
  eval[17]  = full5x5_pss.eLeft*e5x5Inverse;
  eval[18]  = full5x5_pss.eRight*e5x5Inverse;
  eval[19]  = full5x5_pss.e2x5Max/full5x5_pss.e5x5;
  eval[20]  = full5x5_pss.e2x5Left/full5x5_pss.e5x5;
  eval[21]  = full5x5_pss.e2x5Right/full5x5_pss.e5x5;
  eval[22]  = full5x5_pss.e2x5Top/full5x5_pss.e5x5;
  eval[23]  = full5x5_pss.e2x5Bottom/full5x5_pss.e5x5;
  eval[24]  = nSaturatedXtals;
  eval[25]  = std::max(0,numberOfClusters);
      
  // calculate coordinate variables
  EcalClusterLocal _ecalLocal;

  if (iseb) {

    float dummy;
    int ieta;
    int iphi;
    _ecalLocal.localCoordsEB(*theseed, *iSetup_, dummy, dummy, ieta, iphi, dummy, dummy);
    eval[26] = ieta;
    eval[27] = iphi;
    int signieta = ieta > 0 ? +1 : -1;
    eval[28] = (ieta-signieta)%5;
    eval[29] = (iphi-1)%2;
    eval[30] = (abs(ieta)<=25)*((ieta-signieta)) + (abs(ieta)>25)*((ieta-26*signieta)%20);  
    eval[31] = (iphi-1)%20;

  } else {

    float dummy;
    int ix;
    int iy;
    _ecalLocal.localCoordsEE(*theseed, *iSetup_, dummy, dummy, ix, iy, dummy, dummy);
    eval[26] = ix;
    eval[27] = iy;
    eval[28] = raw_es_energy/raw_energy;

  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  constexpr double meanlimlow  = -1.0;
  constexpr double meanlimhigh = 3.0;
  constexpr double meanoffset  = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
  constexpr double meanscale   = 0.5*(meanlimhigh-meanlimlow);
  
  constexpr double sigmalimlow  = 0.0002;
  constexpr double sigmalimhigh = 0.5;
  constexpr double sigmaoffset  = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
  constexpr double sigmascale   = 0.5*(sigmalimhigh-sigmalimlow);  
  
  size_t coridx = 0;
  float raw_pt = raw_energy/cosh(the_sc->eta());

  if (iseb && raw_pt < lowEnergy_ECALonlyThr_)
    coridx = 0;
  else if (iseb && raw_pt >= lowEnergy_ECALonlyThr_)
    coridx = 1;
  else if (!iseb && raw_pt < lowEnergy_ECALonlyThr_)
    coridx = 2;
  else if (!iseb && raw_pt >= lowEnergy_ECALonlyThr_)
    coridx = 3;
  
  //these are the actual BDT responses
  double rawmean = ph_forestH_mean_[coridx]->GetResponse(eval.data());
  double rawsigma = ph_forestH_sigma_[coridx]->GetResponse(eval.data());
  
  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);

  //  std::cout << coridx << " " << raw_pt << " " << the_sc->eta() << " " << the_sc->position().Eta() << "    ";
  //   for (int i =0 ; i<32; i++) {
  //     if (!iseb && i>28) continue;
  //     std::cout << eval[i] << " ";
  //   }
  //   std::cout << mean << " " << sigma << " ";
  //   for (size_t i=0; i<4; i++) std::cout << meanoffset + meanscale*vdt::fast_sin(ph_forestH_mean_[i]->GetResponse(eval.data())) << " "; 
  //   std::cout << std::endl;

  // Correct the energy. A negative energy means that the correction went
  // outside the boundaries of the training. In this case uses raw.
  // The resolution estimation, on the other hand should be ok.
  if (mean < 0.) mean = 1.0;

  const double ecor = mean*(raw_energy + raw_es_energy);
  const double sigmacor = sigma*ecor;
  
  pho.setCorrectedEnergy(reco::Photon::P4type::regression2, ecor, sigmacor, true);     
}

void EGExtraInfoModifierFromDBUser::modifyObject(pat::Photon& pho) const {
  modifyObject(static_cast<reco::Photon&>(pho));
}
