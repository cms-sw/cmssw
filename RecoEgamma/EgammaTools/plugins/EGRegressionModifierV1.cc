#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"

#include <vdt/vdtMath.h>

namespace {
  const edm::InputTag empty_tag;
}

#include <unordered_map>

class EGRegressionModifierV1 : public ModifyObjectValueBase {
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

    std::vector<std::string> condnames_mean_50ns;
    std::vector<std::string> condnames_sigma_50ns;
    std::vector<std::string> condnames_mean_25ns;
    std::vector<std::string> condnames_sigma_25ns;
    std::string condnames_weight_50ns;
    std::string condnames_weight_25ns;
  };

  struct photon_config {
    edm::InputTag photon_src;
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    std::unordered_map<std::string, ValMapFloatTagTokenPair> tag_float_token_map;
    std::unordered_map<std::string, ValMapIntTagTokenPair> tag_int_token_map;

    std::vector<std::string> condnames_mean_50ns;
    std::vector<std::string> condnames_sigma_50ns;
    std::vector<std::string> condnames_mean_25ns;
    std::vector<std::string> condnames_sigma_25ns;
  };

  EGRegressionModifierV1(const edm::ParameterSet& conf);
  ~EGRegressionModifierV1() override {};
    
  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final;
  void setConsumes(edm::ConsumesCollector&) final;
  
  void modifyObject(reco::GsfElectron&) const final;
  void modifyObject(reco::Photon&) const final;
  
  // just calls reco versions
  void modifyObject(pat::Electron&) const final; 
  void modifyObject(pat::Photon&) const final;

private:
  electron_config e_conf;
  photon_config   ph_conf;
  std::unordered_map<unsigned,edm::Ptr<reco::GsfElectron> > eles_by_oop; // indexed by original object ptr
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > ele_vmaps;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<int> > > ele_int_vmaps;
  std::unordered_map<unsigned,edm::Ptr<reco::Photon> > phos_by_oop;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<float> > > pho_vmaps;
  std::unordered_map<unsigned,edm::Handle<edm::ValueMap<int> > > pho_int_vmaps;

  bool autoDetectBunchSpacing_;
  int bunchspacing_;
  edm::InputTag bunchspacingTag_;
  edm::EDGetTokenT<unsigned int> bunchSpacingToken_;
  float rhoValue_;
  edm::InputTag rhoTag_;
  edm::EDGetTokenT<double> rhoToken_;
  int nVtx_;
  edm::InputTag vtxTag_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::Handle<reco::VertexCollection> vtxH_;
  bool applyExtraHighEnergyProtection_;

  const edm::EventSetup* iSetup_;

  std::vector<const GBRForestD*> ph_forestH_mean_;
  std::vector<const GBRForestD*> ph_forestH_sigma_; 
  std::vector<const GBRForestD*> e_forestH_mean_;
  std::vector<const GBRForestD*> e_forestH_sigma_; 
  const GBRForest* ep_forestH_weight_;
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGRegressionModifierV1,
		  "EGRegressionModifierV1");

EGRegressionModifierV1::EGRegressionModifierV1(const edm::ParameterSet& conf) :
  ModifyObjectValueBase(conf) {

  bunchspacing_ = 450;
  autoDetectBunchSpacing_ = conf.getParameter<bool>("autoDetectBunchSpacing");
  applyExtraHighEnergyProtection_ = conf.getParameter<bool>("applyExtraHighEnergyProtection");

  rhoTag_ = conf.getParameter<edm::InputTag>("rhoCollection");
  vtxTag_ = conf.getParameter<edm::InputTag>("vertexCollection");
  
  if (autoDetectBunchSpacing_) {
    bunchspacingTag_ = conf.getParameter<edm::InputTag>("bunchSpacingTag");
  } else {
    bunchspacing_ = conf.getParameter<int>("manualBunchSpacing");
  }

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
    
    e_conf.condnames_mean_50ns  = electrons.getParameter<std::vector<std::string> >("regressionKey_50ns");
    e_conf.condnames_sigma_50ns = electrons.getParameter<std::vector<std::string> >("uncertaintyKey_50ns");
    e_conf.condnames_mean_25ns  = electrons.getParameter<std::vector<std::string> >("regressionKey_25ns");
    e_conf.condnames_sigma_25ns = electrons.getParameter<std::vector<std::string> >("uncertaintyKey_25ns");
    e_conf.condnames_weight_50ns  = electrons.getParameter<std::string>("combinationKey_50ns");
    e_conf.condnames_weight_25ns  = electrons.getParameter<std::string>("combinationKey_25ns");
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

    ph_conf.condnames_mean_50ns = photons.getParameter<std::vector<std::string>>("regressionKey_50ns");
    ph_conf.condnames_sigma_50ns = photons.getParameter<std::vector<std::string>>("uncertaintyKey_50ns");
    ph_conf.condnames_mean_25ns = photons.getParameter<std::vector<std::string>>("regressionKey_25ns");
    ph_conf.condnames_sigma_25ns = photons.getParameter<std::vector<std::string>>("uncertaintyKey_25ns");
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

void EGRegressionModifierV1::setEvent(const edm::Event& evt) {
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
  
  if (autoDetectBunchSpacing_) {
      edm::Handle<unsigned int> bunchSpacingH;
      evt.getByToken(bunchSpacingToken_,bunchSpacingH);
      bunchspacing_ = *bunchSpacingH;
  }

  edm::Handle<double> rhoH;
  evt.getByToken(rhoToken_, rhoH);
  rhoValue_ = *rhoH;
  
  evt.getByToken(vtxToken_, vtxH_);
  nVtx_ = vtxH_->size();
}

void EGRegressionModifierV1::setEventContent(const edm::EventSetup& evs) {

  iSetup_ = &evs;

  edm::ESHandle<GBRForestD> forestDEH;
  edm::ESHandle<GBRForest> forestEH;

  const std::vector<std::string> ph_condnames_mean  = (bunchspacing_ == 25) ? ph_conf.condnames_mean_25ns  : ph_conf.condnames_mean_50ns;
  const std::vector<std::string> ph_condnames_sigma = (bunchspacing_ == 25) ? ph_conf.condnames_sigma_25ns : ph_conf.condnames_sigma_50ns;

  unsigned int ncor = ph_condnames_mean.size();
  for (unsigned int icor=0; icor<ncor; ++icor) {
    evs.get<GBRDWrapperRcd>().get(ph_condnames_mean[icor], forestDEH);
    ph_forestH_mean_.push_back(forestDEH.product());
    evs.get<GBRDWrapperRcd>().get(ph_condnames_sigma[icor], forestDEH);
    ph_forestH_sigma_.push_back(forestDEH.product());
  } 

  const std::vector<std::string> e_condnames_mean  = (bunchspacing_ == 25) ? e_conf.condnames_mean_25ns  : e_conf.condnames_mean_50ns;
  const std::vector<std::string> e_condnames_sigma = (bunchspacing_ == 25) ? e_conf.condnames_sigma_25ns : e_conf.condnames_sigma_50ns;
  const std::string ep_condnames_weight  = (bunchspacing_ == 25) ? e_conf.condnames_weight_25ns  : e_conf.condnames_weight_50ns;

  unsigned int encor = e_condnames_mean.size();
  evs.get<GBRWrapperRcd>().get(ep_condnames_weight, forestEH);
  ep_forestH_weight_ = forestEH.product(); 
    
  for (unsigned int icor=0; icor<encor; ++icor) {
    evs.get<GBRDWrapperRcd>().get(e_condnames_mean[icor], forestDEH);
    e_forestH_mean_.push_back(forestDEH.product());
    evs.get<GBRDWrapperRcd>().get(e_condnames_sigma[icor], forestDEH);
    e_forestH_sigma_.push_back(forestDEH.product());
  }
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

void EGRegressionModifierV1::setConsumes(edm::ConsumesCollector& sumes) {
 
  rhoToken_ = sumes.consumes<double>(rhoTag_);
  vtxToken_ = sumes.consumes<reco::VertexCollection>(vtxTag_);

  if (autoDetectBunchSpacing_)
    bunchSpacingToken_ = sumes.consumes<unsigned int>(bunchspacingTag_);

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

void EGRegressionModifierV1::modifyObject(reco::GsfElectron& ele) const {
  // regression calculation needs no additional valuemaps

  const reco::SuperClusterRef& the_sc = ele.superCluster();
  const edm::Ptr<reco::CaloCluster>& theseed = the_sc->seed();
  const int numberOfClusters =  the_sc->clusters().size();
  const bool missing_clusters = !the_sc->clusters()[numberOfClusters-1].isAvailable();

  if( missing_clusters ) return ; // do not apply corrections in case of missing info (slimmed MiniAOD electrons)
  
  std::array<float, 33> eval;  
  const double raw_energy = the_sc->rawEnergy(); 
  const auto& ess = ele.showerShape();

  // SET INPUTS
  eval[0]  = nVtx_;  
  eval[1]  = raw_energy;
  eval[2]  = the_sc->eta();
  eval[3]  = the_sc->phi();
  eval[4]  = the_sc->etaWidth();
  eval[5]  = the_sc->phiWidth(); 
  eval[6]  = ess.r9;
  eval[7]  = theseed->energy()/raw_energy;
  eval[8]  = ess.eMax/raw_energy;
  eval[9]  = ess.e2nd/raw_energy;
  eval[10] = (ess.eLeft + ess.eRight != 0.f  ? (ess.eLeft-ess.eRight)/(ess.eLeft+ess.eRight) : 0.f);
  eval[11] = (ess.eTop  + ess.eBottom != 0.f ? (ess.eTop-ess.eBottom)/(ess.eTop+ess.eBottom) : 0.f);
  eval[12] = ess.sigmaIetaIeta;
  eval[13] = ess.sigmaIetaIphi;
  eval[14] = ess.sigmaIphiIphi;
  eval[15] = std::max(0,numberOfClusters-1);
  
  // calculate sub-cluster variables
  std::vector<float> clusterRawEnergy;
  clusterRawEnergy.resize(std::max(3, numberOfClusters), 0);
  std::vector<float> clusterDEtaToSeed;
  clusterDEtaToSeed.resize(std::max(3, numberOfClusters), 0);
  std::vector<float> clusterDPhiToSeed;
  clusterDPhiToSeed.resize(std::max(3, numberOfClusters), 0);
  float clusterMaxDR     = 999.;
  float clusterMaxDRDPhi = 999.;
  float clusterMaxDRDEta = 999.;
  float clusterMaxDRRawEnergy = 0.;
  
  size_t iclus = 0;
  float maxDR = 0;
  edm::Ptr<reco::CaloCluster> pclus;
  // loop over all clusters that aren't the seed  
  auto clusend = the_sc->clustersEnd();
  for( auto clus = the_sc->clustersBegin(); clus != clusend; ++clus ) {
    pclus = *clus;
    
    if(theseed == pclus ) 
      continue;
    clusterRawEnergy[iclus]  = pclus->energy();
    clusterDPhiToSeed[iclus] = reco::deltaPhi(pclus->phi(),theseed->phi());
    clusterDEtaToSeed[iclus] = pclus->eta() - theseed->eta();
    
    // find cluster with max dR
    const auto the_dr = reco::deltaR(*pclus, *theseed);
    if(the_dr > maxDR) {
      maxDR = the_dr;
      clusterMaxDR = maxDR;
      clusterMaxDRDPhi = clusterDPhiToSeed[iclus];
      clusterMaxDRDEta = clusterDEtaToSeed[iclus];
      clusterMaxDRRawEnergy = clusterRawEnergy[iclus];
    }      
    ++iclus;
  }
  
  eval[16] = clusterMaxDR;
  eval[17] = clusterMaxDRDPhi;
  eval[18] = clusterMaxDRDEta;
  eval[19] = clusterMaxDRRawEnergy/raw_energy;
  eval[20] = clusterRawEnergy[0]/raw_energy;
  eval[21] = clusterRawEnergy[1]/raw_energy;
  eval[22] = clusterRawEnergy[2]/raw_energy;
  eval[23] = clusterDPhiToSeed[0];
  eval[24] = clusterDPhiToSeed[1];
  eval[25] = clusterDPhiToSeed[2];
  eval[26] = clusterDEtaToSeed[0];
  eval[27] = clusterDEtaToSeed[1];
  eval[28] = clusterDEtaToSeed[2];
  
  // calculate coordinate variables
  const bool iseb = ele.isEB();  
  float dummy;
  int iPhi;
  int iEta;
  float cryPhi;
  float cryEta;
  EcalClusterLocal _ecalLocal;
  if (ele.isEB()) 
    _ecalLocal.localCoordsEB(*theseed, *iSetup_, cryEta, cryPhi, iEta, iPhi, dummy, dummy);
  else 
    _ecalLocal.localCoordsEE(*theseed, *iSetup_, cryEta, cryPhi, iEta, iPhi, dummy, dummy);

  if (iseb) {
    eval[29] = cryEta;
    eval[30] = cryPhi;
    eval[31] = iEta;
    eval[32] = iPhi;
  } else {
    eval[29] = the_sc->preshowerEnergy()/the_sc->rawEnergy();
  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  constexpr double meanlimlow  = 0.2;
  constexpr double meanlimhigh = 2.0;
  constexpr double meanoffset  = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
  constexpr double meanscale   = 0.5*(meanlimhigh-meanlimlow);
  
  constexpr double sigmalimlow  = 0.0002;
  constexpr double sigmalimhigh = 0.5;
  constexpr double sigmaoffset  = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
  constexpr double sigmascale   = 0.5*(sigmalimhigh-sigmalimlow);  
  
  int coridx = 0;
  if (!iseb)
    coridx = 1;
    
  //these are the actual BDT responses
  double rawmean = e_forestH_mean_[coridx]->GetResponse(eval.data());
  double rawsigma = e_forestH_sigma_[coridx]->GetResponse(eval.data());
  
  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);
  
  //regression target is ln(Etrue/Eraw)
  //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
  double ecor = mean*(eval[1]);
  if (!iseb)  
    ecor = mean*(eval[1]+the_sc->preshowerEnergy());
  const double sigmacor = sigma*ecor;
  
  ele.setCorrectedEcalEnergy(ecor);
  ele.setCorrectedEcalEnergyError(sigmacor);
    
  // E-p combination 
  //std::array<float, 11> eval_ep;
  float eval_ep[11];

  const float ep = ele.trackMomentumAtVtx().R();
  const float tot_energy = the_sc->rawEnergy()+the_sc->preshowerEnergy();
  const float momentumError = ele.trackMomentumError();
  const float trkMomentumRelError = ele.trackMomentumError()/ep;
  const float eOverP = tot_energy*mean/ep;
  eval_ep[0] = tot_energy*mean;
  eval_ep[1] = sigma/mean;
  eval_ep[2] = ep; 
  eval_ep[3] = trkMomentumRelError;
  eval_ep[4] = sigma/mean/trkMomentumRelError;
  eval_ep[5] = tot_energy*mean/ep;
  eval_ep[6] = tot_energy*mean/ep*sqrt(sigma/mean*sigma/mean+trkMomentumRelError*trkMomentumRelError);
  eval_ep[7] = ele.ecalDriven();
  eval_ep[8] = ele.trackerDrivenSeed();
  eval_ep[9] = int(ele.classification());//eleClass;
  eval_ep[10] = iseb;
  
  // CODE FOR FUTURE SEMI_PARAMETRIC
  //double rawweight = ep_forestH_mean_[coridx]->GetResponse(eval_ep.data());
  ////rawsigma = ep_forestH_sigma_[coridx]->GetResponse(eval.data());
  //double weight = meanoffset + meanscale*vdt::fast_sin(rawweight);
  ////sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);

  // CODE FOR STANDARD BDT
  double weight = 0.;
  if ( eOverP > 0.025 && 
       std::abs(ep-ecor) < 15.*std::sqrt( momentumError*momentumError + sigmacor*sigmacor ) &&
       (!applyExtraHighEnergyProtection_ || ((momentumError < 10.*ep) || (ecor < 200.)))
       ) {
    // protection against crazy track measurement
    weight = ep_forestH_weight_->GetResponse(eval_ep);
    if(weight>1.) 
      weight = 1.;
    else if(weight<0.) 
      weight = 0.;
  }

  double combinedMomentum = weight*ele.trackMomentumAtVtx().R() + (1.-weight)*ecor;
  double combinedMomentumError = sqrt(weight*weight*ele.trackMomentumError()*ele.trackMomentumError() + (1.-weight)*(1.-weight)*sigmacor*sigmacor);

  math::XYZTLorentzVector oldMomentum = ele.p4();
  math::XYZTLorentzVector newMomentum = math::XYZTLorentzVector(oldMomentum.x()*combinedMomentum/oldMomentum.t(),
								oldMomentum.y()*combinedMomentum/oldMomentum.t(),
								oldMomentum.z()*combinedMomentum/oldMomentum.t(),
								combinedMomentum);
 
  //ele.correctEcalEnergy(combinedMomentum, combinedMomentumError);
  ele.correctMomentum(newMomentum, ele.trackMomentumError(), combinedMomentumError);
}

void EGRegressionModifierV1::modifyObject(pat::Electron& ele) const {
  modifyObject(static_cast<reco::GsfElectron&>(ele));
}

void EGRegressionModifierV1::modifyObject(reco::Photon& pho) const {
  // regression calculation needs no additional valuemaps
  
  std::array<float, 35> eval;
  const reco::SuperClusterRef& the_sc = pho.superCluster();
  const edm::Ptr<reco::CaloCluster>& theseed = the_sc->seed();
  
  const int numberOfClusters =  the_sc->clusters().size();
  const bool missing_clusters = !the_sc->clusters()[numberOfClusters-1].isAvailable();

  if( missing_clusters ) return ; // do not apply corrections in case of missing info (slimmed MiniAOD electrons)

  const double raw_energy = the_sc->rawEnergy(); 
  const auto& ess = pho.showerShapeVariables();

  // SET INPUTS
  eval[0]  = raw_energy;
  eval[1]  = pho.r9();
  eval[2]  = the_sc->etaWidth();
  eval[3]  = the_sc->phiWidth(); 
  eval[4]  = std::max(0,numberOfClusters - 1);
  eval[5]  = pho.hadronicOverEm();
  eval[6]  = rhoValue_;
  eval[7]  = nVtx_;  
  eval[8] = theseed->eta()-the_sc->position().Eta();
  eval[9] = reco::deltaPhi(theseed->phi(),the_sc->position().Phi());
  eval[10] = theseed->energy()/raw_energy;
  eval[11] = ess.e3x3/ess.e5x5;
  eval[12] = ess.sigmaIetaIeta;  
  eval[13] = ess.sigmaIphiIphi;
  eval[14] = ess.sigmaIetaIphi/(ess.sigmaIphiIphi*ess.sigmaIetaIeta);
  eval[15] = ess.maxEnergyXtal/ess.e5x5;
  eval[16] = ess.e2nd/ess.e5x5;
  eval[17] = ess.eTop/ess.e5x5;
  eval[18] = ess.eBottom/ess.e5x5;
  eval[19] = ess.eLeft/ess.e5x5;
  eval[20] = ess.eRight/ess.e5x5;  
  eval[21] = ess.e2x5Max/ess.e5x5;
  eval[22] = ess.e2x5Left/ess.e5x5;
  eval[23] = ess.e2x5Right/ess.e5x5;
  eval[24] = ess.e2x5Top/ess.e5x5;
  eval[25] = ess.e2x5Bottom/ess.e5x5;

  const bool iseb = pho.isEB();
  if (iseb) {
    EBDetId ebseedid(theseed->seed());
    eval[26] = pho.e5x5()/theseed->energy();
    int ieta = ebseedid.ieta();
    int iphi = ebseedid.iphi();
    eval[27] = ieta;
    eval[28] = iphi;
    int signieta = ieta > 0 ? +1 : -1; /// this is 1*abs(ieta)/ieta in original training
    eval[29] = (ieta-signieta)%5;
    eval[30] = (iphi-1)%2;
    //    eval[31] = (abs(ieta)<=25)*((ieta-signieta)%25) + (abs(ieta)>25)*((ieta-26*signieta)%20); //%25 is unnescessary in this formula
    eval[31] = (abs(ieta)<=25)*((ieta-signieta)) + (abs(ieta)>25)*((ieta-26*signieta)%20);  
    eval[32] = (iphi-1)%20;
    eval[33] = ieta;  /// duplicated variables but this was trained like that
    eval[34] = iphi;  /// duplicated variables but this was trained like that
  } else {
    EEDetId eeseedid(theseed->seed());
    eval[26] = the_sc->preshowerEnergy()/raw_energy;
    eval[27] = the_sc->preshowerEnergyPlane1()/raw_energy;
    eval[28] = the_sc->preshowerEnergyPlane2()/raw_energy;
    eval[29] = eeseedid.ix();
    eval[30] = eeseedid.iy();
  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  const double meanlimlow  = 0.2;
  const double meanlimhigh = 2.0;
  const double meanoffset  = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
  const double meanscale   = 0.5*(meanlimhigh-meanlimlow);
  
  const double sigmalimlow  = 0.0002;
  const double sigmalimhigh = 0.5;
  const double sigmaoffset  = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
  const double sigmascale   = 0.5*(sigmalimhigh-sigmalimlow);  
 
  int coridx = 0;
  if (!iseb)
    coridx = 1;

  //these are the actual BDT responses
  double rawmean = ph_forestH_mean_[coridx]->GetResponse(eval.data());
  double rawsigma = ph_forestH_sigma_[coridx]->GetResponse(eval.data());
  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);

  //regression target is ln(Etrue/Eraw)
  //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
  double ecor = mean*eval[0];
  if (!iseb) 
    ecor = mean*(eval[0]+the_sc->preshowerEnergy());

  double sigmacor = sigma*ecor;
  pho.setCorrectedEnergy(reco::Photon::P4type::regression2, ecor, sigmacor, true);     
}

void EGRegressionModifierV1::modifyObject(pat::Photon& pho) const {
  modifyObject(static_cast<reco::Photon&>(pho));
}
