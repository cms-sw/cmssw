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

#include <vdt/vdtMath.h>

namespace {
  const edm::InputTag empty_tag;
}

#include <unordered_map>

class EGExtraInfoModifierFromDB : public ModifyObjectValueBase {
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

  EGExtraInfoModifierFromDB(const edm::ParameterSet& conf);
  ~EGExtraInfoModifierFromDB() {};
    
  void setEvent(const edm::Event&) override final;
  void setEventContent(const edm::EventSetup&) override final;
  void setConsumes(edm::ConsumesCollector&) override final;
  
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

  bool autoDetectBunchSpacing_;
  int bunchspacing_;
  edm::InputTag bunchspacingTag_;
  edm::EDGetTokenT<int> bunchSpacingToken_;
  float rhoValue_;
  edm::InputTag rhoTag_;
  edm::EDGetTokenT<double> rhoToken_;
  int nVtx_;
  edm::InputTag vtxTag_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::Handle<reco::VertexCollection> vtxH_;

  std::vector<const GBRForestD*> ph_forestH_mean_;
  std::vector<const GBRForestD*> ph_forestH_sigma_; 
  std::vector<const GBRForestD*> e_forestH_mean_;
  std::vector<const GBRForestD*> e_forestH_sigma_; 
  const GBRForest* ep_forestH_weight_;
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory,
		  EGExtraInfoModifierFromDB,
		  "EGExtraInfoModifierFromDB");

EGExtraInfoModifierFromDB::EGExtraInfoModifierFromDB(const edm::ParameterSet& conf) :
  ModifyObjectValueBase(conf) {

  bunchspacing_ = 450;
  autoDetectBunchSpacing_ = conf.getParameter<bool>("autoDetectBunchSpacing");

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

template<typename T>
inline void get_product(const edm::Event& evt,
                        const edm::EDGetTokenT<edm::ValueMap<T> >& tok,
                        std::unordered_map<unsigned, edm::Handle<edm::ValueMap<T> > >& map) {
  evt.getByToken(tok,map[tok.index()]);
}

void EGExtraInfoModifierFromDB::setEvent(const edm::Event& evt) {
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
    if (evt.isRealData()) {
      edm::RunNumber_t run = evt.run();
      if (run == 178003 ||
          run == 178004 ||
          run == 209089 ||
          run == 209106 ||
          run == 209109 ||
          run == 209146 ||
          run == 209148 ||
          run == 209151) {
        bunchspacing_ = 25;
      }
      else if (run < 253000) {
        bunchspacing_ = 50;
      } 
      else {
	bunchspacing_ = 25;
      }
    } else {
      edm::Handle<int> bunchSpacingH;
      evt.getByToken(bunchSpacingToken_,bunchSpacingH);
      bunchspacing_ = *bunchSpacingH;
    }
  }

  edm::Handle<double> rhoH;
  evt.getByToken(rhoToken_, rhoH);
  rhoValue_ = *rhoH;
  
  evt.getByToken(vtxToken_, vtxH_);
  nVtx_ = vtxH_->size();
}

void EGExtraInfoModifierFromDB::setEventContent(const edm::EventSetup& evs) {

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

void EGExtraInfoModifierFromDB::setConsumes(edm::ConsumesCollector& sumes) {
 
  rhoToken_ = sumes.consumes<double>(rhoTag_);
  vtxToken_ = sumes.consumes<reco::VertexCollection>(vtxTag_);

  if (autoDetectBunchSpacing_)
    bunchSpacingToken_ = sumes.consumes<int>(bunchspacingTag_);

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

template<typename T, typename U, typename V, typename Z>
inline void assignValue(const T& ptr, const U& tok, const V& map, Z& value) {
  if( !tok.isUninitialized() ) value = map.find(tok.index())->second->get(ptr.id(),ptr.key());
}

void EGExtraInfoModifierFromDB::modifyObject(pat::Electron& ele) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference

  edm::Ptr<reco::Candidate> ptr(ele.originalObjectRef());
  if( !e_conf.tok_electron_src.isUninitialized() ) {
    auto key = eles_by_oop.find(ele.originalObjectRef().key());
    if( key != eles_by_oop.end() ) {
      ptr = key->second;
    } else {
      throw cms::Exception("BadElectronKey")
        << "Original object pointer with key = " << ele.originalObjectRef().key() 
        << " not found in cache!";
    }
  }
  std::array<float, 33> eval;
  
  reco::SuperClusterRef sc = ele.superCluster();
  edm::Ptr<reco::CaloCluster> theseed = sc->seed();

  // SET INPUTS
  eval[0]  = nVtx_;  
  eval[1]  = sc->rawEnergy();
  eval[2]  = sc->eta();
  eval[3]  = sc->phi();
  eval[4]  = sc->etaWidth();
  eval[5]  = sc->phiWidth(); 
  eval[6]  = ele.r9();
  eval[7]  = theseed->energy()/sc->rawEnergy();
  
  float sieip=0, cryPhi=0, cryEta=0; 
  int iPhi=0, iEta=0;
  float eMax=0, e2nd=0, eTop=0, eBottom=0, eLeft=0, eRight=0;
  float clusterMaxDR=0, clusterMaxDRDPhi=0, clusterMaxDRDEta=0, clusterMaxDRRawEnergy=0;
  float clusterRawEnergy0=0, clusterRawEnergy1=0, clusterRawEnergy2=0;
  float clusterDPhiToSeed0=0, clusterDPhiToSeed1=0, clusterDPhiToSeed2=0;
  float clusterDEtaToSeed0=0, clusterDEtaToSeed1=0, clusterDEtaToSeed2=0;
  
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("sigmaIetaIphi"))->second.second, ele_vmaps, sieip);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("eMax"))->second.second, ele_vmaps, eMax);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("e2nd"))->second.second, ele_vmaps, e2nd);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("eTop"))->second.second, ele_vmaps, eTop);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("eBottom"))->second.second, ele_vmaps, eBottom);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("eLeft"))->second.second, ele_vmaps, eLeft);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("eRight"))->second.second, ele_vmaps, eRight);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterMaxDR"))->second.second, ele_vmaps, clusterMaxDR);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterMaxDRDPhi"))->second.second, ele_vmaps, clusterMaxDRDPhi);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterMaxDRDEta"))->second.second, ele_vmaps, clusterMaxDRDEta);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterMaxDRRawEnergy"))->second.second, ele_vmaps, clusterMaxDRRawEnergy); 
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterRawEnergy0"))->second.second, ele_vmaps, clusterRawEnergy0);	    
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterRawEnergy1"))->second.second, ele_vmaps, clusterRawEnergy1);	    
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterRawEnergy2"))->second.second, ele_vmaps, clusterRawEnergy2);	    
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterDPhiToSeed0"))->second.second, ele_vmaps, clusterDPhiToSeed0); 
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterDPhiToSeed1"))->second.second, ele_vmaps, clusterDPhiToSeed1); 
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterDPhiToSeed2"))->second.second, ele_vmaps, clusterDPhiToSeed2); 
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterDEtaToSeed0"))->second.second, ele_vmaps, clusterDEtaToSeed0); 
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterDEtaToSeed1"))->second.second, ele_vmaps, clusterDEtaToSeed1); 
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("clusterDEtaToSeed2"))->second.second, ele_vmaps, clusterDEtaToSeed2); 
  assignValue(ptr, e_conf.tag_int_token_map.find(std::string("iPhi"))->second.second, ele_int_vmaps, iPhi);
  assignValue(ptr, e_conf.tag_int_token_map.find(std::string("iEta"))->second.second, ele_int_vmaps, iEta);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("cryPhi"))->second.second, ele_vmaps, cryPhi);
  assignValue(ptr, e_conf.tag_float_token_map.find(std::string("cryEta"))->second.second, ele_vmaps, cryEta);

  eval[8]  = eMax/sc->rawEnergy();
  eval[9]  = e2nd/sc->rawEnergy();
  eval[10] = (eLeft+eRight!=0. ? (eLeft-eRight)/(eLeft+eRight) : 0.);
  eval[11] = (eTop+eBottom!=0. ? (eTop-eBottom)/(eTop+eBottom) : 0.);
  eval[12] = ele.sigmaIetaIeta();
  eval[13] = sieip;
  eval[14] = ele.sigmaIphiIphi();
  const int N_ECAL = sc->clustersEnd() - sc->clustersBegin();
  eval[15] = std::max(0,N_ECAL - 1);
  eval[16] = clusterMaxDR;
  eval[17] = clusterMaxDRDPhi;
  eval[18] = clusterMaxDRDEta;
  eval[19] = clusterMaxDRRawEnergy/sc->rawEnergy();
  eval[20] = clusterRawEnergy0/sc->rawEnergy();
  eval[21] = clusterRawEnergy1/sc->rawEnergy();
  eval[22] = clusterRawEnergy2/sc->rawEnergy();
  eval[23] = clusterDPhiToSeed0;
  eval[24] = clusterDPhiToSeed1;
  eval[25] = clusterDPhiToSeed2;
  eval[26] = clusterDEtaToSeed0;
  eval[27] = clusterDEtaToSeed1;
  eval[28] = clusterDEtaToSeed2;
  
  bool iseb = ele.isEB();
  
  if (iseb) {
    eval[29] = cryEta;
    eval[30] = cryPhi;
    eval[31] = iEta;
    eval[32] = iPhi;
  } else {
    eval[29] = sc->preshowerEnergy()/sc->rawEnergy();
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
    ecor = mean*(eval[1]+sc->preshowerEnergy());
  const double sigmacor = sigma*ecor;
  
  ele.setCorrectedEcalEnergy(ecor);
  ele.setCorrectedEcalEnergyError(sigmacor);
    
  // E-p combination 
  //std::array<float, 11> eval_ep;
  float eval_ep[11];

  const float ep = ele.trackMomentumAtVtx().R();
  const float tot_energy = sc->rawEnergy()+sc->preshowerEnergy();
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
       std::abs(ep-ecor) < 15.*std::sqrt( momentumError*momentumError + sigmacor*sigmacor ) ) {
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

void EGExtraInfoModifierFromDB::modifyObject(pat::Photon& pho) const {
  // we encounter two cases here, either we are running AOD -> MINIAOD
  // and the value maps are to the reducedEG object, can use original object ptr
  // or we are running MINIAOD->MINIAOD and we need to fetch the pat objects to reference
  edm::Ptr<reco::Candidate> ptr(pho.originalObjectRef());

  if(!ph_conf.tok_photon_src.isUninitialized()) {
    auto key = phos_by_oop.find(pho.originalObjectRef().key());
    if( key != phos_by_oop.end() ) {
      ptr = key->second;
    } else {
      throw cms::Exception("BadPhotonKey")
        << "Original object pointer with key = " << pho.originalObjectRef().key() << " not found in cache!";
    }
  }
  
  std::array<float, 31> eval;
  reco::SuperClusterRef sc = pho.superCluster();
  edm::Ptr<reco::CaloCluster> theseed = sc->seed();
  
  // SET INPUTS
  eval[0]  = sc->rawEnergy();
  //eval[1]  = sc->position().Eta();
  //eval[2]  = sc->position().Phi();
  eval[1]  = pho.r9();
  eval[2]  = sc->etaWidth();
  eval[3]  = sc->phiWidth(); 
  const int N_ECAL = sc->clustersEnd() - sc->clustersBegin();
  eval[4]  = std::max(0,N_ECAL - 1);
  eval[5]  = pho.hadronicOverEm();
  eval[6]  = rhoValue_;
  eval[7]  = nVtx_;  
  eval[8] = theseed->eta()-sc->position().Eta();
  eval[9] = reco::deltaPhi(theseed->phi(),sc->position().Phi());
  eval[10] = pho.seedEnergy()/sc->rawEnergy();
  eval[11] = pho.e3x3()/pho.e5x5();
  eval[12] = pho.sigmaIetaIeta();

  float sipip=0, sieip=0, e2x5Max=0, e2x5Left=0, e2x5Right=0, e2x5Top=0, e2x5Bottom=0;
  assignValue(ptr, ph_conf.tag_float_token_map.find(std::string("sigmaIetaIphi"))->second.second, pho_vmaps, sieip);
  assignValue(ptr, ph_conf.tag_float_token_map.find(std::string("sigmaIphiIphi"))->second.second, pho_vmaps, sipip);
  assignValue(ptr, ph_conf.tag_float_token_map.find(std::string("e2x5Max"))->second.second, pho_vmaps, e2x5Max);
  assignValue(ptr, ph_conf.tag_float_token_map.find(std::string("e2x5Left"))->second.second, pho_vmaps, e2x5Left);
  assignValue(ptr, ph_conf.tag_float_token_map.find(std::string("e2x5Right"))->second.second, pho_vmaps, e2x5Right);
  assignValue(ptr, ph_conf.tag_float_token_map.find(std::string("e2x5Top"))->second.second, pho_vmaps, e2x5Top);
  assignValue(ptr, ph_conf.tag_float_token_map.find(std::string("e2x5Bottom"))->second.second, pho_vmaps, e2x5Bottom);
  
  eval[13] = sipip;
  eval[14] = sieip;
  eval[15] = pho.maxEnergyXtal()/pho.e5x5();
  eval[16] = pho.e2nd()/pho.e5x5();
  eval[17] = pho.eTop()/pho.e5x5();
  eval[18] = pho.eBottom()/pho.e5x5();
  eval[19] = pho.eLeft()/pho.e5x5();
  eval[20] = pho.eRight()/pho.e5x5();  
  eval[21] = e2x5Max/pho.e5x5();
  eval[22] = e2x5Left/pho.e5x5();
  eval[23] = e2x5Right/pho.e5x5();
  eval[24] = e2x5Top/pho.e5x5();
  eval[25] = e2x5Bottom/pho.e5x5();

  bool iseb = pho.isEB();

  if (iseb) {
    EBDetId ebseedid(theseed->seed());
    eval[26] = pho.e5x5()/pho.seedEnergy();
    eval[27] = ebseedid.ieta();
    eval[28] = ebseedid.iphi();
  } else {
    EEDetId eeseedid(theseed->seed());
    eval[26] = sc->preshowerEnergy()/sc->rawEnergy();
    eval[27] = sc->preshowerEnergyPlane1()/sc->rawEnergy();
    eval[28] = sc->preshowerEnergyPlane2()/sc->rawEnergy();
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
    ecor = mean*(eval[0]+sc->preshowerEnergy());

  double sigmacor = sigma*ecor;
  pho.setCorrectedEnergy(reco::Photon::P4type::regression2, ecor, sigmacor, true);     
}
