#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/RegressionHelper.h"

#include "TFile.h"
#include <vdt/vdtMath.h>

namespace {
  //const edm::EDGetTokenT<edm::ValueMap<float> > empty_token;
  const edm::InputTag empty_tag;
}

#include <unordered_map>

class EGExtraInfoModifierFromDB : public ModifyObjectValueBase {
public:
  typedef edm::EDGetTokenT<edm::ValueMap<float> > ValMapFloatToken;
  typedef edm::EDGetTokenT<edm::ValueMap<int> > ValMapIntToken;

  struct electron_config {
    edm::InputTag electron_src;
    edm::EDGetTokenT<edm::View<pat::Electron> > tok_electron_src;
    edm::InputTag sigmaIetaIphi;
    edm::InputTag eMax;
    edm::InputTag e2nd;
    edm::InputTag eTop;
    edm::InputTag eBottom;
    edm::InputTag eLeft;
    edm::InputTag eRight;
    edm::InputTag e2x5Left;
    edm::InputTag e2x5Right;
    edm::InputTag e2x5Top; 
    edm::InputTag e2x5Bottom;
    edm::InputTag e3x3;
    edm::InputTag iPhi;
    edm::InputTag iEta;
    edm::InputTag cryPhi;
    edm::InputTag cryEta;
    ValMapFloatToken tok_sigmaIetaIphi;
    ValMapFloatToken tok_eMax;
    ValMapFloatToken tok_e2nd;
    ValMapFloatToken tok_eTop;
    ValMapFloatToken tok_eLeft;
    ValMapFloatToken tok_eBottom;
    ValMapFloatToken tok_eRight;
    ValMapFloatToken tok_e2x5Left;
    ValMapFloatToken tok_e2x5Right;
    ValMapFloatToken tok_e2x5Top;
    ValMapFloatToken tok_e2x5Bottom;
    ValMapFloatToken tok_e3x3;
    ValMapIntToken tok_iPhi;
    ValMapIntToken tok_iEta;
    ValMapFloatToken tok_cryPhi;
    ValMapFloatToken tok_cryEta;	
    std::string ecalRegressionWeightFile;
  };

  struct photon_config {
    std::string regressionWeightFile;
    edm::InputTag photon_src;
    edm::EDGetTokenT<edm::View<pat::Photon> > tok_photon_src;
    edm::InputTag sigmaIetaIphi;
    edm::InputTag sigmaIphiIphi;
    edm::InputTag e2x5Max;
    edm::InputTag e2x5Left;
    edm::InputTag e2x5Right;
    edm::InputTag e2x5Top;
    edm::InputTag e2x5Bottom;
    ValMapFloatToken tok_sigmaIetaIphi;
    ValMapFloatToken tok_sigmaIphiIphi;
    ValMapFloatToken tok_e2x5Max;
    ValMapFloatToken tok_e2x5Left;
    ValMapFloatToken tok_e2x5Right;
    ValMapFloatToken tok_e2x5Top;
    ValMapFloatToken tok_e2x5Bottom;
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

  std::vector<std::string> e_condnames_mean_50ns;
  std::vector<std::string> e_condnames_sigma_50ns;
  std::vector<std::string> e_condnames_mean_25ns;
  std::vector<std::string> e_condnames_sigma_25ns;

  std::vector<std::string> ph_condnames_mean_50ns;
  std::vector<std::string> ph_condnames_sigma_50ns;
  std::vector<std::string> ph_condnames_mean_25ns;
  std::vector<std::string> ph_condnames_sigma_25ns;
  
  std::vector<const GBRForestD*> ph_forestH_mean_;
  std::vector<const GBRForestD*> ph_forestH_sigma_; 
  std::vector<const GBRForestD*> e_forestH_mean_;
  std::vector<const GBRForestD*> e_forestH_sigma_; 
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
    
    //if (conf.exists("ecalRefinedRegressionWeightFile")) {
    e_conf.ecalRegressionWeightFile = electrons.getParameter<std::string>("ecalRefinedRegressionWeightFile");
    //} else {
    //	e_conf.ecalRegressionWeightFile = "";
    //}
    e_conf.sigmaIetaIphi = electrons.getParameter<edm::InputTag>("sigmaIetaIphi");
    e_conf.eMax          = electrons.getParameter<edm::InputTag>("eMax");
    e_conf.e2nd          = electrons.getParameter<edm::InputTag>("e2nd");
    e_conf.eTop          = electrons.getParameter<edm::InputTag>("eTop");
    e_conf.eBottom       = electrons.getParameter<edm::InputTag>("eBottom");
    e_conf.eLeft         = electrons.getParameter<edm::InputTag>("eLeft");
    e_conf.eRight        = electrons.getParameter<edm::InputTag>("eRight");
    e_conf.e2x5Left      = electrons.getParameter<edm::InputTag>("e2x5Left");
    e_conf.e2x5Right     = electrons.getParameter<edm::InputTag>("e2x5Right");
    e_conf.e2x5Top       = electrons.getParameter<edm::InputTag>("e2x5Top");
    e_conf.e2x5Bottom    = electrons.getParameter<edm::InputTag>("e2x5Bottom");
    e_conf.e3x3          = electrons.getParameter<edm::InputTag>("e3x3");
    e_conf.iPhi          = electrons.getParameter<edm::InputTag>("iPhi");
    e_conf.iEta          = electrons.getParameter<edm::InputTag>("iEta");
    e_conf.cryPhi        = electrons.getParameter<edm::InputTag>("cryPhi");
    e_conf.cryEta        = electrons.getParameter<edm::InputTag>("cryEta");
    
    e_condnames_mean_50ns  = electrons.getParameter<std::vector<std::string> >("conditionsMean50ns");
    e_condnames_sigma_50ns = electrons.getParameter<std::vector<std::string> >("conditionsSigma50ns");
    e_condnames_mean_25ns  = electrons.getParameter<std::vector<std::string> >("conditionsMean25ns");
    e_condnames_sigma_25ns = electrons.getParameter<std::vector<std::string> >("conditionsSigma25ns");
  }
  
  if( conf.exists("photon_config") ) { 
    const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("photon_config");
    if (conf.exists("photonRegressionWeightFile")) {
      ph_conf.regressionWeightFile = photons.getParameter<std::string>("photonRegressionWeightFile");
    } else {
      ph_conf.regressionWeightFile = "";
    }
    
    ph_condnames_mean_50ns = photons.getParameter<std::vector<std::string>>("conditionsMean50ns");
    ph_condnames_sigma_50ns = photons.getParameter<std::vector<std::string>>("conditionsSigma50ns");
    ph_condnames_mean_25ns = photons.getParameter<std::vector<std::string>>("conditionsMean25ns");
    ph_condnames_sigma_25ns = photons.getParameter<std::vector<std::string>>("conditionsSigma25ns");

    if( photons.exists(photonSrc) ) 
      ph_conf.photon_src = photons.getParameter<edm::InputTag>(photonSrc);

    ph_conf.sigmaIetaIphi = photons.getParameter<edm::InputTag>("sigmaIetaIphi");
    ph_conf.sigmaIphiIphi = photons.getParameter<edm::InputTag>("sigmaIphiIphi");
    ph_conf.e2x5Max = photons.getParameter<edm::InputTag>("e2x5Max");
    ph_conf.e2x5Left = photons.getParameter<edm::InputTag>("e2x5Left");
    ph_conf.e2x5Right = photons.getParameter<edm::InputTag>("e2x5Right");
    ph_conf.e2x5Top = photons.getParameter<edm::InputTag>("e2x5Top");
    ph_conf.e2x5Bottom = photons.getParameter<edm::InputTag>("e2x5Bottom");    
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
  
  if( !e_conf.tok_electron_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Electron> > eles;
    evt.getByToken(e_conf.tok_electron_src, eles);
    
    for( unsigned i = 0; i < eles->size(); ++i ) {
      edm::Ptr<pat::Electron> ptr = eles->ptrAt(i);
      eles_by_oop[ptr->originalObjectRef().key()] = ptr;
    }    
  }

  get_product(evt, e_conf.tok_sigmaIetaIphi, ele_vmaps);
  get_product(evt, e_conf.tok_eMax,          ele_vmaps);
  get_product(evt, e_conf.tok_e2nd,          ele_vmaps);
  get_product(evt, e_conf.tok_eTop,          ele_vmaps);
  get_product(evt, e_conf.tok_eBottom,       ele_vmaps);
  get_product(evt, e_conf.tok_eLeft,         ele_vmaps);
  get_product(evt, e_conf.tok_eRight,        ele_vmaps);
  get_product(evt, e_conf.tok_e2x5Left,      ele_vmaps);
  get_product(evt, e_conf.tok_e2x5Right,     ele_vmaps);
  get_product(evt, e_conf.tok_e2x5Top,       ele_vmaps);
  get_product(evt, e_conf.tok_e2x5Bottom,    ele_vmaps);
  get_product(evt, e_conf.tok_e3x3,          ele_vmaps);
  get_product(evt, e_conf.tok_iPhi,          ele_int_vmaps);
  get_product(evt, e_conf.tok_iEta,          ele_int_vmaps);
  get_product(evt, e_conf.tok_cryPhi,        ele_vmaps);
  get_product(evt, e_conf.tok_cryEta,        ele_vmaps);
  
  if( !ph_conf.tok_photon_src.isUninitialized() ) {
    edm::Handle<edm::View<pat::Photon> > phos;
    evt.getByToken(ph_conf.tok_photon_src,phos);
  
    for( unsigned i = 0; i < phos->size(); ++i ) {
      edm::Ptr<pat::Photon> ptr = phos->ptrAt(i);
      phos_by_oop[ptr->originalObjectRef().key()] = ptr;
    }
  }
   
  get_product(evt, ph_conf.tok_sigmaIetaIphi, pho_vmaps);
  get_product(evt, ph_conf.tok_sigmaIphiIphi, pho_vmaps);
  get_product(evt, ph_conf.tok_e2x5Max,       pho_vmaps);
  get_product(evt, ph_conf.tok_e2x5Left,      pho_vmaps);
  get_product(evt, ph_conf.tok_e2x5Right,     pho_vmaps);
  get_product(evt, ph_conf.tok_e2x5Top,       pho_vmaps);
  get_product(evt, ph_conf.tok_e2x5Bottom,    pho_vmaps);
  
  if (autoDetectBunchSpacing_) {
    edm::Handle<int> bunchSpacingH;
    evt.getByToken(bunchSpacingToken_,bunchSpacingH);
    bunchspacing_ = *bunchSpacingH;
  }

  edm::Handle<double> rhoH;
  evt.getByToken(rhoToken_, rhoH);
  rhoValue_ = *rhoH;
  
  evt.getByToken(vtxToken_, vtxH_);
  nVtx_ = vtxH_->size();
}

void EGExtraInfoModifierFromDB::setEventContent(const edm::EventSetup& evs) {

  edm::ESHandle<GBRForestD> forestEH;

  const std::vector<std::string> ph_condnames_mean  = (bunchspacing_ == 25) ? ph_condnames_mean_25ns  : ph_condnames_mean_50ns;
  const std::vector<std::string> ph_condnames_sigma = (bunchspacing_ == 25) ? ph_condnames_sigma_25ns : ph_condnames_sigma_50ns;
  
  std::cout << ph_conf.regressionWeightFile << " " << ph_condnames_mean[0] << std::endl;
  unsigned int ncor = ph_condnames_mean.size();
  if (ph_conf.regressionWeightFile == "") {
    for (unsigned int icor=0; icor<ncor; ++icor) {
      evs.get<GBRDWrapperRcd>().get(ph_condnames_mean[icor], forestEH);
      ph_forestH_mean_.push_back(forestEH.product());
      evs.get<GBRDWrapperRcd>().get(ph_condnames_sigma[icor], forestEH);
      ph_forestH_sigma_.push_back(forestEH.product());
    } 
  } else {
    //load forests from file  
    ph_forestH_mean_.resize(ncor);
    ph_forestH_sigma_.resize(ncor);  
  
    TFile *fgbr = TFile::Open(edm::FileInPath(ph_conf.regressionWeightFile.c_str()).fullPath().c_str());
    fgbr->GetObject(ph_condnames_mean[0].c_str(),  ph_forestH_mean_[0]);
    fgbr->GetObject(ph_condnames_mean[1].c_str(),  ph_forestH_mean_[1]);
    fgbr->GetObject(ph_condnames_sigma[0].c_str(), ph_forestH_sigma_[0]);
    fgbr->GetObject(ph_condnames_sigma[1].c_str(), ph_forestH_sigma_[1]);
    fgbr->Close();
  }

  const std::vector<std::string> e_condnames_mean  = (bunchspacing_ == 25) ? e_condnames_mean_25ns  : e_condnames_mean_50ns;
  const std::vector<std::string> e_condnames_sigma = (bunchspacing_ == 25) ? e_condnames_sigma_25ns : e_condnames_sigma_50ns;

  std::cout << e_conf.ecalRegressionWeightFile << " " << e_condnames_mean[0] << std::endl;    
  unsigned int encor = e_condnames_mean.size();
  if (e_conf.ecalRegressionWeightFile == "") {
    for (unsigned int icor=0; icor<encor; ++icor) {
      evs.get<GBRDWrapperRcd>().get(e_condnames_mean[icor], forestEH);
      e_forestH_mean_.push_back(forestEH.product());
      evs.get<GBRDWrapperRcd>().get(e_condnames_sigma[icor], forestEH);
      e_forestH_sigma_.push_back(forestEH.product());
    }
  } else {
    //load forests from file  
    e_forestH_mean_.resize(encor);
    e_forestH_sigma_.resize(encor);  
    
    TFile *fgbr = TFile::Open(edm::FileInPath(e_conf.ecalRegressionWeightFile.c_str()).fullPath().c_str());
    fgbr->GetObject(e_condnames_mean[0].c_str(), e_forestH_mean_[0]);
    fgbr->GetObject(e_condnames_mean[1].c_str(), e_forestH_mean_[1]);
    fgbr->GetObject(e_condnames_sigma[0].c_str(), e_forestH_sigma_[0]);
    fgbr->GetObject(e_condnames_sigma[1].c_str(), e_forestH_sigma_[1]);
    fgbr->Close();
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
  
  make_consumes(e_conf.sigmaIetaIphi, e_conf.tok_sigmaIetaIphi,sumes);
  make_consumes(e_conf.eMax,          e_conf.tok_eMax,sumes);
  make_consumes(e_conf.e2nd,          e_conf.tok_e2nd,sumes);
  make_consumes(e_conf.eTop,          e_conf.tok_eTop,sumes);
  make_consumes(e_conf.eBottom,       e_conf.tok_eBottom,sumes);
  make_consumes(e_conf.eLeft,         e_conf.tok_eLeft,sumes);
  make_consumes(e_conf.eRight,        e_conf.tok_eRight,sumes);
  make_consumes(e_conf.e2x5Left,      e_conf.tok_e2x5Left,sumes);
  make_consumes(e_conf.e2x5Right,     e_conf.tok_e2x5Right,sumes);
  make_consumes(e_conf.e2x5Top,       e_conf.tok_e2x5Top,sumes);
  make_consumes(e_conf.e2x5Bottom,    e_conf.tok_e2x5Bottom,sumes);   
  make_consumes(e_conf.e3x3,          e_conf.tok_e3x3,sumes); 
  make_int_consumes(e_conf.iPhi,      e_conf.tok_iPhi,sumes); 
  make_int_consumes(e_conf.iEta,      e_conf.tok_iEta,sumes); 
  make_consumes(e_conf.cryPhi,        e_conf.tok_cryPhi,sumes); 
  make_consumes(e_conf.cryEta,        e_conf.tok_cryEta,sumes); 
  
  // setup photons 
  if(!(empty_tag == ph_conf.photon_src)) 
    ph_conf.tok_photon_src = sumes.consumes<edm::View<pat::Photon> >(ph_conf.photon_src);
  
  make_consumes(ph_conf.sigmaIetaIphi,ph_conf.tok_sigmaIetaIphi,sumes);
  make_consumes(ph_conf.sigmaIphiIphi,ph_conf.tok_sigmaIphiIphi,sumes);
  make_consumes(ph_conf.e2x5Max,ph_conf.tok_e2x5Max,sumes);
  make_consumes(ph_conf.e2x5Left,ph_conf.tok_e2x5Left,sumes);
  make_consumes(ph_conf.e2x5Right,ph_conf.tok_e2x5Right,sumes);
  make_consumes(ph_conf.e2x5Top,ph_conf.tok_e2x5Top,sumes);
  make_consumes(ph_conf.e2x5Bottom,ph_conf.tok_e2x5Bottom,sumes);
  
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

  std::array<float, 37> eval;
  
  reco::SuperClusterRef sc = ele.superCluster();
  edm::Ptr<reco::CaloCluster> theseed = sc->seed();
  
  // SET INPUTS
  eval[0]  = sc->rawEnergy();
  eval[1]  = sc->position().Eta();
  eval[2]  = sc->position().Phi();
  eval[3]  = ele.full5x5_r9();
  const int N_ECAL = sc->clustersEnd() - sc->clustersBegin();
  eval[4]  = sc->etaWidth();
  eval[5]  = sc->phiWidth(); 
  eval[6]  = std::max(0,N_ECAL - 1);
  eval[7]  = ele.hadronicOverEm();
  eval[8]  = rhoValue_;
  eval[9]  = nVtx_;  
  eval[10] = theseed->eta()-sc->position().Eta();
  eval[11] = atan2(sin(theseed->phi()-sc->position().Phi()), cos(theseed->phi()-sc->position().Phi()));
  eval[12] = theseed->energy();
  
  float e3x3, cryPhi, cryEta; 
  int iPhi=0, iEta=0;
  float eMax, e2nd, eTop, eBottom, eLeft, eRight;
  float sieip, e2x5Left, e2x5Right, e2x5Top, e2x5Bottom;
  assignValue(ptr, e_conf.tok_sigmaIetaIphi, ele_vmaps, sieip);
  assignValue(ptr, e_conf.tok_eMax, ele_vmaps, eMax);
  assignValue(ptr, e_conf.tok_e2nd, ele_vmaps, e2nd);
  assignValue(ptr, e_conf.tok_eTop, ele_vmaps, eTop);
  assignValue(ptr, e_conf.tok_eBottom, ele_vmaps, eBottom);
  assignValue(ptr, e_conf.tok_eLeft, ele_vmaps, eLeft);
  assignValue(ptr, e_conf.tok_eRight, ele_vmaps, eRight);
  assignValue(ptr, e_conf.tok_e2x5Left, ele_vmaps, e2x5Left);
  assignValue(ptr, e_conf.tok_e2x5Right, ele_vmaps, e2x5Right);
  assignValue(ptr, e_conf.tok_e2x5Top, ele_vmaps, e2x5Top);
  assignValue(ptr, e_conf.tok_e2x5Bottom, ele_vmaps, e2x5Bottom);
  assignValue(ptr, e_conf.tok_e3x3, ele_vmaps, e3x3);
  assignValue(ptr, e_conf.tok_iPhi, ele_int_vmaps, iPhi);
  assignValue(ptr, e_conf.tok_iEta, ele_int_vmaps, iEta);
  assignValue(ptr, e_conf.tok_cryPhi, ele_vmaps, cryPhi);
  assignValue(ptr, e_conf.tok_cryEta, ele_vmaps, cryEta);
  
  eval[13] = e3x3/ele.full5x5_e5x5();
  eval[14] = ele.full5x5_sigmaIetaIeta();
  eval[15] = ele.full5x5_sigmaIphiIphi();
  eval[16] = sieip;
  eval[17] = eMax/ele.full5x5_e5x5();
  eval[18] = e2nd/ele.full5x5_e5x5();
  eval[19] = eTop/ele.full5x5_e5x5();
  eval[20] = eBottom/ele.full5x5_e5x5();
  eval[21] = eLeft/ele.full5x5_e5x5();
  eval[22] = eRight/ele.full5x5_e5x5();  
  eval[23] = ele.full5x5_e2x5Max()/ele.full5x5_e5x5();
  eval[24] = e2x5Left/ele.full5x5_e5x5();
  eval[25] = e2x5Right/ele.full5x5_e5x5();
  eval[26] = e2x5Top/ele.full5x5_e5x5();
  eval[27] = e2x5Bottom/ele.full5x5_e5x5();
  
  bool iseb = ele.isEB();
  
  if (iseb) {
    eval[28] = ele.full5x5_e5x5()/sc->seed()->energy();
    eval[29] = iEta;
    eval[30] = iPhi;
    eval[31] = (iEta-1*abs(iEta)/iEta)%5;
    eval[32] = (iPhi-1)%2;       
    eval[33] = (abs(iEta)<=25)*((iEta-1*abs(iEta)/iEta)%25) + (abs(iEta)>25)*((iEta-26*abs(iEta)/iEta)%20);
    eval[33] = (iPhi-1)%20; 
    eval[34] = cryPhi;
    eval[35] = cryEta;
  } else {
    eval[28] = sc->preshowerEnergy()/sc->rawEnergy();
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
  double rawmean = e_forestH_mean_[coridx]->GetResponse(eval.data());
  double rawsigma = e_forestH_sigma_[coridx]->GetResponse(eval.data());
  
  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);
  
  //regression target is ln(Etrue/Eraw)
  //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
  double ecor = mean*eval[0];
  double sigmacor = sigma*ecor;
  
  ele.setCorrectedEcalEnergy(ecor);
  ele.setCorrectedEcalEnergyError(sigmacor);
  
  // FIXME THE e-p combination is missing
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
  
  std::array<float, 37> eval;

  reco::SuperClusterRef sc = pho.superCluster();
  edm::Ptr<reco::CaloCluster> theseed = sc->seed();
  
  // SET INPUTS
  eval[0]  = sc->rawEnergy();
  eval[1]  = sc->position().Eta();
  eval[2]  = sc->position().Phi();
  eval[3]  = pho.full5x5_r9();
  eval[4]  = sc->etaWidth();
  eval[5]  = sc->phiWidth(); 
  const int N_ECAL = sc->clustersEnd() - sc->clustersBegin();
  eval[6]  = std::max(0,N_ECAL - 1);
  eval[7]  = pho.hadronicOverEm();
  eval[8]  = rhoValue_;
  eval[9]  = nVtx_;  
  eval[10] = theseed->eta()-sc->position().Eta();
  eval[11] = atan2(sin(theseed->phi()-sc->position().Phi()), cos(theseed->phi()-sc->position().Phi()));
  eval[12] = pho.seedEnergy();
  eval[13] = pho.full5x5_e3x3()/pho.full5x5_e5x5();
  eval[14] = pho.full5x5_sigmaIetaIeta();

  float sipip, sieip, e2x5Max, e2x5Left, e2x5Right, e2x5Top, e2x5Bottom;
  assignValue(ptr, ph_conf.tok_sigmaIphiIphi, pho_vmaps, sipip);
  assignValue(ptr, ph_conf.tok_sigmaIetaIphi, pho_vmaps, sieip);
  assignValue(ptr, ph_conf.tok_e2x5Max, pho_vmaps, e2x5Max);
  assignValue(ptr, ph_conf.tok_e2x5Left, pho_vmaps, e2x5Left);
  assignValue(ptr, ph_conf.tok_e2x5Right, pho_vmaps, e2x5Right);
  assignValue(ptr, ph_conf.tok_e2x5Top, pho_vmaps, e2x5Top);
  assignValue(ptr, ph_conf.tok_e2x5Bottom, pho_vmaps, e2x5Bottom);
  
  eval[15] = sipip;
  eval[16] = sieip;
  eval[17] = pho.full5x5_maxEnergyXtal()/pho.full5x5_e5x5();
  eval[18] = pho.e2nd()/pho.full5x5_e5x5();
  eval[19] = pho.eTop()/pho.full5x5_e5x5();
  eval[20] = pho.eBottom()/pho.full5x5_e5x5();
  eval[21] = pho.eLeft()/pho.full5x5_e5x5();
  eval[22] = pho.eRight()/pho.full5x5_e5x5();  
  eval[23] = e2x5Max/pho.full5x5_e5x5();
  eval[24] = e2x5Left/pho.full5x5_e5x5();
  eval[25] = e2x5Right/pho.full5x5_e5x5();
  eval[26] = e2x5Top/pho.full5x5_e5x5();
  eval[27] = e2x5Bottom/pho.full5x5_e5x5();

  bool iseb = pho.isEB();

  if (iseb) {
    int cryIPhi = pho.iPhi();
    int cryIEta = pho.iEta();
    eval[28] = pho.full5x5_e5x5()/pho.seedEnergy();
    eval[29] = cryIEta;
    eval[30] = cryIPhi;
    eval[31] = (cryIEta-1*abs(cryIEta)/cryIEta)%5;
    eval[32] = (cryIPhi-1)%2;       
    eval[33] = (abs(cryIEta)<=25)*((cryIEta-1*abs(cryIEta)/cryIEta)%25) + (abs(cryIEta)>25)*((cryIEta-26*abs(cryIEta)/cryIEta)%20);
    eval[33] = (cryIPhi-1)%20; 
    eval[34] = pho.cryPhi();
    eval[35] = pho.cryEta();
  } else {
    eval[28] = sc->preshowerEnergy()/sc->rawEnergy();
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
  double sigmacor = sigma*ecor;
  
  pho.setCorrectedEnergy(reco::Photon::P4type::regression2, ecor, sigmacor, true);        
}
