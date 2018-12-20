#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "RecoEgamma/EgammaTools/plugins/EGRegressionModifier.h"

class EGRegressionModifierV1 : public ModifyObjectValueBase {
public:

  struct ElectronConfig {
    std::vector<std::string> condnames_mean_50ns;
    std::vector<std::string> condnames_sigma_50ns;
    std::vector<std::string> condnames_mean_25ns;
    std::vector<std::string> condnames_sigma_25ns;
    std::string condnames_weight_50ns;
    std::string condnames_weight_25ns;
  };

  struct PhotonConfig {
    std::vector<std::string> condnames_mean_50ns;
    std::vector<std::string> condnames_sigma_50ns;
    std::vector<std::string> condnames_mean_25ns;
    std::vector<std::string> condnames_sigma_25ns;
  };

  EGRegressionModifierV1(const edm::ParameterSet& conf);
    
  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final;
  void setConsumes(edm::ConsumesCollector&) final;
  
  void modifyObject(reco::GsfElectron&) const final;
  void modifyObject(reco::Photon&) const final;
  
  // just calls reco versions
  void modifyObject(pat::Electron& ele) const final { modifyObject(static_cast<reco::GsfElectron&>(ele)); }
  void modifyObject(pat::Photon& pho) const final { modifyObject(static_cast<reco::Photon&>(pho)); }

private:
  ElectronConfig e_conf;
  PhotonConfig   ph_conf;

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

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, EGRegressionModifierV1, "EGRegressionModifierV1");

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

  const edm::ParameterSet& electrons = conf.getParameter<edm::ParameterSet>("ElectronConfig");
  e_conf.condnames_mean_50ns  = electrons.getParameter<std::vector<std::string> >("regressionKey_50ns");
  e_conf.condnames_sigma_50ns = electrons.getParameter<std::vector<std::string> >("uncertaintyKey_50ns");
  e_conf.condnames_mean_25ns  = electrons.getParameter<std::vector<std::string> >("regressionKey_25ns");
  e_conf.condnames_sigma_25ns = electrons.getParameter<std::vector<std::string> >("uncertaintyKey_25ns");
  e_conf.condnames_weight_50ns  = electrons.getParameter<std::string>("combinationKey_50ns");
  e_conf.condnames_weight_25ns  = electrons.getParameter<std::string>("combinationKey_25ns");
  
  const edm::ParameterSet& photons = conf.getParameter<edm::ParameterSet>("PhotonConfig");
  ph_conf.condnames_mean_50ns = photons.getParameter<std::vector<std::string>>("regressionKey_50ns");
  ph_conf.condnames_sigma_50ns = photons.getParameter<std::vector<std::string>>("uncertaintyKey_50ns");
  ph_conf.condnames_mean_25ns = photons.getParameter<std::vector<std::string>>("regressionKey_25ns");
  ph_conf.condnames_sigma_25ns = photons.getParameter<std::vector<std::string>>("uncertaintyKey_25ns");
}

void EGRegressionModifierV1::setEvent(const edm::Event& evt)
{
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

void EGRegressionModifierV1::setEventContent(const edm::EventSetup& evs)
{
  iSetup_ = &evs;

  ph_forestH_mean_ = retrieveGBRForests(evs, (bunchspacing_ == 25) ? ph_conf.condnames_mean_25ns  : ph_conf.condnames_mean_50ns);
  ph_forestH_sigma_ = retrieveGBRForests(evs, (bunchspacing_ == 25) ? ph_conf.condnames_sigma_25ns  : ph_conf.condnames_sigma_50ns);

  e_forestH_mean_ = retrieveGBRForests(evs, (bunchspacing_ == 25) ? e_conf.condnames_mean_25ns  : e_conf.condnames_mean_50ns);
  e_forestH_sigma_ = retrieveGBRForests(evs, (bunchspacing_ == 25) ? e_conf.condnames_sigma_25ns  : e_conf.condnames_sigma_50ns);

  edm::ESHandle<GBRForest> forestEH;
  const std::string ep_condnames_weight  = (bunchspacing_ == 25) ? e_conf.condnames_weight_25ns  : e_conf.condnames_weight_50ns;
  evs.get<GBRWrapperRcd>().get(ep_condnames_weight, forestEH);
  ep_forestH_weight_ = forestEH.product(); 
}

void EGRegressionModifierV1::setConsumes(edm::ConsumesCollector& sumes) {
 
  rhoToken_ = sumes.consumes<double>(rhoTag_);
  vtxToken_ = sumes.consumes<reco::VertexCollection>(vtxTag_);

  if (autoDetectBunchSpacing_)
    bunchSpacingToken_ = sumes.consumes<unsigned int>(bunchspacingTag_);
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
  // loop over all clusters that aren't the seed  
  for (auto const& pclus : the_sc->clusters())
  {
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
  std::array<float, 11> eval_ep;

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
    weight = std::clamp(ep_forestH_weight_->GetResponse(eval_ep.data()), 0., 1.);
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
