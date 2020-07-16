#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include <TVector3.h>

#include <string>
#include <vector>

namespace {

  std::vector<float> getFeatures(reco::GsfElectronRef const& ele, float rho, float unbiased) {

    float eid_rho = -999.;                  
    float eid_sc_eta = -999.;               
    float eid_shape_full5x5_r9 = -999.;     
    float eid_sc_etaWidth = -999.;          
    float eid_sc_phiWidth = -999.;          
    float eid_shape_full5x5_HoverE = -999.; 
    float eid_trk_nhits = -999.;            
    float eid_trk_chi2red = -999.;          
    float eid_gsf_chi2red = -999.;          
    float eid_brem_frac = -999.;            
    float eid_gsf_nhits = -999.;            
    float eid_match_SC_EoverP = -999.;      
    float eid_match_eclu_EoverP = -999.;    
    float eid_match_SC_dEta   = -999.;      
    float eid_match_SC_dPhi   = -999.;      
    float eid_match_seed_dEta = -999.;      
    float eid_sc_E = -999.;                 
    float eid_trk_p = -999.;                
    float gsf_mode_p = -999.;               
    float core_shFracHits = -999.;          
    float gsf_bdtout1 = -999.;              
    float gsf_dr = -999.;                   
    float trk_dr = -999.; 
    float sc_Nclus = -999.;
    float sc_clus1_nxtal  = -999.;  
    float sc_clus1_dphi = -999.;  
    float sc_clus2_dphi = -999.;  
    float sc_clus1_deta = -999.;  
    float sc_clus2_deta = -999.;  
    float sc_clus1_E = -999.;  
    float sc_clus2_E = -999.;  
    float sc_clus1_E_ov_p = -999.;  
    float sc_clus2_E_ov_p = -999.;    

    // KF tracks                                       
    if ( ele->core().isNonnull() ) {     
      reco::TrackRef trk = ele->closestCtfTrackRef(); 
      if ( trk.isNonnull() ) {
	eid_trk_p = (float)trk->p();                                                   
	eid_trk_nhits = (float)trk->found();                
	eid_trk_chi2red = (float)trk->normalizedChi2();     
	TVector3 trkTV3(0,0,0);
	trkTV3.SetPtEtaPhi(trk->pt(), trk->eta(), trk->phi());  
	TVector3 eleTV3(0,0,0);
	eleTV3.SetPtEtaPhi(ele->pt(), ele->eta(), ele->phi());
	trk_dr = eleTV3.DeltaR(trkTV3);  
      }
    }

    // GSF tracks                                     
    if ( ele->core().isNonnull() ) {
      reco::GsfTrackRef gsf = ele->core()->gsfTrack();
      if ( gsf.isNonnull() ) {
	gsf_mode_p = gsf->pMode();       
	eid_gsf_nhits = (float)gsf->found();       
	eid_gsf_chi2red = gsf->normalizedChi2();       
	TVector3 gsfTV3(0,0,0);
	gsfTV3.SetPtEtaPhi(gsf->ptMode(), gsf->etaMode(), gsf->phiMode());  
	TVector3 eleTV3(0,0,0);
	eleTV3.SetPtEtaPhi(ele->pt(), ele->eta(), ele->phi());
	gsf_dr = eleTV3.DeltaR(gsfTV3);  
      }
    }

    // Super clusters                        
    if ( ele->core().isNonnull() ) {
      reco::SuperClusterRef sc = ele->core()->superCluster();
      if ( sc.isNonnull() ) {
	eid_sc_E = sc->energy();                 
	eid_sc_eta  = sc->eta();                 
	eid_sc_etaWidth = sc->etaWidth();        
	eid_sc_phiWidth = sc->phiWidth();        
	sc_Nclus = sc->clustersSize();          
      }
    }

    // Track-cluster matching              
    if ( ele.isNonnull() ) {
      eid_match_seed_dEta = ele->deltaEtaSeedClusterTrackAtCalo();
      eid_match_eclu_EoverP = (1./ele->ecalEnergy()) - (1./ele->p());
      eid_match_SC_EoverP = ele->eSuperClusterOverP();
      eid_match_SC_dEta = ele->deltaEtaSuperClusterTrackAtVtx();
      eid_match_SC_dPhi = ele->deltaPhiSuperClusterTrackAtVtx();
    }      

    // Shower shape vars        
    if ( ele.isNonnull() ) {
      eid_shape_full5x5_HoverE = ele->full5x5_hcalOverEcal();  
      eid_shape_full5x5_r9     = ele->full5x5_r9();            
    }

    // Misc
    eid_rho = rho; 

    if ( ele.isNonnull() ) {                          
      eid_brem_frac = ele->fbrem();
      core_shFracHits = ele->shFracInnerHits();
    }
    
    // Unbiased BDT from ElectronSeed   
    gsf_bdtout1 = unbiased;

    // Clusters   
    if ( ele->core().isNonnull() ) {
      reco::GsfTrackRef gsf = ele->core()->gsfTrack();
      if ( gsf.isNonnull() ) {
	reco::SuperClusterRef sc = ele->core()->superCluster();
	if ( sc.isNonnull() ) {	
	  
	  // Propagate electron track to ECAL surface 
	  double mass_ = 0.000511*0.000511; 
	  float p2=pow( gsf->p() ,2 );
	  float energy = sqrt(mass_ + p2);
	  math::XYZTLorentzVector mom = math::XYZTLorentzVector(gsf->px(), gsf->py(), gsf->pz(), energy);
	  math::XYZTLorentzVector pos = math::XYZTLorentzVector(gsf->vx(), gsf->vy(), gsf->vz(), 0.);
	  float field_z=3.8;
	  BaseParticlePropagator mypart(RawParticle(mom, pos, gsf->charge()), 0, 0, field_z);
	  mypart.propagateToEcalEntrance(true);    // true only first half loop , false more than one loop 
	  bool reach_ECAL = mypart.getSuccess();   // 0 does not reach ECAL, 1 yes barrel, 2 yes endcaps    

	  // ECAL entry point for track
	  GlobalPoint ecal_pos(mypart.particle().vertex().x(), mypart.particle().vertex().y(), mypart.particle().vertex().z());

	  // Iterate through ECAL clusters and sort in energy
	  int clusNum=0;
	  float maxEne1=-1;
	  float maxEne2=-1;
	  int i1=-1;
	  int i2=-1;
	  try{
	    if(sc->clustersSize()>0 && sc->clustersBegin()!=sc->clustersEnd()){
	      for(auto& cluster : sc->clusters()) {
		if (cluster->energy() > maxEne1){
		  maxEne1=cluster->energy();
		  i1=clusNum;
		}
		clusNum++;
	      }
	      if(sc->clustersSize()>1){
		clusNum=0;
		for(auto& cluster : sc->clusters()) {
		  if (clusNum!=i1) {
		    if (cluster->energy() > maxEne2){
		      maxEne2=cluster->energy();
		      i2=clusNum;
		    }
		  }
		  clusNum++;
		}
	      }
	    } // loop over clusters
	  } catch(...) {
	    std::cout<<"exception caught clusNum="<<clusNum<<" clus size"<<sc->clustersSize()<<" energy="<< sc->energy()<<std::endl;
	  }

	  // Initializations
	  sc_clus1_nxtal  = -999;
	  sc_clus1_dphi   = -999.;
	  sc_clus2_dphi   = -999.;
	  sc_clus1_deta   = -999.;
	  sc_clus2_deta   = -999.;
	  sc_clus1_E      = -999.;
	  sc_clus2_E      = -999.;
	  sc_clus1_E_ov_p = -999.;
	  sc_clus2_E_ov_p = -999.;

	  // track-clusters match
	  clusNum=0;
	  try { 
	    if(sc->clustersSize()>0&& sc->clustersBegin()!=sc->clustersEnd()){
	      for(auto& cluster : sc->clusters()) {
		double pi_=3.1415926535;
		float deta = std::fabs(ecal_pos.eta()-cluster->eta()) ;
		float dphi = std::fabs(ecal_pos.phi()-cluster->phi());
		if (dphi > pi_)  dphi -= 2 * pi_;
		if (ecal_pos.phi()-cluster->phi()<0) dphi=-dphi;
		if (ecal_pos.eta()-cluster->eta()<0) deta=-deta;

		if (clusNum==i1) {
		  sc_clus1_E = cluster->energy();
		  if(gsf->pMode()>0) sc_clus1_E_ov_p = cluster->energy()/gsf->pMode();
		  sc_clus1_nxtal = (int)cluster->size();
		  if (reach_ECAL>0){
		    sc_clus1_deta = deta;
		    sc_clus1_dphi = dphi;
		  } 
		} else if (clusNum==i2) {   
		  sc_clus2_E    = cluster->energy();
		  if(gsf->pMode()>0) sc_clus2_E_ov_p = cluster->energy()/gsf->pMode();
		  if (reach_ECAL>0){
		    sc_clus2_deta = deta;
		    sc_clus2_dphi = dphi;
		  } 
		}
		clusNum++;
	      }
	    }
	  } catch(...) {
	    std::cout<<"caught an exception"<<std::endl;                            
	  }
	}
      }
    }  // clusters

    // Out-of-range
    if (eid_rho<0)   eid_rho=0;
    if (eid_rho>100) eid_rho=100;
    if (eid_sc_eta<-5) eid_sc_eta=-5;
    if (eid_sc_eta>5)  eid_sc_eta=5;
    if (eid_shape_full5x5_r9<0) eid_shape_full5x5_r9=0;
    if (eid_shape_full5x5_r9>2) eid_shape_full5x5_r9=2;
    if (eid_sc_etaWidth<0)    eid_sc_etaWidth=0;
    if (eid_sc_etaWidth>3.14) eid_sc_etaWidth=3.14;
    if (eid_sc_phiWidth<0)    eid_sc_phiWidth=0;
    if (eid_sc_phiWidth>3.14) eid_sc_phiWidth=3.14;
    if (eid_shape_full5x5_HoverE<0) eid_shape_full5x5_HoverE=0;
    if (eid_shape_full5x5_HoverE>50) eid_shape_full5x5_HoverE=50;
    if (eid_trk_nhits<-1) eid_trk_nhits=-1;
    if (eid_trk_nhits>50) eid_trk_nhits=50;
    if (eid_trk_chi2red<-1) eid_trk_chi2red=-1;
    if (eid_trk_chi2red>50) eid_trk_chi2red=50;
    if (eid_gsf_chi2red<-1)  eid_gsf_chi2red=-1;
    if (eid_gsf_chi2red>100) eid_gsf_chi2red=100;
    if (eid_brem_frac<0)  eid_brem_frac=-1;
    if (eid_brem_frac>1)  eid_brem_frac=1;
    if (eid_gsf_nhits<-1) eid_gsf_nhits=-1;
    if (eid_gsf_nhits>50) eid_gsf_nhits=50;
    if (eid_match_SC_EoverP<0)   eid_match_SC_EoverP=0;
    if (eid_match_SC_EoverP>100) eid_match_SC_EoverP=100;
    if (eid_match_eclu_EoverP<-0.001) eid_match_eclu_EoverP=-0.001;
    if (eid_match_eclu_EoverP>0.001)  eid_match_eclu_EoverP=0.001;
    eid_match_eclu_EoverP=eid_match_eclu_EoverP*1.E7;
    if (eid_match_SC_dEta<-10) eid_match_SC_dEta=-10;
    if (eid_match_SC_dEta>10)  eid_match_SC_dEta=10;
    if (eid_match_SC_dPhi<-3.14) eid_match_SC_dPhi=-3.14;
    if (eid_match_SC_dPhi>3.14)  eid_match_SC_dPhi=3.14;
    if (eid_match_seed_dEta<-10) eid_match_seed_dEta=-10;
    if (eid_match_seed_dEta>10)  eid_match_seed_dEta=10;    
    if (eid_sc_E<0)    eid_sc_E=0;
    if (eid_sc_E>1000) eid_sc_E=1000;
    if (eid_trk_p<-1)   eid_trk_p=-1;
    if (eid_trk_p>1000) eid_trk_p=1000;
    if (gsf_mode_p<0)    gsf_mode_p=0;
    if (gsf_mode_p>1000) gsf_mode_p=1000;
    if (core_shFracHits<0) core_shFracHits=0;
    if (core_shFracHits>1) core_shFracHits=1;
    if (gsf_bdtout1<-20) gsf_bdtout1=-20;
    if (gsf_bdtout1>20)  gsf_bdtout1=20;
    if (gsf_dr<0) gsf_dr=5;
    if (gsf_dr>5) gsf_dr=5;
    if (trk_dr<0) trk_dr=5;
    if (trk_dr>5) trk_dr=5;
    if (sc_Nclus<0)  sc_Nclus=0;
    if (sc_Nclus>20) sc_Nclus=20;    
    if (sc_clus1_nxtal<0)   sc_clus1_nxtal=0;
    if (sc_clus1_nxtal>100) sc_clus1_nxtal=100;
    if (sc_clus1_dphi<-3.14) sc_clus1_dphi=-5;
    if (sc_clus1_dphi>3.14)  sc_clus1_dphi=5;
    if (sc_clus2_dphi<-3.14) sc_clus2_dphi=-5;
    if (sc_clus2_dphi>3.14)  sc_clus2_dphi=5;
    if (sc_clus1_deta<-5) sc_clus1_deta=-5;
    if (sc_clus1_deta>5)  sc_clus1_deta=5;
    if (sc_clus2_deta<-5) sc_clus2_deta=-5;
    if (sc_clus2_deta>5)  sc_clus2_deta=5;
    if (sc_clus1_E<0)    sc_clus1_E=0;
    if (sc_clus1_E>1000) sc_clus1_E=1000;
    if (sc_clus2_E<0)    sc_clus2_E=0;
    if (sc_clus2_E>1000) sc_clus2_E=1000;
    if (sc_clus1_E_ov_p<0) sc_clus1_E_ov_p=-1;
    if (sc_clus2_E_ov_p<0) sc_clus2_E_ov_p=-1;

    return {
      eid_rho, 
      eid_sc_eta,           
      eid_shape_full5x5_r9,     
      eid_sc_etaWidth,      
      eid_sc_phiWidth,      
      eid_shape_full5x5_HoverE,
      eid_trk_nhits,        
      eid_trk_chi2red,      
      eid_gsf_chi2red,      
      eid_brem_frac,        
      eid_gsf_nhits,        
      eid_match_SC_EoverP,  
      eid_match_eclu_EoverP,
      eid_match_SC_dEta,    
      eid_match_SC_dPhi,    
      eid_match_seed_dEta,  
      eid_sc_E,
      eid_trk_p,             
      gsf_mode_p,               
      core_shFracHits,          
      gsf_bdtout1,  
      gsf_dr,                   
      trk_dr,                   
      sc_Nclus,
      sc_clus1_nxtal,
      sc_clus1_dphi,
      sc_clus2_dphi,
      sc_clus1_deta,
      sc_clus2_deta,
      sc_clus1_E,
      sc_clus2_E,
      sc_clus1_E_ov_p,
      sc_clus2_E_ov_p  
    };
  }

}  // namespace

class LowPtGsfElectronIDProducer final : public edm::global::EDProducer<> {
public:
  explicit LowPtGsfElectronIDProducer(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  double eval(const std::string& name, const reco::GsfElectronRef&, double rho, float unbiased) const;

  const edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectrons_;
  const edm::EDGetTokenT<double> rho_;
  const edm::EDGetTokenT< edm::ValueMap<float> > unbiased_;
  const std::vector<std::string> names_;
  const bool passThrough_;
  const double minPtThreshold_;
  const double maxPtThreshold_;
  std::vector<std::unique_ptr<const GBRForest> > models_;
  const std::vector<double> thresholds_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronIDProducer::LowPtGsfElectronIDProducer(const edm::ParameterSet& conf)
    : gsfElectrons_(consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("electrons"))),
      rho_(consumes<double>(conf.getParameter<edm::InputTag>("rho"))),
      unbiased_(consumes< edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("unbiased"))),
      names_(conf.getParameter<std::vector<std::string> >("ModelNames")),
      passThrough_(conf.getParameter<bool>("PassThrough")),
      minPtThreshold_(conf.getParameter<double>("MinPtThreshold")),
      maxPtThreshold_(conf.getParameter<double>("MaxPtThreshold")),
      thresholds_(conf.getParameter<std::vector<double> >("ModelThresholds")) {
  for (auto& weights : conf.getParameter<std::vector<std::string> >("ModelWeights")) {
    models_.push_back(createGBRForest(edm::FileInPath(weights)));
  }
  if (names_.size() != models_.size()) {
    throw cms::Exception("Incorrect configuration")
        << "'ModelNames' size (" << names_.size() << ") != 'ModelWeights' size (" << models_.size() << ").\n";
  }
  if (models_.size() != thresholds_.size()) {
    throw cms::Exception("Incorrect configuration")
        << "'ModelWeights' size (" << models_.size() << ") != 'ModelThresholds' size (" << thresholds_.size() << ").\n";
  }
  for (const auto& name : names_) {
    produces<edm::ValueMap<float> >(name);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {

  // Pileup
  edm::Handle<double> rho;
  event.getByToken(rho_, rho);
  if (!rho.isValid()) {
    edm::LogError("Problem with rho handle");
  }

  // Retrieve GsfElectrons from Event
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  event.getByToken(gsfElectrons_, gsfElectrons);
  if (!gsfElectrons.isValid()) {
    edm::LogError("Problem with gsfElectrons handle");
  }

  // ElectronSeed unbiased BDT          
  edm::Handle< edm::ValueMap<float> > unbiasedH;
  event.getByToken(unbiased_,unbiasedH);
  if (!unbiasedH.isValid()) { 
    edm::LogError("Problem with unbiased handle"); 
  }

  // Iterate through Electrons, evaluate BDT, and store result
  std::vector<std::vector<float> > output;
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    output.emplace_back(gsfElectrons->size(), -999.);
  }
  for (unsigned int iele = 0; iele < gsfElectrons->size(); iele++) {
    reco::GsfElectronRef ele(gsfElectrons, iele);

    if ( ele->core().isNull() ) { continue; }
    reco::GsfTrackRef gsf = ele->core()->gsfTrack();
    if ( gsf.isNull() ) { continue; }
    float unbiased = (*unbiasedH)[gsf];  

    //if ( !passThrough_ && ( ele->pt() < minPtThreshold_ ) ) { continue; }
    for (unsigned int iname = 0; iname < names_.size(); ++iname) {
      output[iname][iele] = eval(names_[iname], ele, *rho, unbiased);
    }
  }

  // Create and put ValueMap in Event
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    auto ptr = std::make_unique<edm::ValueMap<float> >(edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*ptr);
    filler.insert(gsfElectrons, output[iname].begin(), output[iname].end());
    filler.fill();
    reco::GsfElectronRef ele(gsfElectrons, 0);
    event.put(std::move(ptr), names_[iname]);
  }
}

double LowPtGsfElectronIDProducer::eval(const std::string& name, const reco::GsfElectronRef& ele, double rho, float unbiased) const {
  auto iter = std::find(names_.begin(), names_.end(), name);
  if (iter != names_.end()) {
    int index = std::distance(names_.begin(), iter);
    std::vector<float> inputs = getFeatures(ele, rho, unbiased);
    return models_.at(index)->GetResponse(inputs.data());
  } else {
    throw cms::Exception("Unknown model name") << "'Name given: '" << name << "'. Check against configuration file.\n";
  }
  return 0.;
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electrons", edm::InputTag("lowPtGsfElectrons"));
  desc.add<edm::InputTag>("unbiased",edm::InputTag("lowPtGsfElectronSeedValueMaps:unbiased"));  
  desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAllTmp"));
  desc.add<std::vector<std::string> >("ModelNames", {""});
  desc.add<std::vector<std::string> >(
      "ModelWeights",
      {"RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Autumn18_LowPtElectrons_mva_id.xml.gz"});
  desc.add<std::vector<double> >("ModelThresholds", {-10.});
  desc.add<bool>("PassThrough", false);
  desc.add<double>("MinPtThreshold", 0.5);
  desc.add<double>("MaxPtThreshold", 15.);
  descriptions.add("lowPtGsfElectronID", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronIDProducer);
