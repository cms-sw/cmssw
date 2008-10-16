//
// $Id: PATMHTProducer.cc,v 1.2 2008/10/16 16:53:41 fblekman Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMHTProducer.h"
#include "DataFormats/Candidate/interface/Particle.h"

pat::PATMHTProducer::PATMHTProducer(const edm::ParameterSet & iConfig){

  // Initialize the configurables
  jetLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("jetTag");
  eleLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("electronTag");
  muoLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("muonTag");
  tauLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("tauTag");
  phoLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("photonTag");
  
  uncertaintyScaleFactor_ = iConfig.getParameter<double>( "uncertaintyScaleFactor") ;

  produces<pat::MHTCollection>();

}


pat::PATMHTProducer::~PATMHTProducer() {
}

void pat::PATMHTProducer::beginJob(const edm::EventSetup& iSetup) {
}
void pat::PATMHTProducer::beginRun(const edm::EventSetup& iSetup) {
}

void pat::PATMHTProducer::endJob() {
}


void pat::PATMHTProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // make sure the SigInputObj container is empty
  while(physobjvector_.size()>0){
    physobjvector_.erase(physobjvector_.begin(),physobjvector_.end());

  }
  // Get the jet object

  edm::Handle<edm::View<pat::Jet> > jetHandle;
  iEvent.getByLabel(jetLabel_,jetHandle);
  edm::View<pat::Jet> jets = *jetHandle;

  // Fill Input Vector with Jets 
  std::string objectname="";
  for(edm::View<pat::Jet>::const_iterator jet_iter = jets.begin(); jet_iter!=jets.end(); ++jet_iter){
    double jet_et = jet_iter->et();
    double jet_phi = jet_iter->phi();
    double sigma_et = 0.;// no longer valid: jet_iter->resolutionEt();
    double sigma_phi =  0.;// no longer valid: jet_iter->resolutionPhi();
    objectname="jet";
    if(sigma_et<=0 || sigma_phi<=0)
      edm::LogWarning("PATMHTProducer") << 
	" uncertainties for "  << objectname <<
	" are (et, phi): " << sigma_et << "," << sigma_phi << " (et,phi): " << jet_et << "," << jet_phi;
    // try to read out the jet resolution from the root file at PatUtils
    //-- Store jet for Significance Calculation --//
    
    if (uncertaintyScaleFactor_ != 1.0){
      sigma_et  = sigma_et  * uncertaintyScaleFactor_;
      sigma_phi = sigma_phi * uncertaintyScaleFactor_;
      // edm::LogWarning("PATMHTProducer") << " using uncertainty scale factor: " << uncertaintyScaleFactor_ <<
      //" , uncertainties for " << objectname <<" changed to (et, phi): " << sigma_et << "," << sigma_phi; 
    }

    metsig::SigInputObj tmp_jet(objectname,jet_et,jet_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_jet);
     
  }

  edm::Handle<edm::View<pat::Electron> > electronHandle;
  iEvent.getByLabel(eleLabel_,electronHandle);
  edm::View<pat::Electron> electrons = *electronHandle;

  // Fill Input Vector with Electrons 
  for(edm::View<pat::Electron>::const_iterator electron_iter = electrons.begin(); electron_iter!=electrons.end(); ++electron_iter){
    double electron_et = electron_iter->et();
    double electron_phi = electron_iter->phi();
    double sigma_et = 0.;// no longer valid:  electron_iter->resolutionEt();
    double sigma_phi = 0.;// no longer valid:  electron_iter->resolutionPhi();
    objectname="electron";
    if(sigma_et<=0 || sigma_phi<=0)
      edm::LogWarning("PATMHTProducer") <<
	" uncertainties for "  << objectname <<
	" are (et, phi): " << sigma_et << "," << sigma_phi << 
	" (et,phi): " << electron_et << "," << electron_phi;
    // try to read out the electron resolution from the root file at PatUtils
    //-- Store electron for Significance Calculation --//
    
    if (uncertaintyScaleFactor_ != 1.0){
      sigma_et  = sigma_et  * uncertaintyScaleFactor_;
      sigma_phi = sigma_phi * uncertaintyScaleFactor_;
      // edm::LogWarning("PATMHTProducer") << " using uncertainty scale factor: " << uncertaintyScaleFactor_ <<
      //" , uncertainties for " << objectname <<" changed to (et, phi): " << sigma_et << "," << sigma_phi; 
    }


    metsig::SigInputObj tmp_electron(objectname,electron_et,electron_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_electron);
     
  }

  edm::Handle<edm::View<pat::Muon> > muonHandle;
  iEvent.getByLabel(muoLabel_,muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;

  // Fill Input Vector with Muons 
  for(edm::View<pat::Muon>::const_iterator muon_iter = muons.begin(); muon_iter!=muons.end(); ++muon_iter){
    double muon_pt = muon_iter->pt();
    double muon_phi = muon_iter->phi();
    double sigma_et = 0.;// no longer valid:  muon_iter->resolutionEt();
    double sigma_phi = 0.;// no longer valid:  muon_iter->resolutionPhi();
    objectname="muon";
    if(sigma_et<=0 || sigma_phi<=0)
      edm::LogWarning("PATMHTProducer") << 
	" uncertainties for "  << objectname << 
	" are (et, phi): " << sigma_et << "," <<
	sigma_phi << " (pt,phi): " << muon_pt << "," << muon_phi;
    // try to read out the muon resolution from the root file at PatUtils
    //-- Store muon for Significance Calculation --//

    if (uncertaintyScaleFactor_ != 1.0){
      sigma_et  = sigma_et  * uncertaintyScaleFactor_;
      sigma_phi = sigma_phi * uncertaintyScaleFactor_;
      //edm::LogWarning("PATMHTProducer") << " using uncertainty scale factor: " << uncertaintyScaleFactor_ <<
      //" , uncertainties for " << objectname <<" changed to (et, phi): " << sigma_et << "," << sigma_phi; 
    }

    metsig::SigInputObj tmp_muon(objectname,muon_pt,muon_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_muon);
     
  }
  
  /* We'll deal with photons and taus later for sure :)


  edm::Handle<edm::View<pat::Photon> > photonHandle;
  iEvent.getByLabel(phoLabel_,photonHandle);
  edm::View<pat::Photon> photons = *photonHandle;

  // Fill Input Vector with Photons 
  for(edm::View<pat::Photon>::const_iterator photon_iter = photons.begin(); photon_iter!=photons.end(); ++photon_iter){
    double photon_et = photon_iter->et();
    double photon_phi = photon_iter->phi();
    double sigma_et = 0.;// no longer valid:  photon_iter->resolutionEt();
    double sigma_phi = 0.;// no longer valid:  photon_iter->resolutionPhi();
    objectname="photon";
    if(sigma_et<=0 || sigma_phi<=0)
      edm::LogWarning("PATMHTProducer") << " uncertainties for "  << objectname << " are (et, phi): " << sigma_et << "," << sigma_phi << " (et,phi): " << photon_et << "," << photon_phi;
    // try to read out the photon resolution from the root file at PatUtils
    //-- Store photon for Significance Calculation --//
    metsig::SigInputObj tmp_photon(objectname,photon_et,photon_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_photon);
     
  }

  edm::Handle<edm::View<pat::Tau> > tauHandle;
  iEvent.getByLabel(tauLabel_,tauHandle);
  edm::View<pat::Tau> taus = *tauHandle;

  // Fill Input Vector with Taus 
  for(edm::View<pat::Tau>::const_iterator tau_iter = taus.begin(); tau_iter!=taus.end(); ++tau_iter){
    double tau_pt = tau_iter->pt();
    double tau_phi = tau_iter->phi();
    double sigma_et =  0.;// no longer valid: tau_iter->resolutionEt();
    double sigma_phi =  0.;// no longer valid: tau_iter->resolutionPhi();
    objectname="tau";
    if(sigma_et<=0 || sigma_phi<=0)
      edm::LogWarning("PATMHTProducer") << " uncertainties for "  << objectname << " are (et, phi): " << sigma_et << "," << sigma_phi << " (pt,phi): " << tau_pt << "," << tau_phi;
    // try to read out the tau resolution from the root file at PatUtils
    //-- Store tau for Significance Calculation --//
    metsig::SigInputObj tmp_tau(objectname,tau_pt,tau_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_tau);
     
  }
  
  */


  double met_x=0;
  double met_y=0;
  double met_et=0;
  double met_phi=0;
  double met_set=0;
  
  // calculate the significance

  double significance = ASignificance(physobjvector_, met_et, met_phi, met_set);
  met_x=met_et*cos(met_phi);
  met_y=met_et*sin(met_phi);
  edm::LogInfo("PATMHTProducer")    << " met x,y: " << met_x << "," << met_y << " met_set: " << met_set << " met_et/sqrt(met_set): " << met_et/sqrt(met_set) << " met_phi: " << met_phi << " met_et: " << met_et << " met_et/sqrt(x,y): " << met_et/sqrt(met_x*met_x+met_y*met_y) << " met_sign: " << significance << std::endl;
  // and fill the output into the event..
  std::auto_ptr<pat::MHTCollection>  themetsigcoll (new pat::MHTCollection);
  pat::MHT themetsigobj(Particle::LorentzVector(met_x,met_y,0,met_et),met_set,significance);
  themetsigcoll->push_back(themetsigobj);

  iEvent.put( themetsigcoll);

}  


using namespace pat; 
DEFINE_FWK_MODULE(PATMHTProducer);

