#ifndef DIOBJECTUPDATEFACTORY_H_
#define DIOBJECTUPDATEFACTORY_H_

#include "CMGTools/H2TauTau/interface/DiTauObjectFactory.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace cmg{

  typedef pat::CompositeCandidate DiTauObject;

  template< typename T, typename U>
  class DiObjectUpdateFactory : public edm::EDProducer  {

  typedef std::vector<DiTauObject> collection;

  public:
    
    DiObjectUpdateFactory(const edm::ParameterSet& ps):
      diObjectLabel_     (ps.getParameter<edm::InputTag>("diObjectCollection")),
      genParticleLabel_  (ps.getParameter<edm::InputTag>("genCollection")),
      nSigma_            (ps.getParameter<double>("nSigma")),
      uncertainty_       (ps.getParameter<double>("uncertainty")),
      shift1ProngNoPi0_  (ps.getParameter<double>("shift1ProngNoPi0")),
      shift1Prong1Pi0_   (ps.getParameter<double>("shift1Prong1Pi0")),
      ptDependence1Pi0_  (ps.getParameter<double>("ptDependence1Pi0")),
      shift3Prong_       (ps.getParameter<double>("shift3Prong")),
      ptDependence3Prong_(ps.getParameter<double>("ptDependence3Prong")),
      shiftMet_          (ps.getParameter<bool>("shiftMet")),
      shiftTaus_         (ps.getParameter<bool>("shiftTaus"))
      {
        produces<collection>();
      }

    void produce(edm::Event&, const edm::EventSetup&);
    
  private:

    const edm::InputTag diObjectLabel_;
    const edm::InputTag genParticleLabel_;
    double nSigma_;
    double uncertainty_; 
    double shift1ProngNoPi0_; 
    double shift1Prong1Pi0_; 
    double ptDependence1Pi0_; 
    double shift3Prong_; 
    double ptDependence3Prong_; 
    bool   shiftMet_ ;
    bool   shiftTaus_ ;
  };
  
} // namespace cmg


template< typename T, typename U >
void cmg::DiObjectUpdateFactory<T, U>::produce(edm::Event& iEvent, const edm::EventSetup&){
    
  edm::Handle<collection> diObjects;
  iEvent.getByLabel(diObjectLabel_,diObjects);

  edm::Handle< std::vector<reco::GenParticle> > genparticles;
  iEvent.getByLabel(genParticleLabel_, genparticles);
   
  std::auto_ptr<collection> result(new collection);
  
  unsigned index = 0;
  for(typename collection::const_iterator it = diObjects->begin(); it != diObjects->end(); ++it, ++index ){
    const DiTauObject& diObject = *it;
    // assert( index < metCands->size() );
    T leg1(*dynamic_cast<const T*>(diObject.daughter(0)));
    U leg2(*dynamic_cast<const U*>(diObject.daughter(1)));
    reco::MET met(*dynamic_cast<const reco::MET*>(diObject.daughter(2)));

    float shift1 = 0.;
    float shift2 = 0.;
    if(typeid(T)==typeid(pat::Tau))
    {
     	shift1 = (nSigma_ * uncertainty_);
      const pat::Tau& tau1 = dynamic_cast<const pat::Tau&>(*diObject.daughter(0));
     	if((tau1.decayMode()==0)&&(shift1ProngNoPi0_!=0))
     	    shift1+=shift1ProngNoPi0_;
        //Also allow decay mode 2 according to synchronisation twiki
     	if((tau1.decayMode()==1 || tau1.decayMode()==2)&&(shift1Prong1Pi0_!=0))
     	    shift1+=shift1Prong1Pi0_+ptDependence1Pi0_*TMath::Min(TMath::Max(diObject.daughter(0)->pt()-45.,0.),10.);
     	if((tau1.decayMode()==10)&&(shift3Prong_!=0))
     	    shift1+=shift3Prong_+ptDependence3Prong_*TMath::Min(TMath::Max(diObject.daughter(0)->pt()-32.,0.),18.);
    }

    if(typeid(U)==typeid(pat::Tau))
    {
     	shift2 = (nSigma_ * uncertainty_);
      const pat::Tau& tau2 = dynamic_cast<const pat::Tau&>(*diObject.daughter(1));
     	if((tau2.decayMode()==0)&&(shift1ProngNoPi0_!=0))
     	    shift2+=shift1ProngNoPi0_;
        //Also allow decay mode 2 according to synchronisation twiki
     	if((tau2.decayMode()==1 || tau2.decayMode()==2)&&(shift1Prong1Pi0_!=0))
     	    shift2+=shift1Prong1Pi0_+ptDependence1Pi0_*TMath::Min(TMath::Max(diObject.daughter(1)->pt()-45.,0.),10.);
     	if((tau2.decayMode()==10)&&(shift3Prong_!=0))
     	    shift2+=shift3Prong_+ptDependence3Prong_*TMath::Min(TMath::Max(diObject.daughter(1)->pt()-32.,0.),18.);
    }

    // the tauES shift must be applied to *real* taus only
    bool l1genMatched = false ;
    bool l2genMatched = false ;
    
    for ( size_t i=0; i< genparticles->size(); ++i) 
    {
      const reco::GenParticle &p = (*genparticles)[i];
      int id       = p.pdgId()           ;
      int status   = p.status()          ;
      int motherId = 0                   ; 
      if ( p.numberOfMothers()>0 ) {
        //std::cout << __LINE__ << "]\tnum of mothers " << p.numberOfMothers() << "\tmy mom " << p.mother()->pdgId() << std::endl ;
        motherId = p.mother(0)->pdgId() ;
      }
      // PDG Id: e 11, mu 13, tau 15, Z 23, h 25, H 35, A 35  
      if ( status == 3 && abs(id) == 15 && (motherId == 23 || motherId == 25 || motherId == 35 || motherId == 36 )){
        // match leg 1
        if(typeid(T)==typeid(pat::Tau)){
          const pat::Tau& tau1 = dynamic_cast<const pat::Tau&>(*diObject.daughter(0));
          if (deltaR(tau1.eta(),tau1.phi(),p.eta(),p.phi())<0.3) {
            l1genMatched = true ;
            //std::cout << __LINE__ << "]\tleg1 matched to a tau" << std::endl ;
          }
        }
        // match leg 2
        if(typeid(U)==typeid(pat::Tau)){
          const pat::Tau& tau2 = dynamic_cast<const pat::Tau&>(*diObject.daughter(1));
          if (deltaR(tau2.eta(),tau2.phi(),p.eta(),p.phi())<0.3) {
            l2genMatched = true ;
            //std::cout << __LINE__ << "]\tleg2 matched to a tau" << std::endl ;
          }
        }
      }
    }

    reco::Candidate::LorentzVector leg1Vec = diObject.daughter(0)->p4();
    reco::Candidate::LorentzVector leg2Vec = diObject.daughter(1)->p4();
    reco::Candidate::LorentzVector metVec = met.p4();

    float dpx = 0.;
    float dpy = 0.;

    // if genMatched compute the transverse momentum variation 
    dpx = l1genMatched * leg1Vec.px() * shift1 + l2genMatched * leg2Vec.px() * shift2;
    dpy = l1genMatched * leg1Vec.py() * shift1 + l2genMatched * leg2Vec.py() * shift2;

    // if genMatched apply the shift 
    if (l1genMatched) leg1Vec *= (1. + shift1);
    if (l2genMatched) leg2Vec *= (1. + shift2);

    // apply the tranverse momentum correction to the MET 
    math::XYZTLorentzVector deltaTauP4(dpx,dpy,0,0);
    math::XYZTLorentzVector scaledmetP4 = metVec - deltaTauP4;

    TLorentzVector metVecNew;
    metVecNew.SetPtEtaPhiM(scaledmetP4.Pt(),scaledmetP4.Eta(),scaledmetP4.Phi(),0.);
        
    if (shiftTaus_ ){ leg1.setP4(leg1Vec); }
    if (shiftTaus_ ){ leg2.setP4(leg2Vec); }
    if (shiftMet_  ){ met.setP4(reco::Candidate::LorentzVector(metVecNew.Px(),metVecNew.Py(),metVecNew.Pz(),metVecNew.E())); }

    result->push_back(diObject);

    DiTauObjectFactory<T, U>::set( std::make_pair(leg1, leg2), met, result->back() );
  }
  
  iEvent.put(result);
}


#endif /*DIOBJECTUPDATEFACTORY_H_*/
