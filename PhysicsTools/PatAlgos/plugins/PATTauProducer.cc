//
// $Id: PATTauProducer.cc,v 1.3 2008/04/17 22:51:56 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATTauProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"

#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"

#include <vector>
#include <memory>


using namespace pat;


PATTauProducer::PATTauProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  tauSrc_         = iConfig.getParameter<edm::InputTag>( "tauSource" );
  addGenMatch_    = iConfig.getParameter<bool>         ( "addGenMatch" );
  addResolutions_ = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_      = iConfig.getParameter<bool>         ( "useNNResolutions" );
  genPartSrc_     = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );
  tauResoFile_    = iConfig.getParameter<std::string>  ( "tauResoFile" );

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new ObjectResolutionCalc(edm::FileInPath(tauResoFile_).fullPath(), useNNReso_);
  }

  // produces vector of taus
  produces<std::vector<Tau> >();
}


PATTauProducer::~PATTauProducer() {
  if (addResolutions_) delete theResoCalc_;
}


void PATTauProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
  std::auto_ptr<std::vector<Tau> > patTaus(new std::vector<Tau>()); 

  edm::Handle<edm::View<TauType> > anyTaus;
  try {
    iEvent.getByLabel(tauSrc_, anyTaus);
  } catch (const edm::Exception &e) {
    edm::LogWarning("DataSource") << "WARNING! No Tau collection found. This missing input will not block the job. Instead, an empty tau collection is being be produced.";
    iEvent.put(patTaus);
    return;
  }
   
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) iEvent.getByLabel(genPartSrc_, genMatch); 

  for (size_t idx = 0, ntaus = anyTaus->size(); idx < ntaus; ++idx) {
    edm::RefToBase<TauType> tausRef = anyTaus->refAt(idx);
    const TauType * originalTau = tausRef.get();

    Tau aTau(tausRef);

    if (typeid(originalTau) == typeid(const reco::PFTau *)) {
      const reco::PFTau *thePFTau = dynamic_cast<const reco::PFTau*>(originalTau);
      const reco::PFJet *pfJet    = dynamic_cast<const reco::PFJet*>(thePFTau->pfTauTagInfoRef()->pfjetRef().get());
      if(pfJet) {
        float ECALenergy=0.;
        float HCALenergy=0.;
        float leadEnergy=0.;
        std::list<const reco::PFBlockElement*> elements;
        reco::PFCandidateRefVector myPFCands=thePFTau->pfTauTagInfoRef()->PFCands();
        for(int i=0;i<(int)myPFCands.size();i++){
          const reco::PFCandidate::ElementsInBlocks& eib = myPFCands[i]->elementsInBlocks();
          for(reco::PFCandidate::ElementsInBlocks::const_iterator iPFBlockElement=eib.begin();
                                                            iPFBlockElement!=eib.end();++iPFBlockElement) {
            elements.push_back(&(iPFBlockElement->first->elements()[iPFBlockElement->second]));
          }
        }
        elements.sort();
        elements.unique();
        const reco::PFCandidate::ElementsInBlocks& eib = thePFTau->leadPFChargedHadrCand()->elementsInBlocks();
        for(reco::PFCandidate::ElementsInBlocks::const_iterator iPFBlockElement=eib.begin();
                                                          iPFBlockElement!=eib.end();++iPFBlockElement) {
          if((iPFBlockElement->first->elements()[iPFBlockElement->second].type()==reco::PFBlockElement::HCAL)||
             (iPFBlockElement->first->elements()[iPFBlockElement->second].type()==reco::PFBlockElement::ECAL)  )
            leadEnergy += iPFBlockElement->first->elements()[iPFBlockElement->second].clusterRef()->energy();
        }
        for(std::list<const reco::PFBlockElement*>::const_iterator ielements = elements.begin();ielements!=elements.end();++ielements) {
          if((*ielements)->type()==reco::PFBlockElement::HCAL)
            HCALenergy += (*ielements)->clusterRef()->energy();
          else if((*ielements)->type()==reco::PFBlockElement::ECAL)
            ECALenergy += (*ielements)->clusterRef()->energy();
        }
	aTau.setEmEnergyFraction(ECALenergy/(ECALenergy+HCALenergy));
	aTau.setEOverP((HCALenergy+ECALenergy)/thePFTau->leadPFChargedHadrCand()->p());
        aTau.setLeadEOverP(leadEnergy/thePFTau->leadPFChargedHadrCand()->p());
        aTau.setHhotOverP(thePFTau->maximumHCALPFClusterEt()/thePFTau->leadPFChargedHadrCand()->p());
        aTau.setHtotOverP(HCALenergy/thePFTau->leadPFChargedHadrCand()->p());
      }
    } else if (typeid(originalTau) == typeid(const reco::CaloTau *)) {
      const reco::CaloTau *theCaloTau = dynamic_cast<const reco::CaloTau*>(originalTau);
      const reco::CaloJet *caloJet    = dynamic_cast<const reco::CaloJet*>(theCaloTau->caloTauTagInfoRef()->calojetRef().get());
      if(caloJet) {
        aTau.setEmEnergyFraction(caloJet->emEnergyFraction());
        aTau.setEOverP(caloJet->energy()/theCaloTau->leadTrack()->p());
	aTau.setLeadEOverP(caloJet->energy()/theCaloTau->leadTrack()->p()/theCaloTau->numberOfTracks()); //just an approx of what can be done for PF
        aTau.setHhotOverP(theCaloTau->maximumHCALhitEt()/theCaloTau->leadTrack()->p());
        aTau.setHtotOverP(caloJet->energy()*caloJet->energyFractionHadronic()/theCaloTau->leadTrack()->p());
      }
    }

    // add MC match if demanded
    if (addGenMatch_) {
      reco::GenParticleRef genTau = (*genMatch)[tausRef];
      if (genTau.isNonnull() && genTau.isAvailable() ) {
        aTau.setGenLepton(*genTau);
      } else {
        aTau.setGenLepton(reco::Particle(0, reco::Particle::LorentzVector(0,0,0,0))); // TQAF way of setting "null"
      }
    }

    // add resolution info if demanded
    if (addResolutions_) {
      (*theResoCalc_)(aTau);
    }

    patTaus->push_back(aTau);
  }

  // sort taus in pT
  std::sort(patTaus->begin(), patTaus->end(), pTTauComparator_);

  // put genEvt object in Event
  iEvent.put(patTaus);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauProducer);


