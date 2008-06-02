//
// $Id: PATTauProducer.cc,v 1.1.2.5 2008/05/14 13:28:31 lowette Exp $
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


PATTauProducer::PATTauProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) 
{

  // initialize the configurables
  tauSrc_               = iConfig.getParameter<edm::InputTag>( "tauSource" );
  embedIsolationTracks_ = iConfig.getParameter<bool>         ( "embedIsolationTracks" );
  embedLeadTrack_       = iConfig.getParameter<bool>         ( "embedLeadTrack" );
  embedSignalTracks_    = iConfig.getParameter<bool>         ( "embedSignalTracks" );
  addGenMatch_    = iConfig.getParameter<bool>         ( "addGenMatch" );
  // Trigger matching configurables
  addTrigMatch_   = iConfig.getParameter<bool>         ( "addTrigMatch" );
  trigPrimSrc_    = iConfig.getParameter<std::vector<edm::InputTag> >( "trigPrimMatch" );
  addResolutions_ = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_      = iConfig.getParameter<bool>         ( "useNNResolutions" );
  genMatchSrc_    = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );
  tauResoFile_    = iConfig.getParameter<std::string>  ( "tauResoFile" );

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new ObjectResolutionCalc(edm::FileInPath(tauResoFile_).fullPath(), useNNReso_);
  }

  if (iConfig.exists("isoDeposits")) {
     edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
     if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
     if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
     if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));
     if (depconf.exists("user")) {
        std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
        std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
        int key = UserBaseIso;
        for ( ; it != ed; ++it, ++key) {
            isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
        }
     }
  }


  // produces vector of taus
  produces<std::vector<Tau> >();
}


PATTauProducer::~PATTauProducer() {
  if (addResolutions_) delete theResoCalc_;
}


void PATTauProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
  std::auto_ptr<std::vector<Tau> > patTaus(new std::vector<Tau>()); 

  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  edm::Handle<View<TauType> > anyTaus;
  try {
    iEvent.getByLabel(tauSrc_, anyTaus);
  } catch (const edm::Exception &e) {
    edm::LogWarning("DataSource") << "WARNING! No Tau collection found. This missing input will not block the job. Instead, an empty tau collection is being be produced.";
    iEvent.put(patTaus);
    return;
  }
   
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) iEvent.getByLabel(genMatchSrc_, genMatch); 

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  for (size_t idx = 0, ntaus = anyTaus->size(); idx < ntaus; ++idx) {
    edm::RefToBase<TauType> tausRef = anyTaus->refAt(idx);
    const TauType * originalTau = tausRef.get();

    Tau aTau(tausRef);
    if (embedIsolationTracks_) aTau.embedIsolationTracks();
    if (embedLeadTrack_) aTau.embedLeadTrack();
    if (embedSignalTracks_) aTau.embedSignalTracks();

    if (typeid(originalTau) == typeid(const reco::PFTau *)) {
      const reco::PFTau *thePFTau = dynamic_cast<const reco::PFTau*>(originalTau);
      const reco::PFJet *pfJet    = dynamic_cast<const reco::PFJet*>(thePFTau->pfTauTagInfoRef()->pfjetRef().get());
      if(pfJet) {
        float ECALenergy=0.;
        float HCALenergy=0.;
        float leadEnergy=0.;
	std::list<const PFBlockElement*> elements;
        PFCandidateRefVector myPFCands=thePFTau->pfTauTagInfoRef()->PFCands();
        for(int i=0;i<(int)myPFCands.size();i++){
          if(myPFCands[i]->blockRef()->elements().size()!=0){
            for(OwnVector<PFBlockElement>::const_iterator iPFBlockElement=myPFCands[i]->blockRef()->elements().begin();
                iPFBlockElement!=myPFCands[i]->blockRef()->elements().end();iPFBlockElement++){
              elements.push_back(&(*iPFBlockElement));
            }
          }
        }
        if(thePFTau->leadPFChargedHadrCand()->blockRef()->elements().size()!=0){
          for(OwnVector<PFBlockElement>::const_iterator iPFBlockElement=thePFTau->leadPFChargedHadrCand()->blockRef()->elements().begin();
              iPFBlockElement!=thePFTau->leadPFChargedHadrCand()->blockRef()->elements().end();iPFBlockElement++){
            if((iPFBlockElement->type()==PFBlockElement::HCAL)||(iPFBlockElement->type()==PFBlockElement::ECAL))
              leadEnergy += iPFBlockElement->clusterRef()->energy();
          }
        }
        elements.sort();
        elements.unique();
        for(std::list<const PFBlockElement*>::const_iterator ielements = elements.begin();ielements!=elements.end();++ielements) {
          if((*ielements)->type()==PFBlockElement::HCAL)
            HCALenergy += (*ielements)->clusterRef()->energy();
          else if((*ielements)->type()==PFBlockElement::ECAL)
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

    // store the match to the generated final state taus
    if (addGenMatch_) {
      reco::GenParticleRef genTau = (*genMatch)[tausRef];
      if (genTau.isNonnull() && genTau.isAvailable() ) {
        aTau.setGenLepton(*genTau);
      } // leave empty if no match found
    }

    // matches to fired trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigPrimSrc_.size(); ++i ) {
        edm::Handle<edm::Association<TriggerPrimitiveCollection> > trigMatch;
        iEvent.getByLabel(trigPrimSrc_[i], trigMatch);
        TriggerPrimitiveRef trigPrim = (*trigMatch)[tausRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          aTau.addTriggerMatch(*trigPrim);
        }
      }
    }

    // Isolation
    if (isolator_.enabled()) {
        isolator_.fill(*anyTaus, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
            aTau.setIsolation(it->first, it->second);
        }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        aTau.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[tausRef]);
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

  if (isolator_.enabled()) isolator_.endEvent();
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauProducer);


