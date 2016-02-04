// -*- C++ -*-
//
// Package:    TruthTauDecayModeProducer
// Class:      TruthTauDecayModeProducer
// 
/**\class TruthTauDecayModeProducer TruthTauDecayModeProducer.cc 

Description: Produces reco::PFTauDecayModes corresponding to MonteCarlo objects
             For signal, it uses decayed MC taus
             For background, it uses GenJets

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
//         Created:  Thu Sep 1 06:19:05 PST 2008
// $Id: TruthTauDecayModeProducer.cc,v 1.6 2010/10/19 20:22:27 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "RecoTauTag/TauTagTools/interface/GeneratorTau.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TruthTauDecayModeProducer : public edm::EDProducer {
   public:
      struct tauObjectsHolder {
         std::vector<const reco::Candidate*> chargedObjects;
         std::vector<const reco::Candidate*> neutralObjects;
      };
      explicit TruthTauDecayModeProducer(const edm::ParameterSet&);
      ~TruthTauDecayModeProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //for signal, the module takes input from a PdgIdAndStatusCandViewSelector 
      //for background, the module takes input from a collection of GenJets
      bool              iAmSignal_; 
      edm::InputTag     inputTag_;
      double            leadTrackPtCut_;
      double            leadTrackEtaCut_;
      double            totalPtCut_;
      double            totalEtaCut_;
      AddFourMomenta    addP4;
};

TruthTauDecayModeProducer::TruthTauDecayModeProducer(const edm::ParameterSet& iConfig)
{
   edm::LogInfo("TruthTauDecayModeProducer") << "Initializing ctor of TruthTauDecayModeProducer";
   iAmSignal_           = iConfig.getParameter<bool>("iAmSignal");
   inputTag_            = iConfig.getParameter<edm::InputTag>("inputTag");
   leadTrackPtCut_      = iConfig.getParameter<double>("leadTrackPtCut");
   leadTrackEtaCut_     = iConfig.getParameter<double>("leadTrackEtaCut");
   totalPtCut_          = iConfig.getParameter<double>("totalPtCut");
   totalEtaCut_         = iConfig.getParameter<double>("totalEtaCut");
   //register your products
   edm::LogInfo("TruthTauDecayModeProducer") << "Registering products";
   produces<std::vector<reco::PFTauDecayMode> >();
   edm::LogInfo("TruthTauDecayModeProducer") << "TruthTauDecayModeProducer initialized";
}


TruthTauDecayModeProducer::~TruthTauDecayModeProducer()
{
}

void
TruthTauDecayModeProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   std::vector<tauObjectsHolder> tausToAdd_;

   /* **********************************************
    * **********  True Tau Case          ***********
    * ********************************************** */
   if (iAmSignal_)
   {
      edm::Handle<RefToBaseVector<reco::Candidate> > decayedMCTaus;
      iEvent.getByLabel(inputTag_, decayedMCTaus);
      for(edm::RefToBaseVector<reco::Candidate>::const_iterator iterGen = decayedMCTaus->begin();
            iterGen != decayedMCTaus->end();
            ++iterGen)
      {
         //copy into custom format (this casting is bullshit)
         GeneratorTau tempTau = static_cast<const GenParticle&>(*(*iterGen));
         //LogInfo("MCTauCandidateProducer") << "Generator tau produced, initializing.. ";
         tempTau.init();
         //LogInfo("MCTauCandidateProducer") << "GenTau initialization done";
         if (tempTau.isFinalStateTau())
         {
            // Build a Tau Candidate from the information contained in the parsing class
            tauObjectsHolder tempTauHolder;
            tempTauHolder.chargedObjects = tempTau.getGenChargedPions();
            tempTauHolder.neutralObjects = tempTau.getGenNeutralPions();
            tausToAdd_.push_back(tempTauHolder);
         }
      }
   } else
   {
   /* **********************************************
    * **********  QCD Case               ***********
    * ********************************************** */
      edm::Handle<GenJetCollection> genJets;
      iEvent.getByLabel(inputTag_, genJets);
      for(GenJetCollection::const_iterator aGenJet = genJets->begin(); aGenJet != genJets->end(); ++aGenJet)
      {
         // get all constituents
         std::vector<const GenParticle*> theJetConstituents = aGenJet->getGenConstituents();

         tauObjectsHolder tempTauHolder;
         // filter the constituents
         for( std::vector<const GenParticle*>::const_iterator aCandidate = theJetConstituents.begin();
               aCandidate != theJetConstituents.end();
               ++aCandidate)
         {
            int pdgId = std::abs((*aCandidate)->pdgId());
            const Candidate* theCandidate = static_cast<const Candidate*>(*aCandidate);
            //filter nus
            if (pdgId == 16 || pdgId == 12 || pdgId == 14)
            {
               //do nothing
            } else 
            {
               // call everything charged a pion
               // call everything neutral a neutral pion
               if (theCandidate->charge() != 0)
                  tempTauHolder.chargedObjects.push_back(theCandidate);
               else
                  tempTauHolder.neutralObjects.push_back(theCandidate);
            }
         }
         tausToAdd_.push_back(tempTauHolder);
      }
   }

   //output collection
   std::auto_ptr<vector<PFTauDecayMode> > pOut( new std::vector<PFTauDecayMode> );
   for(std::vector<tauObjectsHolder>::const_iterator iTempTau  = tausToAdd_.begin();
                                                iTempTau != tausToAdd_.end();
                                              ++iTempTau)
   {
      double leadTrackPt  = 0.;
      double leadTrackEta = 0.;
      VertexCompositeCandidate chargedObjectsToAdd;
      const std::vector<const Candidate*>* chargedObjects = &(iTempTau->chargedObjects);
      for(std::vector<const Candidate*>::const_iterator iCharged  = chargedObjects->begin();
                                                   iCharged != chargedObjects->end();
                                                 ++iCharged)
      {
         chargedObjectsToAdd.addDaughter(**iCharged);
         double trackPt = (*iCharged)->pt();
         if (trackPt > leadTrackPt)
         {
            leadTrackPt  = trackPt;
            leadTrackEta = (*iCharged)->eta();
         }
      }
      //update the composite four vector
      addP4.set(chargedObjectsToAdd);

      CompositeCandidate neutralPionsToAdd;
      const std::vector<const Candidate*>* neutralObjects = &(iTempTau->neutralObjects);
      for(std::vector<const Candidate*>::const_iterator iNeutral  = neutralObjects->begin();
                                                   iNeutral != neutralObjects->end();
                                                 ++iNeutral)
      {
         neutralPionsToAdd.addDaughter(**iNeutral);
      }
      addP4.set(neutralPionsToAdd);

      Particle::LorentzVector myFourVector = chargedObjectsToAdd.p4();
      myFourVector += neutralPionsToAdd.p4();

      if(leadTrackPt > leadTrackPtCut_ && std::abs(leadTrackEta) < leadTrackEtaCut_ && myFourVector.pt() > totalPtCut_ && std::abs(myFourVector.eta()) < totalEtaCut_)
      {
         //TODO: add vertex fitting
         CompositeCandidate theOutliers;
         PFTauRef             noPFTau;     //empty REF to PFTau
         PFTauDecayMode decayModeToAdd(chargedObjectsToAdd, neutralPionsToAdd, theOutliers);
         decayModeToAdd.setPFTauRef(noPFTau);
         pOut->push_back(decayModeToAdd);
      }
   }
   iEvent.put(pOut);
}

void 
TruthTauDecayModeProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TruthTauDecayModeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TruthTauDecayModeProducer);
