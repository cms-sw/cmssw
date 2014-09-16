//
// Translation of BTag MCJetFlavour tool to identify real flavour of a jet
// work with CaloJet objects
// Store Infos by Values in JetFlavour.h
// Author: Attilio
// Date: 05.10.2007
//
//
// \class JetFlavourIdentifier
//
// \brief Interface to pull out the proper flavour identifier from a
//        jet->parton matching collection.
//
// In detail, the definitions are as follows:
//
// Definitions:
// The default behavior is that the "definition" is NULL_DEF,
// so the software will fall back to either "physics" or "algorithmic" definition
// as per the "physDefinition" switch.
//
// If the user specifies "definition", then that definition is taken. However,
// if the requested definition is not defined, the software reverts back to
// either "physics" or "algorithmic" based on the "physDefinition" switch.
// For example, if the user specifies "heaviest" as a flavor ID, and there
// are no bottom, charm, or top quarks in the event that match to the jet,
// then the software will fall back to the "physics" or "algorithmic" definition.
//
// Modifications:
//
//     09.03.2008: Sal Rappoccio.
//                 Added capability for all methods of identification, not just
//                 "physics" or "algorithmic". If the requested method does not exist
//                 (i.e. is unphysical), the "physics" or "algorithmic" definition
//                 is defaulted to.

//=======================================================================

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/MatchedPartons.h"
#include "SimDataFormats/JetMatching/interface/JetMatchedPartons.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <Math/VectorUtil.h>
#include <TMath.h>

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;

namespace reco { namespace modules {

//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
class JetFlavourIdentifier : public edm::EDProducer
{
  public:
  enum DEFINITION_T { PHYSICS=0, ALGO, NEAREST_STATUS2, NEAREST_STATUS3, HEAVIEST,
		      N_DEFINITIONS,
		      NULL_DEF};

    JetFlavourIdentifier( const edm::ParameterSet & );
    ~JetFlavourIdentifier();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup& ) override;

    JetFlavour::Leptons findLeptons(const GenParticleRef &);
    std::vector<const reco::Candidate*> findCandidates(const reco::Candidate*, int);
    void fillLeptons(const std::vector<const reco::Candidate*> &, JetFlavour::Leptons &, int, int);
    static int heaviestFlavour(int);

    Handle<JetMatchedPartonsCollection> theTagByRef;
    EDGetTokenT<JetMatchedPartonsCollection> sourceByReferToken_;
    bool physDefinition;
    bool leptonInfo_;
    DEFINITION_T definition;
    math::XYZTLorentzVector thePartonLV;

};
} }
using reco::modules::JetFlavourIdentifier;

//=========================================================================

JetFlavourIdentifier::JetFlavourIdentifier( const edm::ParameterSet& iConfig )
{
    produces<JetFlavourMatchingCollection>();
    sourceByReferToken_ = consumes<JetMatchedPartonsCollection>(iConfig.getParameter<InputTag>("srcByReference"));
    physDefinition = iConfig.getParameter<bool>("physicsDefinition");
    leptonInfo_ = iConfig.exists("leptonInfo") ? iConfig.getParameter<bool>("leptonInfo") : true;
    // If we have a definition of which parton to identify, use it,
    // otherwise we default to the "old" behavior of either "physics" or "algorithmic".
    // Furthermore, if the specified definition is not sensible for the given jet,
    // then the "physDefinition" switch is used to identify the flavour of the jet.
    if ( iConfig.exists("definition") ) {
      definition = static_cast<DEFINITION_T>( iConfig.getParameter<int>("definition") );
    } else {
      definition = NULL_DEF;
    }
}

//=========================================================================

JetFlavourIdentifier::~JetFlavourIdentifier()
{
}

// ------------ method called to produce the data  ------------

void JetFlavourIdentifier::produce( Event& iEvent, const EventSetup& iEs )
{
  // Get the JetMatchedPartons
  iEvent.getByToken (sourceByReferToken_, theTagByRef);

  // Create a JetFlavourMatchingCollection
  JetFlavourMatchingCollection *jfmc;
  if (!theTagByRef->empty()) {
    RefToBase<Jet> jj = theTagByRef->begin()->first;
    jfmc = new JetFlavourMatchingCollection(RefToBaseProd<Jet>(jj));
  } else {
    jfmc = new JetFlavourMatchingCollection();
  }
  auto_ptr<reco::JetFlavourMatchingCollection> jetFlavMatching(jfmc);

  // Loop over the matched partons and see which match.
  for ( JetMatchedPartonsCollection::const_iterator j  = theTagByRef->begin();
                                                    j != theTagByRef->end();
                                                    j ++ ) {


    // Consider this match.
    const MatchedPartons aMatch = (*j).second;

    // This will hold the 4-vector, vertex, flavour and the leptonian decays (0: no lepton, xyz: x leptons in layer 2, y in layer 1 and z in layer 0) of the requested object.
    math::XYZTLorentzVector thePartonLorentzVector(0,0,0,0);
    math::XYZPoint          thePartonVertex(0,0,0);
    int                     thePartonFlavour = 0;
    JetFlavour::Leptons     theLeptons;

    // get the partons based on which definition to use.
    switch (definition) {
      case PHYSICS: {
        const GenParticleRef aPartPhy = aMatch.physicsDefinitionParton();
        if (aPartPhy.isNonnull()) {
	        thePartonLorentzVector = aPartPhy.get()->p4();
	        thePartonVertex        = aPartPhy.get()->vertex();
	        thePartonFlavour       = aPartPhy.get()->pdgId();
          if (leptonInfo_) theLeptons = findLeptons(aPartPhy);
        }
        break;
      }
      case ALGO: {
        const GenParticleRef aPartAlg = aMatch.algoDefinitionParton();
        if (aPartAlg.isNonnull()) {
	        thePartonLorentzVector = aPartAlg.get()->p4();
	        thePartonVertex        = aPartAlg.get()->vertex();
	        thePartonFlavour       = aPartAlg.get()->pdgId();
          if (leptonInfo_) theLeptons = findLeptons(aPartAlg);
        }
        break;
      }
      case NEAREST_STATUS2 : {
        const GenParticleRef aPartN2 = aMatch.nearest_status2();
        if (aPartN2.isNonnull()) {
	        thePartonLorentzVector = aPartN2.get()->p4();
	        thePartonVertex        = aPartN2.get()->vertex();
	        thePartonFlavour       = aPartN2.get()->pdgId();
          if (leptonInfo_) theLeptons = findLeptons(aPartN2);
        }
        break;
      }
      case NEAREST_STATUS3: {
        const GenParticleRef aPartN3 = aMatch.nearest_status3();
        if (aPartN3.isNonnull()) {
	        thePartonLorentzVector = aPartN3.get()->p4();
	        thePartonVertex        = aPartN3.get()->vertex();
	        thePartonFlavour       = aPartN3.get()->pdgId();
          if (leptonInfo_) theLeptons = findLeptons(aPartN3);
        }
        break;
      }
      case HEAVIEST: {
        const GenParticleRef aPartHeaviest = aMatch.heaviest();
        if (aPartHeaviest.isNonnull()) {
	        thePartonLorentzVector = aPartHeaviest.get()->p4();
	        thePartonVertex        = aPartHeaviest.get()->vertex();
	        thePartonFlavour       = aPartHeaviest.get()->pdgId();
          if (leptonInfo_) theLeptons = findLeptons(aPartHeaviest);
        }
        break;
      }
      // Default case is backwards-compatible
      default:{
        if (physDefinition) {
          const GenParticleRef aPartPhy = aMatch.physicsDefinitionParton();
	  if (aPartPhy.isNonnull()) {
            thePartonLorentzVector = aPartPhy.get()->p4();
            thePartonVertex        = aPartPhy.get()->vertex();
            thePartonFlavour       = aPartPhy.get()->pdgId();
            if (leptonInfo_) theLeptons = findLeptons(aPartPhy);
          }
        } else {
          const GenParticleRef aPartAlg = aMatch.algoDefinitionParton();
          if (aPartAlg.isNonnull()) {
            thePartonLorentzVector = aPartAlg.get()->p4();
            thePartonVertex        = aPartAlg.get()->vertex();
            thePartonFlavour       = aPartAlg.get()->pdgId();
            if (leptonInfo_) theLeptons = findLeptons(aPartAlg);
          }
        }
      } break;
    } // end switch on definition

    // Now make sure we have a match. If the user specified "heaviest", for instance,
    // and there is no b- or c-quarks, then fall back to the "physDefinition" switch.

    if (thePartonFlavour == 0) {
      if (physDefinition) {
        const GenParticleRef aPartPhy = aMatch.physicsDefinitionParton();
        if (aPartPhy.isNonnull()) {
          thePartonLorentzVector = aPartPhy.get()->p4();
          thePartonVertex        = aPartPhy.get()->vertex();
          thePartonFlavour       = aPartPhy.get()->pdgId();
          if (leptonInfo_) theLeptons = findLeptons(aPartPhy);
        }
      } else {
        const GenParticleRef aPartAlg = aMatch.algoDefinitionParton();
        if (aPartAlg.isNonnull()) {
          thePartonLorentzVector = aPartAlg.get()->p4();
          thePartonVertex        = aPartAlg.get()->vertex();
          thePartonFlavour       = aPartAlg.get()->pdgId();
          if (leptonInfo_) theLeptons = findLeptons(aPartAlg);
        }
      }
    }

/*
     std::cout << "Leptons of " <<thePartonFlavour << " Jet: " << std::endl;
     std::cout << "  electrons: " <<theLeptons.electron << std::endl;
     std::cout << "  muons    : " <<theLeptons.muon << std::endl;
     std::cout << "  tau      : " <<theLeptons.tau << std::endl;
*/
    // Add the jet->flavour match to the map.
    (*jetFlavMatching)[(*j).first] = JetFlavour(thePartonLorentzVector, thePartonVertex, thePartonFlavour, theLeptons);
  }// end loop over jets


  // Put the object into the event.
  iEvent.put(  jetFlavMatching );

}

JetFlavour::Leptons JetFlavourIdentifier::findLeptons(const GenParticleRef &parton)
{
  JetFlavour::Leptons theLeptons;

  thePartonLV = parton->p4();

  ///first daughter of the parton should be an MC particle (pdgId==92,93)
  const reco::Candidate *mcstring = parton->daughter(0);
  int partonFlavour = std::abs(parton->pdgId());
//  std::cout << "parton DeltaR: " << DeltaR(thePartonLV, parton->p4()) << std::endl;

  ///lookup particles with parton flavour and weak decay
  std::vector<const reco::Candidate*> candidates = findCandidates(mcstring, partonFlavour);
//  std::cout << "Candidates are:" << std::endl;
//  for(unsigned int j = 0; j < candidates.size(); j++) std::cout << "   --> " << candidates[j]->pdgId() << std::endl;

  ///count leptons of candidates
  fillLeptons(candidates, theLeptons, 1, partonFlavour);

  return theLeptons;
}

std::vector<const reco::Candidate*> JetFlavourIdentifier::findCandidates(const reco::Candidate *cand, int partonFlavour)
{
  std::vector<const reco::Candidate*> cands;
  if(!cand) return cands;

  for(unsigned int i = 0; i < cand->numberOfDaughters(); i++) {
/*
    std::cout << "DeltaR - " << partonFlavour << " ";
    if (DeltaR(thePartonLV, cand->daughter(i)->p4()) > 0.7) std::cout << "(";
    std::cout << cand->daughter(i)->pdgId() << ": " << DeltaR(thePartonLV, cand->daughter(i)->p4());
    if (DeltaR(thePartonLV, cand->daughter(i)->p4()) > 0.7) std::cout << ")";
    std::cout << std::endl;
*/
    if (DeltaR(thePartonLV, cand->daughter(i)->p4()) < 0.7) {
      int pdgId = std::abs(cand->daughter(i)->pdgId());
      int flavour = heaviestFlavour(pdgId);
      if (flavour == partonFlavour ||
          (flavour >= 10 && partonFlavour >= 10)) {
//        std::cout << "<------- " << std::endl;
        std::vector<const reco::Candidate*> newcands = findCandidates(cand->daughter(i), partonFlavour);
//        std::cout << " ------->" << std::endl;
        std::copy(newcands.begin(), newcands.end(), std::back_inserter(cands));
      }
      if (partonFlavour >= 10)
        cands.push_back(cand->daughter(i));
    }
  }

  if (cands.empty() && std::abs(cand->pdgId()) > 110 &&
      !(partonFlavour >= 4 && partonFlavour < 10 &&
        heaviestFlavour(cand->pdgId()) < 4))
    cands.push_back(cand);

  return cands;
}

void JetFlavourIdentifier::fillLeptons(const std::vector<const reco::Candidate*> &cands, JetFlavour::Leptons &leptons, int rank, int flavour)
{
  for(unsigned int j = 0; j < cands.size(); j++) {
    for(unsigned int i = 0; i < cands[j]->numberOfDaughters(); i++) {
      int pdgId = std::abs(cands[j]->daughter(i)->pdgId());

//      for(int z = 1; z <= rank; z *= 10) std::cout << " ------ ";
//      std::cout << pdgId << std::endl;

      ///test for neutrinos because of conversions and dalitz pions
      if (pdgId == 12)
        leptons.electron += rank;
      else if (pdgId == 14)
        leptons.muon += rank;
      else if (pdgId == 16)
        leptons.tau += rank;
      else {
        int heaviest = heaviestFlavour(pdgId);
        int heaviest_ = heaviest < 10 ? heaviest : 0;
        if (!heaviest || (flavour < 4 ? (heaviest_ < 4) : (heaviest >= 4))) {
          std::vector<const reco::Candidate*> newcands = findCandidates(cands[j]->daughter(i), heaviest);
          if (pdgId <= 110) newcands.push_back(cands[j]->daughter(i));
          fillLeptons(newcands, leptons, rank * 10, std::max(heaviest_, flavour));
        }
      }
    }
  }
}

int JetFlavourIdentifier::heaviestFlavour(int pdgId)
{
  int flavour = 0;

  pdgId = std::abs(pdgId) % 100000;
  if (pdgId > 110) {
    while(pdgId % 10 > 0 && pdgId % 10 < 6) {
      pdgId /= 10;
      if (pdgId % 10 > flavour)
        flavour = pdgId % 10;
    }
  } else
    flavour = pdgId;

  return flavour;
}


//define this as a plug-in
DEFINE_FWK_MODULE(JetFlavourIdentifier);
