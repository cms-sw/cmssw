//
// Translation of BTag MCJetFlavour tool to identify real flavour of a jet
// work with CaloJet objects
// Store Infos by Values in JetFlavour.h
// Author: Attilio
// Date: 05.10.2007
//
//
// \class JetPartonMatcher
//
// \brief Interface to create a specified "matched" parton to jet match based
//        on the user interpretation.
//
// Algorithm for definitions:
//
// If the particle is matched to any item in the priority list (given by user),
// then count that prioritized particle as the match.
//
// If not resort to default behavior:
//
//
// 1) Algorithmic Definition:
//
// A particle is a parton if its daughter is a string(code=92) or a cluster(code=93)
// If (parton is within the cone defined by coneSizeToAssociate) then:
//           if (parton is a b)                                   then associatedParton is the b
//      else if (associatedParton =! b and parton is a c)         then associatedParton is the c
//      else if (associatedParton =! b and associatedParton =! c) then associatedParton is the one with the highest pT
// associatedParton can be -1 --> no partons in the cone
// True Flavour of the jet --> flavour of the associatedParton
//
// ToDo: if more than one b(c) in the cone --> the selected parton is not always the one with highest pT
//
//
// 2) Physics Definition:
//
// A particle is a parton if its daughter is a string(code=92) or a cluster(code=93)
// if( only one initialParticle within the cone defined by coneSizeToAssociate) associatedInitialParticle is the initialParticle
// TheBiggerConeSize = 0.7 --> it's hard coded!
// if( a parton not coming from associatedInitialParticle in TheBiggerConeSize is a b or a c) reject the association
//
// associatedInitialParticle can be -1 --> no initialParticle in the cone or rejected association
// True Flavour of the jet --> flavour of the associatedInitialParticle
//
//
// Modifications:
//
//     09.03.2008: Sal Rappoccio.
//                 Added capability for "priority" list.
//

//=======================================================================

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/MatchedPartons.h"
#include "SimDataFormats/JetMatching/interface/JetMatchedPartons.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/getRef.h"
//#include "DataFormats/Candidate/interface/Candidate.h"
//#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

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

class JetPartonMatcher : public edm::global::EDProducer<>
{
  public:
    JetPartonMatcher( const edm::ParameterSet & );
    ~JetPartonMatcher() override;

  private:
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup& ) const override;

    struct WorkingVariables {
      Handle <GenParticleRefVector> particles;
      int theHeaviest = 0;
      int theNearest2 =0;
      int theNearest3 = 0;
      int theHardest = 0;
    };

    int fillAlgoritDefinition( const Jet&, WorkingVariables& ) const;
    int fillPhysicsDefinition( const Jet&, WorkingVariables& ) const;

    edm::EDGetTokenT<edm::View <reco::Jet> > m_jetsSrcToken;
    edm::EDGetTokenT<GenParticleRefVector> m_ParticleSrcToken;
    double coneSizeToAssociate;
    bool physDefinition;
    bool doPriority;
    vector<int> priorityList;

};

//=========================================================================

JetPartonMatcher::JetPartonMatcher( const edm::ParameterSet& iConfig ) 
{
    produces<JetMatchedPartonsCollection>();
    m_jetsSrcToken           = consumes<edm::View <reco::Jet> >(iConfig.getParameter<edm::InputTag>("jets"));
    m_ParticleSrcToken       = consumes<GenParticleRefVector>(iConfig.getParameter<edm::InputTag>("partons"));
    coneSizeToAssociate = iConfig.getParameter<double>("coneSizeToAssociate");
    if ( iConfig.exists("doPriority") ) {
      doPriority = iConfig.getParameter<bool>("doPriority");
      priorityList = iConfig.getParameter<vector<int> >("priorityList");
    } else {
      doPriority = false;
      priorityList.clear();
    }
}

//=========================================================================

JetPartonMatcher::~JetPartonMatcher()
{
}

// ------------ method called to produce the data  ------------

void JetPartonMatcher::produce(edm::StreamID, Event& iEvent, const EventSetup& iEs ) const
{
  WorkingVariables wv;
  edm::Handle <edm::View <reco::Jet> > jets_h;
  iEvent.getByToken(m_jetsSrcToken,     jets_h    );
  iEvent.getByToken(m_ParticleSrcToken, wv.particles );

  edm::LogVerbatim("JetPartonMatcher") << "=== Partons size:" << wv.particles->size();

  for( size_t m = 0; m < wv.particles->size(); ++ m ) {
    const GenParticle & aParton = *(wv.particles->at(m).get());
    edm::LogVerbatim("JetPartonMatcher") <<  aParton.status() << " " <<
                                             aParton.pdgId()  << " " <<
                                             aParton.pt()     << " " <<
                                             aParton.eta()    << " " <<
                                             aParton.phi()    << endl;
  }

  auto jetMatchedPartons = std::make_unique<JetMatchedPartonsCollection>(reco::JetRefBaseProd(jets_h));

  for (size_t j = 0; j < jets_h->size(); j++) {

    const int theMappedPartonAlg = fillAlgoritDefinition( (*jets_h)[j], wv );
    const int theMappedPartonPhy = fillPhysicsDefinition( (*jets_h)[j], wv );

    GenParticleRef pHV;
    GenParticleRef pN2;
    GenParticleRef pN3;
    GenParticleRef pPH;
    GenParticleRef pAL;

    if(wv.theHeaviest>=0)        pHV = wv.particles->at( wv.theHeaviest        );
    if(wv.theNearest2>=0)        pN2 = wv.particles->at( wv.theNearest2        );
    if(wv.theNearest3>=0)        pN3 = wv.particles->at( wv.theNearest3        );
    if(theMappedPartonPhy>=0) pPH = wv.particles->at( theMappedPartonPhy );
    if(theMappedPartonAlg>=0) pAL = wv.particles->at( theMappedPartonAlg );

    (*jetMatchedPartons)[jets_h->refAt(j)]=MatchedPartons(pHV,pN2,pN3,pPH,pAL);
  }

  iEvent.put(std::move(jetMatchedPartons));

}

//
// Algorithmic Definition:
// Output: define one associatedParton
// Loop on all particle.
//
// (Note: This part added by Salvatore Rappoccio)
// [begin]
// If the particle is matched to any item in the priority list (given by user),
// then count that prioritized particle as the match.
//
// If not resort to default behavior:
// [end]
//
// A particle is a parton if its daughter is a string(code=92) or a cluster(code=93)
// If (parton is within the cone defined by coneSizeToAssociate) then:
//           if (parton is a b)                                   then associatedParton is the b
//      else if (associatedParton =! b and parton is a c)         then associatedParton is the c
//      else if (associatedParton =! b and associatedParton =! c) then associatedParton is the one with the highest pT
// associatedParton can be -1 --> no partons in the cone
// True Flavour of the jet --> flavour of the associatedParton
//
// ToDo: if more than one b(c) in the cone --> the selected parton is not always the one with highest pT
//
int JetPartonMatcher::fillAlgoritDefinition( const Jet& theJet, WorkingVariables& wv ) const {

  int tempParticle = -1;
  int tempPartonHighestPt = -1;
  int tempNearest = -1;
  float maxPt = 0;
  float minDr = 1000;
  bool foundPriority = false;

  // Loop over the particles in question until we find a priority
  // "hit", or if we find none in the priority list (or we don't want
  // to consider priority), then loop through all particles and fill
  // standard definition.

  //
  // Matching:
  //
  // 1) First try to match by hand. The "priority list" is given
  //    by the user. The algorithm finds any particles in that
  //    "priority list" that are within the specified cone size.
  //    If it finds one, it counts the object as associating to that
  //    particle.
  //    NOTE! The objects in the priority list are given in order
  //    of priority. So if the user specifies:
  //    6, 24, 21,
  //    then it will first look for top quarks, then W bosons, then gluons.
  // 2) If no priority items are found, do the default "standard"
  //    matching.
  for( size_t m = 0; m < wv.particles->size() && !foundPriority; ++ m ) {
    //    const Candidate & aParton = *(wv.particles->at(m).get());
    const GenParticle & aParton = *(wv.particles->at(m).get());
    int flavour = abs(aParton.pdgId());

    // "Priority" behavoir:
    // Associate to the first particle found in the priority list, regardless
    // of delta R.
    if ( doPriority ) {
      vector<int>::const_iterator ipriority = find( priorityList.begin(), priorityList.end(), flavour );
      // Check to see if this particle is in our priority list
      if ( ipriority != priorityList.end() ) {
	// This particle is on our priority list. If it matches,
	// we break, since we take in order of priority, not deltaR
	double dist = DeltaR( theJet.p4(), aParton.p4() );
	if( dist <= coneSizeToAssociate ) {
	  tempParticle = m;
	  foundPriority = true;
	}
      }
    }
    // Here we do not want to do priority matching. Ensure the "foundPriority" swtich
    // is turned off.
    else {
      foundPriority = false;
    }

    // Default behavior:
    // Look for partons before the color string to associate.
    // Do this if we don't want to do priority matching, or if
    // we didn't find a priority object.

    if( !foundPriority && aParton.status() != 3 ) {
      double dist = DeltaR( theJet.p4(), aParton.p4() );
      if( dist <= coneSizeToAssociate ) {
	if( dist < minDr ) {
	  minDr = dist;
	  tempNearest = m;
	}
	if( tempParticle == -1 && ( flavour == 4 )  ) tempParticle = m;
	if(                         flavour == 5    ) tempParticle = m;
	if( aParton.pt() > maxPt ) {
	  maxPt = aParton.pt();
	  tempPartonHighestPt = m;
	}
      }
    }
  }

  if ( foundPriority ) {
    wv.theHeaviest = tempParticle; // The priority-matched particle
    wv.theHardest  = -1;  //  set the rest to -1
    wv.theNearest2 = -1;  // "                  "
  } else {
    wv.theHeaviest = tempParticle;
    wv.theHardest  = tempPartonHighestPt;
    wv.theNearest2 = tempNearest;
    if ( tempParticle == -1 ) tempParticle = tempPartonHighestPt;
  }
  return tempParticle;
}

//
// Physics Definition:
// A initialParticle is a particle with status=3
// Output: define one associatedInitialParticle
// Loop on all particles
//
// (Note: This part added by Salvatore Rappoccio)
// [begin]
// If the particle is matched to any item in the priority list (given by user),
// then count that prioritized particle as the match.
//
// If not resort to default behavior:
// [end]
//
// A particle is a parton if its daughter is a string(code=92) or a cluster(code=93)
// if( only one initialParticle within the cone defined by coneSizeToAssociate) associatedInitialParticle is the initialParticle
// TheBiggerConeSize = 0.7 --> it's hard coded!
// if( a parton not coming from associatedInitialParticle in TheBiggerConeSize is a b or a c) reject the association
//
// associatedInitialParticle can be -1 --> no initialParticle in the cone or rejected association
// True Flavour of the jet --> flavour of the associatedInitialParticle
//
int JetPartonMatcher::fillPhysicsDefinition( const Jet& theJet, WorkingVariables& wv ) const {

  float TheBiggerConeSize = 0.7; // In HepMC it's 0.3 --> it's a mistake: value has to be 0.7
  int tempParticle = -1;
  int nInTheCone = 0;
  int tempNearest = -1;
  float minDr = 1000;
  bool foundPriority = false;

  vector<const reco::Candidate *> theContaminations;
  theContaminations.clear();

  // Loop over the particles in question until we find a priority
  // "hit", or if we find none in the priority list (or we don't want
  // to consider priority), then loop through all particles and fill
  // standard definition.

  //
  // Matching:
  //
  // 1) First try to match by hand. The "priority list" is given
  //    by the user. The algorithm finds any particles in that
  //    "priority list" that are within the specified cone size.
  //    If it finds one, it counts the object as associating to that
  //    particle.
  //    NOTE! The objects in the priority list are given in order
  //    of priority. So if the user specifies:
  //    6, 24, 21,
  //    then it will first look for top quarks, then W bosons, then gluons.
  // 2) If no priority items are found, do the default "standard"
  //    matching.
  for( size_t m = 0; m < wv.particles->size() && !foundPriority; ++ m ) {

    //    const Candidate & aParticle = *(wv.particles->at(m).get());
    const GenParticle & aParticle = *(wv.particles->at(m).get());
    int flavour = abs(aParticle.pdgId());

    // "Priority" behavoir:
    // Associate to the first particle found in the priority list, regardless
    // of delta R.
    if ( doPriority ) {
      vector<int>::const_iterator ipriority = find( priorityList.begin(), priorityList.end(), flavour );
      // Check to see if this particle is in our priority list
      if ( ipriority != priorityList.end() ) {
	// This particle is on our priority list. If it matches,
	// we break, since we take in order of priority, not deltaR
	double dist = DeltaR( theJet.p4(), aParticle.p4() );
	if( dist <= coneSizeToAssociate ) {
	  tempParticle = m;
	  foundPriority = true;
	}
      }
    }
    // Here we do not want to do priority matching. Ensure the "foundPriority" swtich
    // is turned off.
    else {
      foundPriority = false;
    }

    // Default behavior:
    if ( !foundPriority ) {

      // skipping all particle but udscbg (is this correct/enough?!?!)
      bool isAParton = false;
      if(flavour == 1 ||
	 flavour == 2 ||
	 flavour == 3 ||
	 flavour == 4 ||
	 flavour == 5 ||
	 flavour == 21 ) isAParton = true;
      if(!isAParton) continue;
      double dist = DeltaR( theJet.p4(), aParticle.p4() );
      if( aParticle.status() == 3 && dist < minDr ) {
	minDr = dist;
	tempNearest = m;
      }
      if( aParticle.status() == 3 && dist <= coneSizeToAssociate ) {
	//cout << "particle in small cone=" << aParticle.pdgId() << endl;
	tempParticle = m;
	nInTheCone++;
      }
      // Look for heavy partons in TheBiggerConeSize now
      if( aParticle.numberOfDaughters() > 0 && aParticle.status() != 3 ) {
	if( flavour ==  1 ||
	    flavour ==  2 ||
	    flavour ==  3 ||
	    flavour == 21 ) continue;
	if( dist < TheBiggerConeSize ) theContaminations.push_back( &aParticle );
      }
    }
  }


  // Here's the default behavior for assignment if there is no priority.
  if ( !foundPriority ) {
    wv.theNearest3 = tempNearest;

    if(nInTheCone != 1) return -1; // rejected --> only one initialParton requested
    if(theContaminations.empty() ) return tempParticle; //no contamination
    int initialPartonFlavour = abs( (wv.particles->at(tempParticle).get()) ->pdgId() );

    vector<const Candidate *>::const_iterator itCont = theContaminations.begin();
    for( ; itCont != theContaminations.end(); itCont++ ) {
      int contaminatingFlavour = abs( (*itCont)->pdgId() );
      if( (*itCont)->numberOfMothers()>0 && (*itCont)->mother(0) == wv.particles->at(tempParticle).get() ) continue; // mother is the initialParton --> OK
      if( initialPartonFlavour == 4 ) {
	if( contaminatingFlavour == 4 ) continue; // keep association --> the initialParton is a c --> the contaminated parton is a c
	tempParticle = -1; // all the other cases reject!
	return tempParticle;
      }
    }
  }
  // If there is priority, then just set the heaviest to priority, the rest are -1.
  else {
    wv.theHeaviest = tempParticle; // Set the heaviest to tempParticle
    wv.theNearest2 = -1; //  Set the rest to -1
    wv.theNearest3 = -1; // "                  "
    wv.theHardest = -1;  // "                  "
  }

  return tempParticle;
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetPartonMatcher);

