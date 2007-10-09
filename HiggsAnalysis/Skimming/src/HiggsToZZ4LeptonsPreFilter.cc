
/* \class HiggsTo4LeptonsPreFilter 
 *
 * Consult header file for description
 *
 * author:  Dominique Fortin - UC Riverside
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsPreFilter.h>

// User include files
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

// C++
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsToZZ4LeptonsPreFilter::HiggsToZZ4LeptonsPreFilter(const edm::ParameterSet& pset) {

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsToZZ4LeptonsPreFilter");

  ikept = 0;
  evt = 0;
}


// Destructor
HiggsToZZ4LeptonsPreFilter::~HiggsToZZ4LeptonsPreFilter() {

  std::cout << "number of events processed: " << evt << std::endl;
  std::cout << "number of events kept: " << ikept << std::endl;

}


// Filter event
bool HiggsToZZ4LeptonsPreFilter::filter(edm::Event& event, const edm::EventSetup& setup ) {

  bool keepEvent   = false;
  evt++;


  // get gen particle candidates 
  Handle<CandidateCollection> genCandidates;

  try {
    event.getByLabel("genParticleCandidates", genCandidates);


  int nElec = 0;
  int nMuon = 0;

  for ( CandidateCollection::const_iterator mcIter=genCandidates->begin(); mcIter!=genCandidates->end(); ++mcIter ) {
    // Muons:
    if ( mcIter->pdgId() == 13 || mcIter->pdgId() == -13) {
      // Mother is a Z
      if ( mcIter->mother()->pdgId() == 23 ) {
       // In fiducial volume:
        if ( mcIter->eta() > -2.4 && mcIter->eta() < 2.4 ) nMuon++;
      }
    }
    // Electrons:
    if ( mcIter->pdgId() == 11 || mcIter->pdgId() == -11) 
      // Mother is a Z
      if ( mcIter->mother()->pdgId() == 23 ) {
        // In fiducial volume:
        if ( mcIter->eta() > -2.4 && mcIter->eta() < 2.4 ) nElec++;
      }
    }
     
  if (nElec > 3) keepEvent = true;
  if (nMuon > 3) keepEvent = true;
  if (nMuon > 1 && nElec > 1) keepEvent = true;

  }

  catch(const edm::Exception& e) {
    //wrong reason for exception
    if ( e.categoryCode() != edm::errors::ProductNotFound ) throw;    
  }

  if (keepEvent ) ikept++;

  return keepEvent;

}


