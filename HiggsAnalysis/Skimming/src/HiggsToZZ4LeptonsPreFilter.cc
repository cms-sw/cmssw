
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

// LeptonFlavour
// 0 = no tau
// 1 = 4 mu
// 2 = 4 e
// 3 = 2e 2mu

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsToZZ4LeptonsPreFilter");
  leptonFlavour      = pset.getParameter<int>("HiggsToZZ4LeptonsPreFilterLeptonFlavour");

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

  bool FourL    = false;
  bool FourE    = false;
  bool FourM    = false;
  bool TwoETwoM = false;

  // get gen particle candidates 
  Handle<CandidateCollection> genCandidates;

  event.getByLabel("genParticleCandidates", genCandidates);

  if ( genCandidates.isValid() ) {

  int nElec = 0;
  int nMuon = 0;

  for ( CandidateCollection::const_iterator mcIter=genCandidates->begin(); mcIter!=genCandidates->end(); ++mcIter ) {
    // Muons:
    if ( mcIter->pdgId() == 13 || mcIter->pdgId() == -13) {
      // Mother is a Z
      if ( mcIter->mother()->pdgId() == 23 ) {
       // In fiducial volume:
        if ( mcIter->pt() < 3 ) continue;
        if ( mcIter->eta() > -2.4 && mcIter->eta() < 2.4 ) nMuon++;
      }
    }
    // Electrons:
    if ( mcIter->pdgId() == 11 || mcIter->pdgId() == -11) 
      // Mother is a Z
      if ( mcIter->mother()->pdgId() == 23 ) {
        // In fiducial volume:
        if ( mcIter->pt() < 3 ) continue;
        if ( mcIter->eta() > -2.5 && mcIter->eta() < 2.5 ) nElec++;
      }
    }

     
    if (nElec > 3) FourE = true;
    if (nMuon > 3) FourM = true;
    if (nMuon > 1 && nElec > 1) TwoETwoM = true;
    if ( FourE || FourM || TwoETwoM ) FourL = true;

    if ( leptonFlavour == 0 && FourL    ) keepEvent = true;    
    if ( leptonFlavour == 1 && FourM    ) keepEvent = true;    
    if ( leptonFlavour == 2 && FourE    ) keepEvent = true;    
    if ( leptonFlavour == 3 && TwoETwoM ) keepEvent = true;    

  }

  if (keepEvent ) ikept++;

  return keepEvent;

}


