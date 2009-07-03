// File: SISConeJetProducer.cc
// Description:  see SISConeJetProducer.h
// Author:  Fedor Ratnikov, Maryland, June 30, 2007
// $Id: SISConeJetProducer.cc,v 1.3 2009/07/02 21:21:15 srappocc Exp $
//
//--------------------------------------------
#include <memory>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "SISConeJetProducer.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

}

namespace cms
{


  struct GreaterByPtCaloTower {
    bool operator()( const JetReco::InputItem & t1, const JetReco::InputItem & t2 ) const {
      return t1->pt() > t2->pt();
    }
  };


  // Constructor takes input parameters now: to be replaced with parameter set.

  SISConeJetProducer::SISConeJetProducer(edm::ParameterSet const& conf):
    BaseJetProducer (conf),
    alg_(conf)
  {}


  // run algorithm itself
  bool SISConeJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
     //do not run algorithm for more than 1000 input elements
    if(fInput.size() > 1000) {
      // sort in pt
      JetReco::InputCollection & fInputMutable = const_cast<JetReco::InputCollection &>( fInput );
      GreaterByPtCaloTower                   pTComparator;
      std::sort(fInputMutable.begin(), fInputMutable.end(), pTComparator);
      // now restrict first 1000 events
      fInputMutable.resize(1000);
      edm::LogWarning("SISConeTooManyEntries") << "Too many calo towers in the event, sisCone is limiting to first 1000. Output is suspect.";
//       edm::LogError("SISConeNotRun") << "Too many calo tower in the event, sisCone collection will be empty";
//       return false;
      alg_.run( fInputMutable, fOutput );
      return true;
    }
    
    alg_.run (fInput, fOutput);
    return true;
  }
}

