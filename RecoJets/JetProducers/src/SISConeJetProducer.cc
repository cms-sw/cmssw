// File: SISConeJetProducer.cc
// Description:  see SISConeJetProducer.h
// Author:  Fedor Ratnikov, Maryland, June 30, 2007
// $Id: SISConeJetProducer.cc,v 1.4 2009/07/03 15:06:00 srappocc Exp $
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
    alg_(conf),
    ncut_(conf.getParameter<uint>("maxInputSize"))
  {}


  // run algorithm itself
  bool SISConeJetProducer::runAlgorithm (const JetReco::InputCollection& fInput, 
		     JetReco::OutputCollection* fOutput) {
     //do not run algorithm for more than <ncut_> input elements
    if(fInput.size() > ncut_) {
      // sort in pt
      JetReco::InputCollection & fInputMutable = const_cast<JetReco::InputCollection &>( fInput );
      GreaterByPtCaloTower                   pTComparator;
      std::sort(fInputMutable.begin(), fInputMutable.end(), pTComparator);
      // now restrict to first <ncut_> entries
      fInputMutable.resize(ncut_);
      edm::LogWarning("SISConeTooManyEntries") << "Too many calo towers in the event, sisCone is limiting to first " << ncut_ << ". Output is suspect.";
//       edm::LogError("SISConeNotRun") << "Too many calo tower in the event, sisCone collection will be empty";
//       return false;
      alg_.run( fInputMutable, fOutput );
      return true;
    }
    
    alg_.run (fInput, fOutput);
    return true;
  }
}

