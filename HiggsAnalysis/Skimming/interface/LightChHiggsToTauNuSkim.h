#ifndef LightChHiggsToTauNuSkim_h
#define LightChHiggsToTauNuSkim_h

/** \class HeavyChHiggsToTauNuSkim
 *
 *  
 *  Filter to select events passing 
 *  HLT muon/electron trigger
 *  with at least one offline lepton and two jets
 *
 *  \author Nuno Almeida  -  LIP Lisbon
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

using namespace edm;
using namespace std;

class LightChHiggsToTauNuSkim : public edm::EDFilter {

    public:
     explicit LightChHiggsToTauNuSkim(const edm::ParameterSet&);
     ~LightChHiggsToTauNuSkim();

     virtual bool filter(edm::Event&, const edm::EventSetup& );

   private:

  
    
     InputTag jetsTag_;
     InputTag muonsTag_;
     InputTag electronsTag_;
    
     int minNumbOfjets_;
     double jetPtMin_;
     double jetEtaMin_;
     double jetEtaMax_;

     double leptonPtMin_;
     double leptonEtaMin_;
     double leptonEtaMax_;

     double nEvents_;
     double nSelectedEvents_;

};
#endif
