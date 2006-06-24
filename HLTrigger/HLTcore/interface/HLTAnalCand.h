#ifndef HLTAnalCand_h
#define HLTAnalCand_h

/** \class HLTAnalCand
 *
 *  
 *  This class is an EDAnalyzer implementing a very basic HLT
 *  EDProduct analysis
 *
 *  $Date: 2006/05/22 22:04:50 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class HLTAnalCand : public edm::EDAnalyzer {

   public:
      explicit HLTAnalCand(const edm::ParameterSet&);
      ~HLTAnalCand();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_;  // input tag identifying product to analyze

};

#endif //HLTAnalCand_h
