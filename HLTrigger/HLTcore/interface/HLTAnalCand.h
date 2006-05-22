#ifndef HLTAnalCand_h
#define HLTAnalCand_h

/** \class HLTAnalCand
 *
 *  
 *  This class is an EDAnalyzer implementing a very basic HLT
 *  EDProduct analysis
 *
 *  $Date: 2006/05/20 15:33:35 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>
//
// class decleration
//

class HLTAnalCand : public edm::EDAnalyzer {

   public:
      explicit HLTAnalCand(const edm::ParameterSet&);
      ~HLTAnalCand();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:
      std::string src_;  // module label for product to analyse from the event
};

#endif //HLTAnalCand_h
