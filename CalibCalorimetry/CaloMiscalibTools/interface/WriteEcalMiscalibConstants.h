/*     <Notes on implementation>
 */
//
// Original Author:  Stephanie BEAUCERON
//         Created:  Tue May 15 16:23:21 CEST 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class decleration
//

class WriteEcalMiscalibConstants : public edm::EDAnalyzer {
 public:
  explicit WriteEcalMiscalibConstants(const edm::ParameterSet&);
  ~WriteEcalMiscalibConstants() override;
  
  
 private:
      void beginJob() override ;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
      
      // ----------member data ---------------------------
      std::string newTagRequest_;
};
