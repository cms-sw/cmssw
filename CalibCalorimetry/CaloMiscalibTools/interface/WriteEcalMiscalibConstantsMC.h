/*     <Notes on implementation>
 */
//
// Original Author:  Stephanie BEAUCERON
//         Created:  Tue May 15 16:23:21 CEST 2007
// $Id: WriteEcalMiscalibConstantsMC.h,v 1.1 2009/04/08 22:29:38 fra Exp $
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

class WriteEcalMiscalibConstantsMC : public edm::EDAnalyzer {
 public:
  explicit WriteEcalMiscalibConstantsMC(const edm::ParameterSet&);
  ~WriteEcalMiscalibConstantsMC();
  
  
 private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      std::string newTagRequest_;
};
