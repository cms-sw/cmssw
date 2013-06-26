#ifndef AnalysisJV_H
#define AnalysisJV_H

// system include files
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "TFile.h"
#include "TH1.h"

class TFile;
class TH1D;

//
// class decleration
//

class AnalysisJV : public edm::EDAnalyzer {
   public:
      explicit AnalysisJV(const edm::ParameterSet&);
      ~AnalysisJV();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::string fOutputFileName ;

      TFile*      fOutputFile ;
      TH1D*       fHistAlpha ;

  
};
#endif
