#ifndef VtxTester_H
#define VtxTester_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

// forward declarations
class TFile;
class TH1D;

class VtxTester : public edm::EDAnalyzer
{

   public:
   
      //
      explicit VtxTester( const edm::ParameterSet& ) ;
      virtual ~VtxTester() {}
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob( const edm::EventSetup& ) ;
      virtual void endJob() ;

   private:
   
     //
     TFile* fOutputFile ;
     TH1D*  fVtxHist ;
     TH1D*  fPhiHistO ;
     TH1D*  fPhiHistS ;
     TH1D*  fEtaHistO ;
     TH1D*  fEtaHistS ; 
     
};

#endif
