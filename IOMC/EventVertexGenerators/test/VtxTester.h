#ifndef VtxTester_H
#define VtxTester_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

// forward declarations
class TFile;
class TH1D;
class TH2D;

class VtxTester : public edm::EDAnalyzer
{

   public:
   
      //
      explicit VtxTester( const edm::ParameterSet& ) ;
      virtual ~VtxTester() {}
      
      virtual void analyze( const edm::Event&, const edm::EventSetup&) override;
      virtual void beginJob() ;
      virtual void endJob() ;

   private:
   
     //
     TFile* fOutputFile ;
     TH1D*  fVtxHistz ;
	 TH1D*  fVtxHistx ;
	 TH1D*  fVtxHisty ;
	 TH2D* fVtxHistxy;
     TH1D*  fPhiHistOrg ;
     TH1D*  fPhiHistSmr ;
     TH1D*  fEtaHistOrg ;
     TH1D*  fEtaHistSmr ; 
     
};

#endif
