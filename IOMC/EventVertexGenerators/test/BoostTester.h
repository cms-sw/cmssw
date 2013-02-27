#ifndef BoostTester_H
#define BoostTester_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "TFile.h"
#include "TTree.h"

// forward declarations
//class TFile;
//class TTree;
//class TH1D;
//class TH2D;

class BoostTester : public edm::EDAnalyzer
{

   public:
   
      //
      explicit BoostTester( const edm::ParameterSet& ) ;
      virtual ~BoostTester() {}
      
      virtual void analyze( const edm::Event&, const edm::EventSetup&) override;
      virtual void beginJob() ;
      virtual void endJob() ;

   private:
   
     //
     TFile* fOutputFile ;
	 TTree* ftreevtx;
	 TTree* ftreep;
	 
	 double fvx,fvy,fvz;
	 double fpx,fpy,fpz,fpt,fp,fe,feta,fphi;
	 
	 /*
     TH1D*  fVtxHistz ;
     TH1D*  fVtxHistx ;
     TH1D*  fVtxHisty ;
     TH2D* fVtxHistxy;
     TH1D*  fpxHist;
     TH1D*  fpyHist;
     TH1D*  fpzHist;
     TH1D*  fpHist;
     TH1D*  feHist;
	 TH1D*  fptHist;
	 
     TH1D*  fPhiHistOrg ;
     TH1D*  fPhiHistSmr ;
     TH1D*  fEtaHistOrg ;
     TH1D*  fEtaHistSmr ; 
     */
	 
};

#endif
