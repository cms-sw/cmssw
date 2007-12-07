#include <iostream>

#include "PyquenAnalyzer.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HepMC/HeavyIon.h"

 
#include "TFile.h"
#include "TH1.h"
 
using namespace edm;
using namespace std;

/************************************************************************
 *
 *  This code is analyzing pyquen events produced with pyquen+analysis.cfg
 *
 *  Author: cmironov@lanl.gov
 *
 *************************************************************************/

 
PyquenAnalyzer::PyquenAnalyzer(const ParameterSet& pset)
  : sOutFileName(pset.getUntrackedParameter<string>("HistOutFile",std::string("testPyquen.root")) ),
pfOutFile(0), phdNdEta(0), phdNdY(0), phdNdPt(0),phdNdPhi(0)
{
  // constructor

}


//_______________________________________________________________________
void PyquenAnalyzer::beginJob( const EventSetup& )
{
  //runs at the begining of the job

   pfOutFile     = new TFile(sOutFileName.c_str(),"RECREATE");
   phdNdEta      = new TH1D("phdNdEta",";#eta;",100,-10.,10.);
   phdNdY        = new TH1D("phdNdY",";y;",100,-10.,10.) ;
   phdNdPt       = new TH1D("phdNdPt",";p_{T}(GeV/c);",100, 0.,10.) ;    
   phdNdPhi      = new TH1D("phdNdPhi",";d#phi(rad);",100,-3.15,3.15);

   return ;
}
 

//______________________________________________________________________
void PyquenAnalyzer::analyze( const Event& e, const EventSetup& )
{
  //runs every event

   Handle<HepMCProduct> EvtHandle;
   
   // find initial (unsmeared, unfiltered,...) HepMCProduct
   // by its label - PyquenSource, that is
   e.getByLabel( "source", EvtHandle ) ;
   
   double part_eta, part_y, part_pt, part_phi;
   const HepMC::GenEvent* myEvt = EvtHandle->GetEvent() ;
   for( HepMC::GenEvent::particle_const_iterator p = myEvt->particles_begin();
	p != myEvt->particles_end(); p++ )
     {
       if( !(*p)->end_vertex() && abs( (*p)->pdg_id() ) == 211)
	 {
	   part_eta = (*p)->momentum().eta();
	   part_y   = (*p)->momentum().y();
	   part_pt  = (*p)->momentum().perp();
	   part_phi = (*p)->momentum().phi();
	   
	   phdNdEta->Fill(part_eta);
	   phdNdY->Fill(part_y);
	   phdNdPt->Fill(part_pt);
	   phdNdPhi->Fill(part_phi);
	 }
     }

   return ;   
}


//_____________________________________________________________
void PyquenAnalyzer::endJob()
{
  // executed at the end of the job 

  phdNdEta->Scale(phdNdEta->GetBinWidth(0));
  phdNdY->Scale(phdNdY->GetBinWidth(0));
  phdNdPt->Scale(phdNdPt->GetBinWidth(0));
  phdNdPhi->Scale(phdNdPhi->GetBinWidth(0));  

  pfOutFile->Write();
  pfOutFile->Close();  
  return ;
}

//define as a plug-in
DEFINE_FWK_MODULE(PyquenAnalyzer);
