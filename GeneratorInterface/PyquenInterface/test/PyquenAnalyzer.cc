// $Id: PyquenAnalyzer.cc,v 1.10 2010/01/12 11:59:35 hegner Exp $

#include <iostream>

#include "PyquenAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/Common/interface/Handle.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HepMC/HeavyIon.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

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
  : phdNdEta(0), phdNdY(0), phdNdPt(0),phdNdPhi(0)
{
  // constructor
}


//_______________________________________________________________________
void PyquenAnalyzer::beginJob()
{
  //runs at the begining of the job
  edm::Service<TFileService> fs;
  TH1::SetDefaultSumw2(true);

  phdNdEta      = fs->make<TH1D>("phdNdEta",";#eta;",100,-10.,10.);
  phdNdY        = fs->make<TH1D>("phdNdY",";y;",100,-10.,10.) ;
  phdNdPt       = fs->make<TH1D>("phdNdPt",";p_{T}(GeV/c);",100, 0.,10.) ;    
  phdNdPhi      = fs->make<TH1D>("phdNdPhi",";d#phi(rad);",100,-3.15,3.15);

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

   //  EvtHandle->GetEvent()->print();
   
   double part_eta, part_y, part_pt, part_phi, part_e, part_pz;
   const HepMC::GenEvent* myEvt = EvtHandle->GetEvent() ;
   for( HepMC::GenEvent::particle_const_iterator p = myEvt->particles_begin();
	p != myEvt->particles_end(); p++ )
     {
       if( !(*p)->end_vertex() && abs( (*p)->pdg_id() ) == 211)
	 {
	   part_eta = (*p)->momentum().eta();
	   part_e   = (*p)->momentum().e();
	   part_pt  = (*p)->momentum().perp();
	   part_phi = (*p)->momentum().phi();
           part_pz  = (*p)->momentum().z();
	   part_y = 0.5*log((part_e+part_pz)/(part_e-part_pz));

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

}

//define as a plug-in
DEFINE_FWK_MODULE(PyquenAnalyzer);
