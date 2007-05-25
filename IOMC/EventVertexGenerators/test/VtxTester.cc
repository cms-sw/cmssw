
#include <iostream>

#include "IOMC/EventVertexGenerators/test/VtxTester.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "TFile.h"
#include "TH1.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace std;

VtxTester::VtxTester( const ParameterSet& )
{
   fOutputFile = 0 ;
   fVtxHist = 0 ;
   fPhiHistOrg = 0 ;
   fPhiHistSmr = 0 ;
   fEtaHistOrg = 0 ;
   fEtaHistSmr = 0 ;
}

void VtxTester::beginJob( const EventSetup& )
{

   fOutputFile   = new TFile( "VtxTest.root", "RECREATE" ) ;
   fVtxHist      = new TH1D("VtxHist","Test Z-spread", 100, -250., 250. ) ;
   fPhiHistOrg   = new TH1D("PhiHistOrg","Test Phi, org.", 80, -4., 4. ) ;
   fPhiHistSmr   = new TH1D("PhiHistSmr","Test Phi, smr.", 80, -4., 4. ) ;
   fEtaHistOrg   = new TH1D("EtaHistOrg","Test Eta, org.", 80, -4., 4. ) ;
   fEtaHistSmr   = new TH1D("EtaHistSmr","Test Eta, smr.", 80, -4., 4. ) ;

   return ;
}

void VtxTester::analyze( const Event& e, const EventSetup& )
{
   
   vector< Handle< HepMCProduct > > EvtHandles ;
   e.getManyByType( EvtHandles ) ;
   
   for ( unsigned int i=0; i<EvtHandles.size(); i++ )
   {
   
      if ( EvtHandles[i].isValid() )
      {
   
         const HepMC::GenEvent* Evt = EvtHandles[i]->GetEvent() ;
   
         // take only 1st vertex for now - it's been tested only of PGuns...
	 //
	 HepMC::GenEvent::vertex_const_iterator Vtx = Evt->vertices_begin() ;
   
   
         for (HepMC::GenEvent::particle_const_iterator 
            Part = Evt->particles_begin() ; Part!=Evt->particles_end(); Part++ )
	 {
            HepLorentzVector Mom = (*Part)->momentum() ;
            double Phi = Mom.phi() ;
            double Eta = -log(tan(Mom.theta()/2.));
	 
	 if ( EvtHandles[i].provenance()->moduleLabel() == "VtxSmeared" )
	    {	 
               fVtxHist->Fill( (*Vtx)->position().z() ) ;
               fPhiHistSmr->Fill( Phi ) ;
               fEtaHistSmr->Fill( Eta ) ;
            }
	    else
	    {
	       fPhiHistOrg->Fill( Phi ) ;
	       fEtaHistOrg->Fill( Eta ) ;
	    }
	 }

      }

   }
   
   return ;
}

void VtxTester::endJob()
{
   
   fOutputFile->Write() ;
   fOutputFile->Close() ;
   
   return ;
}

DEFINE_FWK_MODULE(VtxTester);
