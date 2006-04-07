
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
   fPhiHistO = 0 ;
   fPhiHistS = 0 ;
   fEtaHistO = 0 ;
   fEtaHistS = 0 ;
}

void VtxTester::beginJob( const EventSetup& )
{

   fOutputFile = new TFile( "VtxTest.root", "RECREATE" ) ;
   fVtxHist    = new TH1D("VtxHist","Test Z-spread", 100, -250., 250. ) ;
   fPhiHistO   = new TH1D("PhiHistO","Test Phi, org.", 80, -4., 4. ) ;
   fPhiHistS   = new TH1D("PhiHistS","Test Phi, smr.", 80, -4., 4. ) ;
   fEtaHistO   = new TH1D("EtaHistO","Test Eta, org.", 80, -4., 4. ) ;
   fEtaHistS   = new TH1D("EtaHistS","Test Eta, smr.", 80, -4., 4. ) ;

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
   
         HepMC::GenEvent::vertex_const_iterator Vtx = Evt->vertices_begin() ;
   
   
         HepMC::GenEvent::particle_const_iterator 
            Part = Evt->particles_begin() ;
         HepLorentzVector Mom = (*Part)->momentum() ;
         double Phi = Mom.phi() ;
         double Eta = -log(tan(Mom.theta()/2.));
	 
	 if ( (EvtHandles[i].provenance()->product).module.moduleLabel_ == "VtxSmeared" )
	 {	 
            fVtxHist->Fill( (*Vtx)->position().z() ) ;
            fPhiHistS->Fill( Phi ) ;
            fEtaHistS->Fill( Eta ) ;
         }
	 else
	 {
	    fPhiHistO->Fill( Phi ) ;
	    fEtaHistO->Fill( Eta ) ;
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

DEFINE_FWK_MODULE(VtxTester)
