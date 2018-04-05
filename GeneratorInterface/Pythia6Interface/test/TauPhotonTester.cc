#include <iostream>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "HepPDT/ParticleDataTable.hh"


class TauPhotonTester : public edm::EDAnalyzer
{

   public:
   
      //
      explicit TauPhotonTester( const edm::ParameterSet& ) ;
      virtual ~TauPhotonTester() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
      virtual void beginJob() override;
      virtual void beginRun( const edm::Run &, const edm::EventSetup& ) override;
      virtual void endJob() override;

   private:
   
      int    fTauPhotonCounter;
      int    fTauDecPhotonCounter;
      int    fTauPhotonVtxCounter;
      TH1D*  fPhotonEnergy;
      TH1D*  fPhotonPt;
      TH1D*  fPhotonEnergyGt10MeV;
      TH1D*  fPhotonPtGt10MeV;
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      int    fVerbosity;

     
}; 

using namespace edm;
using namespace std;

TauPhotonTester::TauPhotonTester( const ParameterSet& pset )
  : fTauPhotonCounter(0), fTauDecPhotonCounter(0), fTauPhotonVtxCounter(0), fVerbosity(0)
{

   fVerbosity = pset.getUntrackedParameter<int>( "Verbosity", 0 );

}

void TauPhotonTester::beginJob()
{
  
  Service<TFileService> fs;
  fPhotonEnergy = fs->make<TH1D>(  "PhotonEnergy", "Energy of the Brem Photon (off tau)", 
                                   100,  0.0, 2.0 );
  fPhotonPt     = fs->make<TH1D>(  "PhotonPt", "Pt of the Brem Photon (off tau)", 
                                   100,  0.0, 2.0 );
  fPhotonEnergyGt10MeV = fs->make<TH1D>(  "PhotonEnergyGt10MeV", "Energy of the Brem Photon (off tau), Pt>10MeV", 
                                   100,  0.01, 2.01 );
  fPhotonPtGt10MeV     = fs->make<TH1D>(  "PhotonPtGt10MeV", "Pt of the Brem Photon (off tau), Pt>10MeV", 
                                   100,  0.01, 2.01 );
  
  return ;
  
}

void TauPhotonTester::beginRun( const edm::Run& r, const edm::EventSetup& es )
{

   es.getData( fPDGTable ) ;
      
   return ;

}

void TauPhotonTester::analyze( const Event& e, const EventSetup& )
{
    
   // here's an example of accessing particles in the event record (HepMCProduct)
   //
   Handle< HepMCProduct > EvtHandle ;
  
   // find initial (unsmeared, unfiltered,...) HepMCProduct
   //
   e.getByLabel("VtxSmeared", EvtHandle);
  
   const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
   
   // loop over, find a vertex with outgoing tau of no-null vtx
   // see if there's also a photon
   //
   bool tau_in_vtx_flag = false;
   bool tau_out_vtx_flag = false;
   
   vector<double> PhotonE;
   vector<double> PhotonPt;
   
   for ( HepMC::GenEvent::vertex_const_iterator vtx = Evt->vertices_begin();
	                                  vtx != Evt->vertices_end(); ++vtx ) 
   {
      
      tau_in_vtx_flag = false;
      tau_out_vtx_flag = false;
            
      if ( (*vtx)->particles_in_size() == 1 && 
	         abs((*((*vtx)->particles_in_const_begin()))->pdg_id()) == 15 )
      {
         tau_in_vtx_flag = true;
      } 
      
      if ( (*vtx)->particles_out_size() == 1 && 
	   abs((*((*vtx)->particles_out_const_begin()))->pdg_id()) == 15 &&
           tau_in_vtx_flag )
      {
         continue; // "intermediate" tau vtx, skip it
      } 
      
      
      int NGammaInVtx = 0;
      int NTauInVtx = 0;
      PhotonE.clear();
      PhotonPt.clear();
      
      for (HepMC::GenVertex::particle_iterator pitr= (*vtx)->particles_begin(HepMC::children);
                                               pitr != (*vtx)->particles_end(HepMC::children); ++pitr) 
      {
         if ( abs((*pitr)->pdg_id()) == 15 ) 
	 {
	    NTauInVtx++ ;
	    tau_out_vtx_flag = true;
	 }
	 if ( (*pitr)->pdg_id() == 22 ) 
	 {
	    NGammaInVtx++ ;
	    double ee = (*pitr)->momentum().t();
	    double px = (*pitr)->momentum().x();
	    double py = (*pitr)->momentum().y();
	    double pt = sqrt( px*px + py*py );
	    PhotonE.push_back( ee );
	    PhotonPt.push_back( pt );
	 }
      } 
      
      if ( !tau_in_vtx_flag && !tau_out_vtx_flag ) continue; // no-tau vtx  
      
      if ( tau_in_vtx_flag && tau_out_vtx_flag && NGammaInVtx > 0 )
      {
         if ( fVerbosity > 0 ) (*vtx)->print();
	 fTauPhotonVtxCounter++;
	 fTauPhotonCounter += NGammaInVtx;
	 for ( int i1=0; i1<NGammaInVtx; i1++ )
	 {
	    fPhotonEnergy->Fill( PhotonE[i1] );
	    fPhotonPt->Fill( PhotonPt[i1] );
	    if ( PhotonPt[i1] > 0.01 )
	    {
	       fPhotonEnergyGt10MeV->Fill( PhotonE[i1] );
	       fPhotonPtGt10MeV->Fill( PhotonPt[i1] );
	    }
	 }
      }
      
      if ( !tau_in_vtx_flag && tau_out_vtx_flag )
      {
         if ( fVerbosity > 0 && NGammaInVtx > 0 ) (*vtx)->print();
	 if ( NGammaInVtx > 0 )
	 {
	    fTauPhotonCounter += NGammaInVtx;
	    for ( int i2=0; i2<NGammaInVtx; i2++ )
	    {
	       fPhotonEnergy->Fill( PhotonE[i2] );
	       fPhotonPt->Fill( PhotonPt[i2] );
	       if ( PhotonPt[i2] > 0.01 )
	       {
	          fPhotonEnergyGt10MeV->Fill( PhotonE[i2] );
	          fPhotonPtGt10MeV->Fill( PhotonPt[i2] );
	       }
	    }
	 }
      }
      
      if ( tau_in_vtx_flag && !tau_out_vtx_flag ) // real tau decay vtx
      {
	 if ( fVerbosity > 0 && NGammaInVtx > 0 ) (*vtx)->print();
	 if ( NGammaInVtx > 0 )
	 {
            fTauDecPhotonCounter += NGammaInVtx;
	    for ( int i3=0; i3<NGammaInVtx; i3++ )
	    {
	       fPhotonEnergy->Fill( PhotonE[i3] );
	       fPhotonPt->Fill( PhotonPt[i3] );
	       if ( PhotonPt[i3] > 0.01 )
	       {
	          fPhotonEnergyGt10MeV->Fill( PhotonE[i3] );
	          fPhotonPtGt10MeV->Fill( PhotonPt[i3] );
	       }
	    }
	 }
      }
      
   }
     
   return ;
   
}

void TauPhotonTester::endJob()
{
   
   std::cout << "Tau-Photon Vtx Counter: " << fTauPhotonVtxCounter << std::endl;
   std::cout << "Tau-Photon Counter: " << fTauPhotonCounter << std::endl;
   std::cout << "TauDec-Photon Counter: " << fTauDecPhotonCounter << std::endl;
   
   return ;
}
 
DEFINE_FWK_MODULE(TauPhotonTester);
