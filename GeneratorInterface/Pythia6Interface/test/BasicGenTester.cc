#include <iostream>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

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

class BasicGenTester : public edm::EDAnalyzer
{

   public:
   
      //
      explicit BasicGenTester( const edm::ParameterSet& ) ;
      virtual ~BasicGenTester() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob() ;
      virtual void beginRun( const edm::Run &, const edm::EventSetup& );
      virtual void endRun( const edm::Run&, const edm::EventSetup& ) ;
      virtual void endJob() ;

   private:
   
     TH1D*       fNChgPartFinalState ;
     TH1D*       fNNeuPartFinalState ;
     TH1D*       fNPartFinalState;
     TH1D*       fPtChgPartFinalState ;
     TH1D*       fPtNeuPartFinalState ;
     TH1D*       fEtaChgPartFinalState ;
     TH1D*       fEtaNeuPartFinalState ;
     int         fNPart;
     double      fPtMin;
     double      fPtMax;
     edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
     
}; 

using namespace edm;
using namespace std;

BasicGenTester::BasicGenTester( const ParameterSet& pset )
  : fNChgPartFinalState(0),  fNNeuPartFinalState(0), fNPartFinalState(0),
    fPtChgPartFinalState(0), fPtNeuPartFinalState(0)
{

   fNPart = pset.getUntrackedParameter<int>( "NPartForHisto", 500 );
   fPtMin = pset.getUntrackedParameter<double>( "PtMinForHisto",  0. );
   fPtMax = pset.getUntrackedParameter<double>( "PtMaxForHisto", 25. );   

}

void BasicGenTester::beginJob()
{
  
  Service<TFileService> fs;
  
  fNChgPartFinalState = fs->make<TH1D>(  "NChgPartFinalState",  "Number of final state charged particles", 
                                         fNPart,  0., (double)fNPart );
  fNNeuPartFinalState = fs->make<TH1D>(  "NNeuPartFinalState",  "Number of final state neutral particles", 
                                         fNPart,  0., (double)fNPart );
  fNPartFinalState = fs->make<TH1D>(     "NPartFinalState",  "Total number of final state particles", 
                                         fNPart,  0., (double)fNPart );
  fPtChgPartFinalState = fs->make<TH1D>( "PtChgPartFinalState", "Pt of final state charged particles", 
                                         500, fPtMin, fPtMax );
  fPtNeuPartFinalState = fs->make<TH1D>( "PtNeuPartFinalState", "Pt of final state neutral particles", 
                                         500, fPtMin, fPtMax );
  fEtaChgPartFinalState = fs->make<TH1D>( "EtaChgPartFinalState", "Eta of final state charged particles", 
                                         100, -5.0, 5.0 );
  fEtaNeuPartFinalState = fs->make<TH1D>( "EtaNeuPartFinalState", "Eta of final state neutral particles", 
                                         100, -5.0, 5.0 );
  return ;
  
}

void BasicGenTester::beginRun( const edm::Run& r, const edm::EventSetup& es )
{
   
   es.getData( fPDGTable ) ;
   
   return ;

}

void BasicGenTester::analyze( const Event& e, const EventSetup& )
{
  
   // here's an example of accessing GenEventInfoProduct
/*
   Handle< GenEventInfoProduct > GenInfoHandle;
   e.getByLabel( "generator", GenInfoHandle );
   double qScale = GenInfoHandle->qScale();
   double pthat = ( GenInfoHandle->hasBinningValues() ? 
                  (GenInfoHandle->binningValues())[0] : 0.0);
   cout << " qScale = " << qScale << " pthat = " << pthat << endl;
   double evt_weight1 = GenInfoHandle->weights()[0]; // this is "stanrd Py6 evt weight;
                                                     // corresponds to PYINT1/VINT(97)
   double evt_weight2 = GenInfoHandle->weights()[1]; // in case you run in CSA mode or otherwise
                                                     // use PYEVWT routine, this will be weight
						     // as returned by PYEVWT, i.e. PYINT1/VINT(99)
   //std::cout << " evt_weight1 = " << evt_weight1 << std::endl;
   //std::cout << " evt_weight2 = " << evt_weight2 << std::endl;
   double weight = GenInfoHandle->weight();
   //std::cout << " as returned by the weight() method, integrated event weight = " << weight << std::endl;
*/
  
   // here's an example of accessing particles in the event record (HepMCProduct)
   //
   Handle< HepMCProduct > EvtHandle ;
  
   // find initial (unsmeared, unfiltered,...) HepMCProduct
   //
   e.getByLabel("VtxSmeared", EvtHandle);
  
   const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
  
   double NChgPartFS = 0;
   double NNeuPartFS = 0; 
   for ( HepMC::GenEvent::particle_const_iterator part = Evt->particles_begin();
	 part != Evt->particles_end(); ++part ) 
   {
      
/*
      int pid = (*part)->pdg_id();
      if ( abs(pid) == 15 )
      {      
         std::cout << "found tau " << std::endl;
	 int stat = (*part)->status();
	 if ( (*part)->end_vertex() )
	 {
	    (*part)->end_vertex()->print();
	    std::cout << "done with tau end vertex " << std::endl;
	 }
	 std::cout << " end looking at tau" << std::endl;      
      }
*/      
      
      if ( (*part)->status() == 1 && !((*part)->end_vertex()) ) 
      {	  
      
         int PartID = (*part)->pdg_id();
	 
	 const HepPDT::ParticleData* 
             PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
         double charge = PData->charge();
         if ( fabs(charge) != 0. )
	 {
	    NChgPartFS++;
	    fPtChgPartFinalState->Fill( ((*part)->momentum()).perp() );
	    fEtaChgPartFinalState->Fill( ((*part)->momentum()).eta() );
	 }
	 else
	 {
	    NNeuPartFS++;
	    fPtNeuPartFinalState->Fill( ((*part)->momentum()).perp() );
	    fEtaNeuPartFinalState->Fill( ((*part)->momentum()).perp() );
	 }
      }
   }
  
   fNChgPartFinalState->Fill( NChgPartFS );
   fNNeuPartFinalState->Fill( NNeuPartFS );
   fNPartFinalState->Fill( (NChgPartFS+NNeuPartFS) );
   
   return ;
   
}

void BasicGenTester::endRun( const edm::Run& r, const edm::EventSetup& )
{

   return;

}


void BasicGenTester::endJob()
{
   
   return ;
}
 
DEFINE_FWK_MODULE(BasicGenTester);
