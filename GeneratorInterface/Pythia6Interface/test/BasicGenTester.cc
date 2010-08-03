#include <iostream>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"

class BasicGenTester : public edm::EDAnalyzer
{

   public:
   
      //
      explicit BasicGenTester( const edm::ParameterSet& ) ;
      virtual ~BasicGenTester() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob() ;
      virtual void endRun( const edm::Run&, const edm::EventSetup& ) ;
      virtual void endJob() ;

   private:
   
     TH1D*       fNPartFinalState ;
     TH1D*       fPtPartFinalState ;
     int         fNPart;
     double      fPtMin;
     double      fPtMax;
     
}; 

using namespace edm;
using namespace std;

BasicGenTester::BasicGenTester( const ParameterSet& pset )
  : fNPartFinalState(0), fPtPartFinalState(0)
{

   fNPart = pset.getUntrackedParameter<int>( "NPartForHisto", 500 );
   fPtMin = pset.getUntrackedParameter<double>( "PtMinForHisto",  0. );
   fPtMax = pset.getUntrackedParameter<double>( "PtMaxForHisto", 25. );   

}

void BasicGenTester::beginJob()
{
  
  Service<TFileService> fs;
  fNPartFinalState = fs->make<TH1D>(  "NPartFinalState",  "Number of final state particles", fNPart,  0., (double)fNPart );
  fPtPartFinalState = fs->make<TH1D>( "PtPartFinalState", "Pt of final state, particles", 500, fPtMin, fPtMax );
    
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
   e.getByLabel( "generator", EvtHandle ) ;
  
   const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
  
   double NPartFS = 0; 
   for ( HepMC::GenEvent::particle_const_iterator part = Evt->particles_begin();
	 part != Evt->particles_end(); ++part ) 
   {
      if ( (*part)->status() == 1 && !((*part)->end_vertex()) ) 
      {
         NPartFS++;
	 fPtPartFinalState->Fill( ((*part)->momentum()).perp() );
      }
   }
  
   fNPartFinalState->Fill( NPartFS );
   
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
