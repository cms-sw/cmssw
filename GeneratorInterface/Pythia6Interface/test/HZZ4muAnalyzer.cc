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

class HZZ4muAnalyzer : public edm::EDAnalyzer
{

   public:
   
      //
      explicit HZZ4muAnalyzer( const edm::ParameterSet& ) ;
      virtual ~HZZ4muAnalyzer() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
      virtual void beginJob() override;

   private:
   
     TH1D*       fHist2muMass ;
     TH1D*       fHist4muMass ;
     TH1D*       fHistZZMass ;
     
}; 

using namespace edm;
using namespace std;

HZZ4muAnalyzer::HZZ4muAnalyzer( const ParameterSet& pset )
  : fHist2muMass(0), fHist4muMass(0), fHistZZMass(0)
{
// actually, pset is NOT in use - we keep it here just for illustratory putposes
}

void HZZ4muAnalyzer::beginJob()
{
  
  Service<TFileService> fs;
  fHist2muMass = fs->make<TH1D>(  "Hist2muMass", "2-mu inv. mass", 100,  60., 120. ) ;
  fHist4muMass = fs->make<TH1D>(  "Hist4muMass", "4-mu inv. mass", 100, 170., 210. ) ;
  fHistZZMass  = fs->make<TH1D>(  "HistZZMass",  "ZZ inv. mass",   100, 170., 210. ) ;    
    
  return ;
  
}

void HZZ4muAnalyzer::analyze( const Event& e, const EventSetup& )
{
  
  // here's an example of accessing GenEventInfoProduct
  Handle< GenEventInfoProduct > GenInfoHandle;
  e.getByLabel( "generator", GenInfoHandle );
  double qScale = GenInfoHandle->qScale();
  double pthat = ( GenInfoHandle->hasBinningValues() ? 
                  (GenInfoHandle->binningValues())[0] : 0.0);
  cout << " qScale = " << qScale << " pthat = " << pthat << endl;
  //
  // this (commented out) code below just exemplifies how to access certain info 
  //
  //double evt_weight1 = GenInfoHandle->weights()[0]; // this is "stanrd Py6 evt weight;
                                                    // corresponds to PYINT1/VINT(97)
  //double evt_weight2 = GenInfoHandle->weights()[1]; // in case you run in CSA mode or otherwise
                                                    // use PYEVWT routine, this will be weight
						    // as returned by PYEVWT, i.e. PYINT1/VINT(99)
  //std::cout << " evt_weight1 = " << evt_weight1 << std::endl;
  //std::cout << " evt_weight2 = " << evt_weight2 << std::endl;
  //double weight = GenInfoHandle->weight();
  //std::cout << " as returned by the weight() method, integrated event weight = " << weight << std::endl;
  
  // here's an example of accessing particles in the event record (HepMCProduct)
  //
  Handle< HepMCProduct > EvtHandle ;
  
  // find initial (unsmeared, unfiltered,...) HepMCProduct
  //
  e.getByLabel( "generator", EvtHandle ) ;
  
  const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
  
  // this a pointer - and not an array/vector/... 
  // because this example explicitely assumes
  // that there one and only Higgs in the record
  //
  HepMC::GenVertex* HiggsDecVtx = 0 ;
  
  // find the 1st vertex with outgoing Higgs 
  // and get Higgs decay vertex from there;
  //
  // in principal, one can look for the vertex 
  // with incoming Higgs as well...
  //
  for ( HepMC::GenEvent::vertex_const_iterator
          vit=Evt->vertices_begin(); vit!=Evt->vertices_end(); vit++ )
  {
      for ( HepMC::GenVertex::particles_out_const_iterator
              pout=(*vit)->particles_out_const_begin();
            pout!=(*vit)->particles_out_const_end(); pout++ )
      {
          if ( (*pout)->pdg_id() == 25 && (*pout)->status() == 2 ) 
          {	    
              if ( (*pout)->end_vertex() != 0 )
              {
                  HiggsDecVtx = (*pout)->end_vertex() ;
                  break ;
              }
          }
      }
      if ( HiggsDecVtx != 0 )
      {
          break ; // break the initial loop over vertices
      }
  }
  
  if ( HiggsDecVtx == 0 ) 
  {
      cout << " There is NO Higgs in this event ! " << endl ;
      return ;
  }
  
  if ( e.id().event() == 1 )
  {
      cout << " " << endl ;
      cout << " We do some example printouts in the event 1 " << endl ;
      cout << " Higgs decay found at the vertex " << HiggsDecVtx->barcode() <<" (barcode)" << endl ;

      vector<HepMC::GenParticle*> HiggsChildren;

      for ( HepMC::GenVertex::particles_out_const_iterator H0in = 
              HiggsDecVtx->particles_out_const_begin(); 
	      H0in != HiggsDecVtx->particles_out_const_end(); 
	      H0in++ ) 
      { 
         HiggsChildren.push_back(*H0in);
      }
      cout << " Number of Higgs (immediate) children = " << HiggsChildren.size() << endl ;
      for (unsigned int ic=0; ic<HiggsChildren.size(); ic++ )
      {
          HiggsChildren[ic]->print() ;   
      }
  }
   
   // select and store stable descendants of the Higgs
   //   
   vector<HepMC::GenParticle*> StableHiggsDesc ;
   
   if ( e.id().event() == 1 )
      cout << " Now let us list all descendents of the Higgs" << endl ;
   for ( HepMC::GenVertex::particle_iterator
         des=HiggsDecVtx->particles_begin(HepMC::descendants);
	 des!=HiggsDecVtx->particles_end(HepMC::descendants); des++ )
   {
      if ( e.id().event() == 1 ) (*des)->print() ;
      if ( (*des)->status() == 1 ) StableHiggsDesc.push_back(*des) ;
   }
   
   HepMC::FourVector Mom2part ;
   double            XMass2part = 0.;
   double            XMass4part = 0.;
   double            XMass2pairs  = 0.;
   vector< HepMC::FourVector > Mom2partCont ;
    
   // browse the array of stable descendants
   // and do 2-mu inv.mass
   //
   for ( unsigned int i=0; i<StableHiggsDesc.size(); i++ )
     {
       // skip other than mu
       //
       if ( abs(StableHiggsDesc[i]->pdg_id()) != 13 ) continue ; 
       
       for ( unsigned int j=i+1; j<StableHiggsDesc.size(); j++ )
         {
           // skip other than mu
           //
           if ( abs(StableHiggsDesc[j]->pdg_id()) != 13 ) continue ;
           //
           // skip same charge combo's
           //
           if ( (StableHiggsDesc[i]->pdg_id()*StableHiggsDesc[j]->pdg_id()) > 0 ) 
             continue ;
           //
           // OK, opposite charges, do the job
           //
	   Mom2part = HepMC::FourVector((StableHiggsDesc[i]->momentum().px()+StableHiggsDesc[j]->momentum().px()),
	                                (StableHiggsDesc[i]->momentum().py()+StableHiggsDesc[j]->momentum().py()),
					(StableHiggsDesc[i]->momentum().pz()+StableHiggsDesc[j]->momentum().pz()),
					(StableHiggsDesc[i]->momentum().e()+StableHiggsDesc[j]->momentum().e())) ;
	   
           XMass2part = Mom2part.m() ;
           fHist2muMass->Fill( XMass2part ) ;
           //cout << " counters : " << StableHiggsDesc[i]->barcode() << " " 
           //                       << StableHiggsDesc[j]->barcode() 
           //			<< " -> 2-part mass = " << XMass2part << endl ;
           //
           // store if 2-part. inv. mass fits into (roughly) Z-mass interval 
           //
           if ( XMass2part > 80. && XMass2part < 100. )
           {
               Mom2partCont.push_back(Mom2part) ;
           } 
         }
     }
   
   // make 4-part inv.mass
   //
   
   double px4, py4, pz4, e4;
   px4=py4=pz4=e4=0. ;
   if ( StableHiggsDesc.size() == 4 )
   {
       for ( unsigned int i=0; i<StableHiggsDesc.size(); i++ )
       {
          px4 += StableHiggsDesc[i]->momentum().px();
          py4 += StableHiggsDesc[i]->momentum().py();
          pz4 += StableHiggsDesc[i]->momentum().pz();
          e4  += StableHiggsDesc[i]->momentum().e();
       }
       XMass4part = HepMC::FourVector(px4,py4,pz4,e4).m() ;
       fHist4muMass->Fill( XMass4part ) ;
   }
   //cout << " 4-part inv. mass = " << XMass4part << endl ;
   
   // make 2-pairs (ZZ) inv.mass
   //
   //cout << " selected Z-candidates in this event : " << Mom2partCont.size() << endl ;
   for ( unsigned int i=0; i<Mom2partCont.size(); i++ )
   {
      for ( unsigned int j=i+1; j<Mom2partCont.size(); j++ )
      {
         // Mom2pairs = Mom2partCont[i] + Mom2partCont[j] ;
	 XMass2pairs = HepMC::FourVector((Mom2partCont[i].px()+Mom2partCont[j].px()),
	                                 (Mom2partCont[i].py()+Mom2partCont[j].py()),
					 (Mom2partCont[i].pz()+Mom2partCont[j].pz()),
					 (Mom2partCont[i].e() +Mom2partCont[j].e())).m() ;
	 fHistZZMass->Fill( XMass2pairs ) ;
         //cout << " 2-pairs (ZZ) inv. mass = " << XMass2pairs << endl ;
      }
   }
   
   return ;
   
}

typedef HZZ4muAnalyzer HZZ4muExampleAnalyzer;
DEFINE_FWK_MODULE(HZZ4muExampleAnalyzer);
