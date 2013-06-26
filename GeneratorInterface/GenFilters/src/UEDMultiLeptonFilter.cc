// -*- C++ -*-
//
// Package:    UEDMultiLeptonFilter
// Class:      UEDMultiLeptonFilter
// 
/**\class UEDMultiLeptonFilter UEDMultiLeptonFilter.cc UEDFilters/UEDMultiLeptonFilter/src/UEDMultiLeptonFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  
//         Created:  Sat Jul 10 10:32:40 BRT 2010
// $Id: UEDMultiLeptonFilter.cc,v 1.3 2013/05/23 14:30:27 gartung Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>

#include "DataFormats/Math/interface/LorentzVector.h"

//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TFile.h"

#include <map>

//
// class declaration
//

class UEDMultiLeptonFilter : public edm::EDFilter {
   public:
      explicit UEDMultiLeptonFilter(const edm::ParameterSet&);
      ~UEDMultiLeptonFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      bool isLepton(HepMC::GenVertex::particles_out_const_iterator part);      
      bool isLeptonPlus(HepMC::GenVertex::particles_out_const_iterator part);
      bool isLeptonMinus(HepMC::GenVertex::particles_out_const_iterator part);      
      void nLeptons(const std::vector<int>&, int& e,int& mu);
      void AllVetoedOff(bool inclusive_message);
      bool AcceptEvent();
      // ----------member data ---------------------------

      int UseFilter;

      int SSDiMuFilter;
      int SSDiMuVetoedFilter;

      int SSDiEFilter;
      int SSDiEVetoedFilter;

      int SSDiEMuFilter;
      int SSDiEMuVetoedFilter;

      int SSDiLepFilter;
      int SSDiLepVetoedFilter;

      int Vetoed3muFilter;
      int Vetoed2mu1eFilter;
      int Vetoed1mu2eFilter;
      int Vetoed3eFilter; 

      int Vetoed4muFilter;
      int Vetoed2mu2eFilter;
      int Vetoed4eFilter;

      int nFilteredEvents;
      int nProcessed;

      int nDileptons;

      int nDidileptons;
      int nDimumu;
      int nDiee;
      int nDiemu;

      int nSSdileptons;
      int nSSmumu;
      int nSSee;
      int nSSemu;

      int nTrileptons;
      int nTri3mu;
      int nTri2mu1e;
      int nTri1mu2e;
      int nTri3e;
      int nFourleptons; 
      int nFour4mu;
      int nFour4e;
      int nFour2mu2e; 
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
UEDMultiLeptonFilter::UEDMultiLeptonFilter(const edm::ParameterSet& iConfig) :
UseFilter(iConfig.getUntrackedParameter("UseFilter",0)),
SSDiMuFilter(iConfig.getUntrackedParameter("SSDiMuFilter",0)),
SSDiMuVetoedFilter(iConfig.getUntrackedParameter("SSDiMuVetoedFilter",0)),
SSDiEFilter(iConfig.getUntrackedParameter("SSDiEFilter",0)),
SSDiEVetoedFilter(iConfig.getUntrackedParameter("SSDiEVetoedFilter",0)),
SSDiEMuFilter(iConfig.getUntrackedParameter("SSDiEMuFilter",0)),
SSDiEMuVetoedFilter(iConfig.getUntrackedParameter("SSDiEMuVetoedFilter",0)),
SSDiLepFilter(iConfig.getUntrackedParameter("SSDiLepFilter",0)),
SSDiLepVetoedFilter(iConfig.getUntrackedParameter("SSDiLepVetoedFilter",0)),
Vetoed3muFilter(iConfig.getUntrackedParameter("Vetoed3muFilter",0)),
Vetoed2mu1eFilter(iConfig.getUntrackedParameter("Vetoed2mu1eFilter",0)),
Vetoed1mu2eFilter(iConfig.getUntrackedParameter("Vetoed1mu2eFilter",0)),
Vetoed3eFilter(iConfig.getUntrackedParameter("Vetoed3eFilter",0)),
Vetoed4muFilter(iConfig.getUntrackedParameter("Vetoed4muFilter",0)),
Vetoed2mu2eFilter(iConfig.getUntrackedParameter("Vetoed2mu2eFilter",0)),
Vetoed4eFilter(iConfig.getUntrackedParameter("Vetoed4eFilter",0))
{
   //now do what ever initialization is needed

   if(UseFilter==0){//std::cout << "************ No Filter ************" << std::endl;
   }else{
//     std::cout <<"Filters On" << std::endl;
     if(SSDiLepFilter+SSDiMuFilter+SSDiEFilter+SSDiEMuFilter>1){
//     std::cout << "Bad configuration: more than one non-vetoed filter is On" << std::endl;
//     std::cout << "You should use only one inclusive filter" << std::endl;
//     std::cout << "All events will pass: UseFilter=0"<< std::endl;
     UseFilter=0;
     }
    }

   if(UseFilter==1){ 
     if(SSDiLepFilter==1){
//       std::cout << "SSDiLepFilter: On " << std::endl;
       //To avoid multiple counting
       AllVetoedOff(true);
     }
     if(SSDiLepVetoedFilter==1){
       SSDiMuVetoedFilter=1;
       SSDiEMuVetoedFilter=1;   
       SSDiEVetoedFilter=1;
     }
     if(SSDiMuFilter==1){
//       std::cout <<"SSDiMuFilter: On" << std::endl;
       AllVetoedOff(true);
       SSDiMuVetoedFilter=1;
       Vetoed3muFilter=1;
       Vetoed4muFilter=1;
     }
     if(SSDiEFilter==1){
//       std::cout <<"SSDiEFilter: On" << std::endl;
       AllVetoedOff(true);
       SSDiEVetoedFilter=1;
       Vetoed3eFilter=1;
       Vetoed4eFilter=1;
     }
     if(SSDiEMuFilter==1){
//       std::cout <<"SSDiEMuFilter: On" << std::endl;       
       AllVetoedOff(true);
       SSDiEMuVetoedFilter=1;
       Vetoed2mu1eFilter=1;
       Vetoed1mu2eFilter=1;
       Vetoed2mu2eFilter=1; 
     } 
   }


   nFilteredEvents = 0;
   nProcessed = 0;

   nDileptons = 0;

   nDimumu = 0;
   nDiee = 0;
   nDiemu = 0;  

   nSSdileptons = 0;

   nSSmumu = 0;
   nSSee = 0;
   nSSemu = 0;

   nTrileptons = 0;

   nTri3mu = 0;
   nTri2mu1e = 0;
   nTri1mu2e = 0;
   nTri3e = 0;

   nFourleptons = 0;

   nFour4mu = 0;
   nFour4e = 0;
   nFour2mu2e = 0;

}

void 
UEDMultiLeptonFilter::AllVetoedOff(bool inclusive_message){

//      if(inclusive_message)std::cout << "All Vetoed Filters Off" << std::endl;

      SSDiMuVetoedFilter=0;
      SSDiEVetoedFilter=0;
      SSDiEMuVetoedFilter=0;
      SSDiLepVetoedFilter=0;

      Vetoed3muFilter=0;
      Vetoed2mu1eFilter=0;
      Vetoed1mu2eFilter=0;
      Vetoed3eFilter=0;

      Vetoed4muFilter=0;
      Vetoed2mu2eFilter=0;
      Vetoed4eFilter=0;

}

UEDMultiLeptonFilter::~UEDMultiLeptonFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

/*

   std::cout << "nProcessed: " << nProcessed << std::endl;
   std::cout << "nFilteredEvents: " << nFilteredEvents << std::endl;

   std::cout << "nDileptons: " << nDileptons << std::endl;

   std::cout << "nDimumu: " << nDimumu << std::endl;
   std::cout << "nDiemu: " << nDiemu << std::endl;
   std::cout << "nDiee: " << nDiee << std::endl;


   std::cout << "nSSdileptons: " << nSSdileptons << std::endl;

   std::cout << "nSSmumu: " << nSSmumu << std::endl;
   std::cout << "nSSemu: " << nSSemu << std::endl;
   std::cout << "nSSee: " << nSSee << std::endl;   

   std::cout << "nTrileptons: " << nTrileptons << std::endl;

   std::cout << "nTri3mu: " << nTri3mu << std::endl; 
   std::cout << "nTri2mu1e: " << nTri2mu1e << std::endl;
   std::cout << "nTri1mu2e: " << nTri1mu2e << std::endl;
   std::cout << "nTri3e: " << nTri3e <<std::endl;

   std::cout << "nFourleptons: " << nFourleptons << std::endl;
   std::cout << "nFour4mu: " << nFour4mu << std::endl;
   std::cout << "nFour4e: " << nFour4e << std::endl;
   std::cout << "nFour2mu2e: " << nFour2mu2e << std::endl;

*/

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
UEDMultiLeptonFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;

   std::map<std::string, int> accept; 

//   std::cout << "===================== New event ========================: " << nFilteredEvents  << std::endl;

   nProcessed++;
   if(UseFilter==0){
     nFilteredEvents++;
     //std::cout << "Event accepted" << std::endl;
     return true;
   }
   std::vector<int> leptonIDminus;
   std::vector<int> leptonIDplus;

   Handle<HepMCProduct> mcEventHandle;

   iEvent.getByLabel("generator",mcEventHandle);

   const HepMC::GenEvent* mcEvent = mcEventHandle->GetEvent() ;

   HepMC::GenEvent::particle_const_iterator i;
   for(i = mcEvent->particles_begin(); i!= mcEvent->particles_end(); i++){

   bool D_part = (fabs((*i)->pdg_id())-5100000>0 && fabs((*i)->pdg_id())-5100000<40);
   bool S_part = (fabs((*i)->pdg_id())-6100000>0 && fabs((*i)->pdg_id())-6100000<40 );
     if(D_part || S_part){

       //debug std::cout << "UED particle ID: " << (*i)->pdg_id() << " status: "<< (*i)->status()<< std::endl;
       HepMC::GenVertex* vertex = (*i)->end_vertex();
       if(vertex!=0){
       for(HepMC::GenVertex::particles_out_const_iterator part = vertex->particles_out_const_begin();
     part != vertex->particles_out_const_end(); part++ ){
          //debug std::cout << "   Outgoing particle id :"<< (*part)->pdg_id() << " status:  " << (*part)->status() << std::endl;
            if(isLepton(part)){
                if((*part)->status()==1){//debug std::cout << "Final State Lepton " << std::endl;
                }
                HepMC::GenVertex* lepton_vertex = (*part)->end_vertex();
                if(lepton_vertex!=0){
                for(HepMC::GenVertex::particles_out_const_iterator lepton_part = lepton_vertex->particles_out_const_begin();
     lepton_part != lepton_vertex->particles_out_const_end(); lepton_part++ ){
                        //debug std::cout << "      Part Id: "<< (*lepton_part)->pdg_id() << " status: "<< (*lepton_part)->status()<< std::endl;
                        if((*part)->pdg_id() == (*lepton_part)->pdg_id()){//debug std::cout << "         Part pt: " << (*part)->momentum().perp() << " mu 1: " << (*lepton_part)->momentum().perp() << std::endl;
                        }
                        //std::cout << "         Part px: " << (*part)->momentum().px() << " end_vtx px: " << (*mu_part)->momentum().px() << std::endl;
                        if((*part)->pdg_id() == (*lepton_part)->pdg_id() && (*lepton_part)->status()==1){
                            if(isLeptonPlus(lepton_part))leptonIDplus.push_back((*lepton_part)->pdg_id());
                            if(isLeptonMinus(lepton_part))leptonIDminus.push_back((*lepton_part)->pdg_id());
                          }
                   }
                 }
              }
          }
          }
      }
     }

        if((leptonIDplus.size()==1 && leptonIDminus.size()==1) || (leptonIDplus.size()==1 && leptonIDminus.size()==1)){
        //Opposite sign only
        nDileptons++; 
          int e=0;
          int mu=0;

          nLeptons(leptonIDplus,e,mu);
          nLeptons(leptonIDminus,e,mu);
          if(mu==2 && e==0)nDimumu++;
          if(mu==1 && e==1)nDiemu++;
          if(mu==0 && e==2)nDiee++;

        }


     if(leptonIDplus.size()>1 || leptonIDminus.size()>1){
//	std::cout << "SS dilepton" << std::endl;
        //Check dimuon inclusive filter
        nSSdileptons++;
        if(SSDiLepFilter==1)return AcceptEvent();
        //Vetoed decays
        if((leptonIDplus.size()==2 && leptonIDminus.size()==0) || (leptonIDplus.size()==0 && leptonIDminus.size()==2)){

          int e=0;
          int mu=0;      

          nLeptons(leptonIDplus,e,mu);
          if(mu==2 && e==0){
           nSSmumu++;
           if(SSDiMuVetoedFilter==1)return AcceptEvent();
          }
          if(mu==1 && e==1){
           nSSemu++;
           if(SSDiEMuVetoedFilter==1)return AcceptEvent();
          }
          if(mu==0 && e==2){
	   nSSee++;
	   if(SSDiEVetoedFilter==1)return AcceptEvent(); 
 	  }

          e=0;
          mu=0;

          nLeptons(leptonIDminus,e,mu);
          if(mu==2 && e==0){
           nSSmumu++;
           if(SSDiMuVetoedFilter==1)return AcceptEvent();           
          }
          if(mu==1 && e==1){
	   nSSemu++;
           if(SSDiEMuVetoedFilter==1)return AcceptEvent();
          }           
          if(mu==0 && e==2){
           nSSee++;
           if(SSDiEVetoedFilter==1)return AcceptEvent();
          }
        }        

     }

     if(leptonIDplus.size()+leptonIDminus.size()==4){
//        std::cout << "Four lepton" << std::endl;
        nFourleptons++;
        int e=0;
        int mu=0;
        nLeptons(leptonIDplus,e,mu);
        nLeptons(leptonIDminus,e,mu);
        if(mu==4 && e==0){
	 nFour4mu++;
         if(Vetoed4muFilter==1)return AcceptEvent();         
        }
        if(mu==2 && e==2){
         nFour2mu2e++;
         if(Vetoed2mu2eFilter==1)return AcceptEvent();
	}
        if(mu==0 && e==4){
	 nFour4e++;
         if(Vetoed4eFilter==1)return AcceptEvent();	 
	}

     }
     if(leptonIDplus.size()+leptonIDminus.size()==3){
//        std::cout << "Three lepton " << std::endl;
        nTrileptons++;
        int e=0;
        int mu=0;
        nLeptons(leptonIDplus,e,mu);
        nLeptons(leptonIDminus,e,mu);
        if(mu==3 && e==0){
	 nTri3mu++;
         if(Vetoed3muFilter==1)return AcceptEvent();	 
	}
        if(mu==2 && e==1){
         nTri2mu1e++;
         if(Vetoed2mu1eFilter==1)return AcceptEvent();
        }
        if(mu==1 && e==2){
	 nTri1mu2e++;
         if(Vetoed1mu2eFilter==1)return AcceptEvent();
	}
        if(mu==0 && e==3){
  	 nTri3e++;
         if(Vetoed3eFilter==1)return AcceptEvent();
	}
     }




#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
   return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
UEDMultiLeptonFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
UEDMultiLeptonFilter::endJob() {
}

bool 
UEDMultiLeptonFilter::isLepton(HepMC::GenVertex::particles_out_const_iterator part){

  return (fabs((*part)->pdg_id())==13 || fabs((*part)->pdg_id())==11);

} 

bool 
UEDMultiLeptonFilter::isLeptonPlus(HepMC::GenVertex::particles_out_const_iterator part){
  return ((*part)->pdg_id()==-13 || (*part)->pdg_id()==-11);  
}

bool 
UEDMultiLeptonFilter::isLeptonMinus(HepMC::GenVertex::particles_out_const_iterator part){
  return ((*part)->pdg_id()==13 || (*part)->pdg_id()==11);
}

void 
UEDMultiLeptonFilter::nLeptons(const std::vector<int>& leptons_id, int& e,int& mu){
  int nentries = (int)leptons_id.size();
  for(int i=0; i<nentries; i++){
   if(abs(leptons_id.at(i))==11)e++;
   if(abs(leptons_id.at(i))==13)mu++;
  }

//  std::cout <<"mu "<< mu << std::endl;
//  std::cout <<"e "<< e <<std::endl; 
}

bool
UEDMultiLeptonFilter::AcceptEvent(){
     nFilteredEvents++;
//     std::cout << "Event accepted" << std::endl;
     return true;
}
//define this as a plug-in
DEFINE_FWK_MODULE(UEDMultiLeptonFilter);
