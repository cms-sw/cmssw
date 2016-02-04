/*******************************************************
*
*   Original Author:  Alexander Proskuryakov
*           Created:  Sat Aug  1 10:42:50 CEST 2009
*
*       Modified by:  Sheila Amaral
* Last modification:  Thu Aug 13 09:46:26 CEST 2009
* 
* Allows events which have at least 2 highest ET jets,
* at generator level, with deltaEta between jets higher
* than 3.5
*
*******************************************************/

#include "GeneratorInterface/GenFilters/interface/JGJFilter.h"

using namespace edm;
using namespace reco;
using namespace math;

JGJFilter::JGJFilter(const edm::ParameterSet& iConfig)
{
   //Number of events and Number Accepted
   nEvents = 0;
   nAccepted = 0;
}


JGJFilter::~JGJFilter()
{
std::cout << "Total number of tested events = " << nEvents << std::endl;
std::cout << "Total number of accepted events = " << nAccepted << std::endl;
std::cout << "nEfficiency = " << ((double)nAccepted)/((double)nEvents) << std::endl; 
}

bool JGJFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   nEvents++;
   bool accepted = false;

   using namespace edm;
   using namespace reco;
   using namespace std;

   //KT4 jets ***************************************************************************
   Handle<reco::GenJetCollection> o4;
   iEvent.getByLabel("kt4GenJetsjgj",o4);

   if( o4.isValid()) {
 
     float jet_pt[3];
     float jet_eta[3];

     int njets(0);
     for( GenJetCollection::const_iterator jet = o4->begin(); jet != o4->end(); ++jet ) {
       if(njets<4) {
        jet_pt[njets]=jet->pt();
        jet_eta[njets]=jet->eta();
       }
       njets++;
     }

     if(njets>1 && fabs(jet_eta[0]-jet_eta[1])>3.5) accepted = true;

     if(njets>2) {
       if((jet_pt[2]/jet_pt[1])>0.8) {
	   if(fabs(jet_eta[0]-jet_eta[2])>3.5) accepted = true;
	   if(fabs(jet_eta[1]-jet_eta[2])>3.5) accepted = true;
       }
     }
   } //valid

  
   //KT6 jets ***************************************************************************
   Handle<reco::GenJetCollection> o6;
   iEvent.getByLabel("kt6GenJetsjgj",o6);

   if( o6.isValid()) {
 
     float jet_pt[3];
     float jet_eta[3];

     int njets(0);
     for( GenJetCollection::const_iterator jet = o6->begin(); jet != o6->end(); ++jet ) {
       if(njets<4) {
        jet_pt[njets]=jet->pt();
        jet_eta[njets]=jet->eta();
       }
       njets++;
     }

     if(njets>1 && fabs(jet_eta[0]-jet_eta[1])>3.5) accepted = true;

     if(njets>2) {
       if((jet_pt[2]/jet_pt[1])>0.8) {
	   if(fabs(jet_eta[0]-jet_eta[2])>3.5) accepted = true;
	   if(fabs(jet_eta[1]-jet_eta[2])>3.5) accepted = true;
       }
     }
   } //valid
  

  //SC5 jets ***************************************************************************
   Handle<reco::GenJetCollection> oo;
   iEvent.getByLabel("sisCone5GenJetsjgj",oo);

   if( oo.isValid()) {
 
     float jet_pt[3];
     float jet_eta[3];

     int njets(0);
     for( GenJetCollection::const_iterator jet = oo->begin(); jet != oo->end(); ++jet ) {
       if(njets<4) {
        jet_pt[njets]=jet->pt();
        jet_eta[njets]=jet->eta();
       }
       njets++;
     }

     if(njets>1 && fabs(jet_eta[0]-jet_eta[1])>3.5) accepted = true;



     if(njets>2) {
       if((jet_pt[2]/jet_pt[1])>0.8) {
	   if(fabs(jet_eta[0]-jet_eta[2])>3.5) accepted = true;
	   if(fabs(jet_eta[1]-jet_eta[2])>3.5) accepted = true;
       }
     }
   } //valid


   if ( accepted ){
	nAccepted++;
        return true;
   }
   else return false;

}

void 
JGJFilter::beginJob()
{
}


void 
JGJFilter::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(JGJFilter);
