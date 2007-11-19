// -*- C++ -*-
//
// Package:    RecoCentrality
// Class:      RecoCentrality
// 
/**\class RecoCentrality RecoCentrality.cc CentralityAnalysis/RecoCentrality/src/RecoCentrality.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Jun 14 06:49:13 EDT 2007
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"

using namespace std;
using namespace reco;

//
// class decleration
//

class RecoCentrality : public edm::EDProducer {
   public:
      explicit RecoCentrality(const edm::ParameterSet&);
      ~RecoCentrality();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  string src_;
edm::ESHandle<CentralityTable> input;

};

//
// constructors and destructor
//
RecoCentrality::RecoCentrality(const edm::ParameterSet& iConfig)
{
  src_ = iConfig.getUntrackedParameter<string>("signal","hfreco");
   produces<Centrality>( "hfBasedCent" );
}


RecoCentrality::~RecoCentrality()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
RecoCentrality::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   int centBin    = -1;
   double eHF     =  0;        // Dummy variable for computing total HF energy 
   double npart   =  0;
   double npsigma =  0;
   double ncoll   =  0;
   double ncsigma =  0;
   double b       =  0;
   double bsigma  =  0;

   
   Handle<HFRecHitCollection> hits;
   iEvent.getByLabel(src_,hits);
 
   // Summing the HF energy 
   for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
      const HFRecHit & rechit = (*hits)[ ihit ];
      eHF = eHF + rechit.energy();
      }

   // See which bin the total energy corresponds to:
   for(int ie = 1; ie<20; ++ie){
      if(eHF>(*input).m_table[ie].hf_low_cut && eHF<(*input).m_table[ie-1].hf_low_cut){
        centBin = ie;
        }
      }
   if(eHF>(*input).m_table[0].hf_low_cut) centBin = 0;

   //The values for the event are assigned here:
   npart = (*input).m_table[centBin].n_part_mean;
   npsigma = (*input).m_table[centBin].n_part_var;
   ncoll = (*input).m_table[centBin].n_coll_mean;
   ncsigma = (*input).m_table[centBin].n_coll_var;
   b = (*input).m_table[centBin].b_mean;
   bsigma = (*input).m_table[centBin].b_var;

   std::auto_ptr<Centrality> myCent(new Centrality(eHF, centBin, npart, npsigma, ncoll, ncsigma, b, bsigma));
   iEvent.put(myCent, "hfBasedCent" );
   
}

// ------------ method called once each job just before starting event loop  ------------
void 
RecoCentrality::beginJob(const edm::EventSetup& iSetup)
{
iSetup.get<HeavyIonRcd>().get(input);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RecoCentrality::endJob() {

}

//define this as a plug-in
DEFINE_FWK_MODULE(RecoCentrality);
