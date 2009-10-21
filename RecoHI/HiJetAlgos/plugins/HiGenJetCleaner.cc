// -*- C++ -*-
//
// Package:    HiGenJetCleaner
// Class:      HiGenJetCleaner
// 
/**\class HiGenJetCleaner HiGenJetCleaner.cc yetkin/HiGenJetCleaner/src/HiGenJetCleaner.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Jul 21 04:26:01 EDT 2009
// $Id: HiGenJetCleaner.cc,v 1.4 2009/08/03 13:44:19 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace edm;


//
// class decleration
//

class HiGenJetCleaner : public edm::EDProducer {
   public:
      explicit HiGenJetCleaner(const edm::ParameterSet&);
      ~HiGenJetCleaner();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

   string jetSrc_;
   double deltaR_;
   double ptCut_;
   bool makeNew_;
   bool fillDummy_;

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

HiGenJetCleaner::HiGenJetCleaner(const edm::ParameterSet& iConfig) :
   jetSrc_(iConfig.getUntrackedParameter<string>( "src","iterativeCone5HiGenJets")),
   deltaR_(iConfig.getUntrackedParameter<double>("deltaR",0.125)),
   ptCut_(iConfig.getUntrackedParameter<double>("ptCut",20)),
   makeNew_(iConfig.getUntrackedParameter<bool>("createNewCollection",true)),
   fillDummy_(iConfig.getUntrackedParameter<bool>("fillDummyEntries",true))
{
   std::string alias = jetSrc_;

   if(makeNew_)
      produces<reco::GenJetCollection>().setBranchAlias (alias);
   else
      produces<reco::GenJetRefVector>().setBranchAlias (alias);
}

HiGenJetCleaner::~HiGenJetCleaner()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HiGenJetCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   auto_ptr<GenJetCollection> jets;
   auto_ptr<GenJetRefVector> jetrefs;

   if(makeNew_)
      jets = auto_ptr<GenJetCollection>(new GenJetCollection);
   else
      jetrefs = auto_ptr<GenJetRefVector>(new GenJetRefVector);
   
   edm::Handle<reco::GenJetCollection> genjets;
   iEvent.getByLabel(jetSrc_,genjets);

   int jetsize = genjets->size();

   vector<int> selection;
   for(int ijet = 0; ijet < jetsize; ++ijet){
      selection.push_back(-1);
   }

   vector<int> selectedIndices;
   vector<int> removedIndices;

   for(int ijet = 0; ijet < jetsize; ++ijet){

      const reco::GenJet* jet1 = &((*genjets)[ijet]);

      if(selection[ijet] == -1){
	 selection[ijet] = 1;
	 for(int ijet2 = 0; ijet2 < jetsize; ++ijet2){

	    if(ijet2 == ijet) continue;
	    
	    const reco::GenJet* jet2 = &((*genjets)[ijet2]);
	    
	    if(Geom::deltaR(jet1->momentum(),jet2->momentum()) < deltaR_){
	       if(jet1->et() < jet2->et()){
		  selection[ijet] = 0;
		  removedIndices.push_back(ijet);
		  break;
	       }else{
		  selection[ijet2] = 0;
		  removedIndices.push_back(ijet2);
	       }
	    }
	 }
      }
      
      double etjet = ((*genjets)[ijet]).et();
      
      if(selection[ijet] == 1 && etjet > ptCut_){ 
	 selectedIndices.push_back(ijet);
	 GenJetRef ref(genjets,ijet);
	 
	 if(makeNew_)
	    jets->push_back(*jet1);
	 else
	    jetrefs->push_back(ref);
	 
      }else if(fillDummy_){
	 reco::GenJet dummy(math::XYZTLorentzVector(-0.19,-0.19,-499,99),math::XYZPoint(-99,-99,-99),reco::GenJet::Specific());
	 if(makeNew_)
	    jets->push_back(dummy);
      }
      
      
      
   }

   if(makeNew_)
      iEvent.put(jets);
   else
      iEvent.put(jetrefs);

 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HiGenJetCleaner::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiGenJetCleaner::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiGenJetCleaner);
