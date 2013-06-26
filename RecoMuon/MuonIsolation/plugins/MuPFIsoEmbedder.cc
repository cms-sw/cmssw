// -*- C++ -*-
//
// Package:    MuPFIsoEmbedder
// Class:      MuPFIsoEmbedder
// 
/**\class MuPFIsoEmbedder MuPFIsoEmbedder.cc RecoMuon/MuPFIsoEmbedder/src/MuPFIsoEmbedder.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michail Bachtis,32 3-B16,+41227675567,
//         Created:  Thu Jun  9 01:36:17 CEST 2011
// $Id: MuPFIsoEmbedder.cc,v 1.3 2013/02/25 21:46:26 chrjones Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/MuonIsolation/interface/MuPFIsoHelper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//
// class declaration
//

class MuPFIsoEmbedder : public edm::EDProducer {
   public:
      explicit MuPFIsoEmbedder(const edm::ParameterSet&);
      ~MuPFIsoEmbedder();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      

      // ----------member data ---------------------------
  edm::InputTag muons_;

  MuPFIsoHelper *helper_;

};




//
MuPFIsoEmbedder::MuPFIsoEmbedder(const edm::ParameterSet& iConfig):
  muons_(iConfig.getParameter<edm::InputTag>("src"))
{

  //decide what to read
    //Define a map between the isolation and the PSet for the PFHelper
    std::map<std::string,edm::ParameterSet> psetMap;

    //First declare what isolation you are going to read
    std::vector<std::string> isolationLabels;
    isolationLabels.push_back("pfIsolationR03");
    isolationLabels.push_back("pfIsoMeanDRProfileR03");
    isolationLabels.push_back("pfIsoSumDRProfileR03");
    isolationLabels.push_back("pfIsolationR04");
    isolationLabels.push_back("pfIsoMeanDRProfileR04");
    isolationLabels.push_back("pfIsoSumDRProfileR04");

    //Fill the label,pet map and initialize MuPFIsoHelper
    for( std::vector<std::string>::const_iterator label = isolationLabels.begin();label != isolationLabels.end();++label)
      psetMap[*label] =iConfig.getParameter<edm::ParameterSet >(*label); 

    helper_ = new MuPFIsoHelper(psetMap);

  produces<reco::MuonCollection>();
}


MuPFIsoEmbedder::~MuPFIsoEmbedder()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuPFIsoEmbedder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;


   helper_->beginEvent(iEvent);
  
   edm::Handle<reco::MuonCollection > muons;
   iEvent.getByLabel(muons_,muons);


   std::auto_ptr<MuonCollection> out(new MuonCollection);

   for(unsigned int i=0;i<muons->size();++i) {
     MuonRef muonRef(muons,i);
     Muon muon = muons->at(i);
     helper_->embedPFIsolation(muon,muonRef);
     out->push_back(muon);
   }

   iEvent.put(out);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MuPFIsoEmbedder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuPFIsoEmbedder);
