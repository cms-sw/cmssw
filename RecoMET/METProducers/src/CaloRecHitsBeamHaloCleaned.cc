// -*- C++ -*-
//
// Package:    RecoMET/METProducers
// Class:      CaloRecHitsBeamHaloCleaned
// 
/**\class CaloRecHitsBeamHaloCleaned CaloRecHitsBeamHaloCleaned.cc RecoMET/METProducers/plugins/CaloRecHitsBeamHaloCleaned.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Laurent
//         Created:  Tue, 09 Feb 2016 13:09:37 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include <vector>
#include <iostream>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/METReco/interface/GlobalHaloData.h"
#include "RecoMET/METAlgorithms/interface/GlobalHaloAlgo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

using namespace reco;

class CaloRecHitsBeamHaloCleaned : public edm::stream::EDProducer<> {
public:
  explicit CaloRecHitsBeamHaloCleaned(const edm::ParameterSet&);
  ~CaloRecHitsBeamHaloCleaned();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
 
  edm::EDGetTokenT<EcalRecHitCollection> ecalebhits_token;
  edm::EDGetTokenT<EcalRecHitCollection> ecaleehits_token;
  edm::EDGetTokenT<HBHERecHitCollection> hbhehits_token; 
  edm::EDGetTokenT<GlobalHaloData> globalhalo_token;
  
  //Input tags
  edm::InputTag it_EBRecHits;
  edm::InputTag it_EERecHits;
  edm::InputTag it_HBHERecHits;
  edm::InputTag it_GlobalHaloData;

  
  bool ishlt;
};

//
// constructors and destructor
//
CaloRecHitsBeamHaloCleaned::CaloRecHitsBeamHaloCleaned(const edm::ParameterSet& iConfig)
{

  ishlt = iConfig.getParameter< bool> ("IsHLT");



  produces<EcalRecHitCollection>("EcalRecHitsEB");
  produces<EcalRecHitCollection>("EcalRecHitsEE");
  produces<HBHERecHitCollection>();


  it_EBRecHits = iConfig.getParameter<edm::InputTag>("EBRecHitsLabel");
  it_EERecHits = iConfig.getParameter<edm::InputTag>("EERecHitsLabel");
  it_HBHERecHits = iConfig.getParameter<edm::InputTag>("HBHERecHitsLabel");
  it_GlobalHaloData = iConfig.getParameter<edm::InputTag>("GlobalHaloDataLabel");


  ecalebhits_token= consumes<EcalRecHitCollection>(it_EBRecHits);
  ecaleehits_token= consumes<EcalRecHitCollection>(it_EERecHits);
  hbhehits_token= consumes<HBHERecHitCollection>(it_HBHERecHits);

  globalhalo_token=consumes<GlobalHaloData>(it_GlobalHaloData);
}


CaloRecHitsBeamHaloCleaned::~CaloRecHitsBeamHaloCleaned()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CaloRecHitsBeamHaloCleaned::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco; 
   using namespace std;


   Handle<EcalRecHitCollection> ebrhitsuncleaned;
   iEvent.getByToken(ecalebhits_token, ebrhitsuncleaned );

   Handle<EcalRecHitCollection> eerhitsuncleaned;
   iEvent.getByToken(ecaleehits_token, eerhitsuncleaned );

   Handle<HBHERecHitCollection> hbherhitsuncleaned;
   iEvent.getByToken(hbhehits_token, hbherhitsuncleaned );

   // Get GlobalHaloData
   edm::Handle<reco::GlobalHaloData> TheGlobalHaloData;
   iEvent.getByToken(globalhalo_token, TheGlobalHaloData);


   
   const GlobalHaloData TheSummaryHalo = (*TheGlobalHaloData );
   

   //Cleaning of the various rechits collections:

   //  EcalRecHit EB
   auto_ptr<EcalRecHitCollection> ebrhitscleaned(new EcalRecHitCollection()); 
   for(unsigned int i = 0;  i < ebrhitsuncleaned->size(); i++){
     const EcalRecHit & rhit = (*ebrhitsuncleaned)[i];
     bool isclean(true);
     edm::RefVector<EcalRecHitCollection> refbeamhalorechits =  TheSummaryHalo.GetEBRechits();
     for(unsigned int j = 0; j <refbeamhalorechits.size() ; j++){
       const EcalRecHit &rhitbeamhalo = *(refbeamhalorechits)[j];
       if( rhit.detid() == rhitbeamhalo.detid() ) { 
	 isclean  = false;
	 break;
       }
     }
     if(isclean) ebrhitscleaned->push_back(rhit);
   }
   
   //  EcalRecHit EE
   auto_ptr<EcalRecHitCollection> eerhitscleaned(new EcalRecHitCollection()); 
   for(unsigned int i = 0;  i < eerhitsuncleaned->size(); i++){
     const EcalRecHit & rhit = (*eerhitsuncleaned)[i];
     bool isclean(true);
     edm::RefVector<EcalRecHitCollection> refbeamhalorechits =  TheSummaryHalo.GetEERechits();
     for(unsigned int j = 0; j <refbeamhalorechits.size() ; j++){
       const EcalRecHit &rhitbeamhalo = *(refbeamhalorechits)[j];
       if( rhit.detid() == rhitbeamhalo.detid() ) { 
	 isclean  = false;
	 break;
       }
     }
     if(isclean) eerhitscleaned->push_back(rhit);
   }

   //  HBHERecHit
   auto_ptr<HBHERecHitCollection> hbherhitscleaned(new HBHERecHitCollection()); 
   for(unsigned int i = 0;  i < hbherhitsuncleaned->size(); i++){
     const HBHERecHit & rhit = (*hbherhitsuncleaned)[i];
     bool isclean(true);
     edm::RefVector<HBHERecHitCollection> refbeamhalorechits =  TheSummaryHalo.GetHBHERechits();
     for(unsigned int j = 0; j <refbeamhalorechits.size() ; j++){
       const HBHERecHit &rhitbeamhalo = *(refbeamhalorechits)[j];
       if( rhit.detid() == rhitbeamhalo.detid() ) { 
	 isclean  = false;
	 break;
       }
     }  
     if(isclean) hbherhitscleaned->push_back(rhit);
   }



   iEvent.put(ebrhitscleaned,"EcalRecHitsEB");
   iEvent.put(eerhitscleaned,"EcalRecHitsEE");
   iEvent.put(hbherhitscleaned);

}

 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CaloRecHitsBeamHaloCleaned::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloRecHitsBeamHaloCleaned);
