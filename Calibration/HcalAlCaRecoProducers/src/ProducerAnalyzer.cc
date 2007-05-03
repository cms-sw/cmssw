// system include files

// user include files

#include "Calibration/HcalAlCaRecoProducers/src/ProducerAnalyzer.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

/*
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
*/

using namespace std;

namespace cms
{

//
// constructors and destructor
//
ProducerAnalyzer::ProducerAnalyzer(const edm::ParameterSet& iConfig)
{
  // get name of output file with histogramms  
   
   nameProd_ = iConfig.getUntrackedParameter<std::string>("nameProd");
   jetCalo_ = iConfig.getUntrackedParameter<std::string>("jetCalo");
   gammaClus_ = iConfig.getUntrackedParameter<std::string>("gammaClus");
   ecalInput_=iConfig.getUntrackedParameter<std::string>("ecalInput");
   hbheInput_ = iConfig.getUntrackedParameter<std::string>("hbheInput");
   hoInput_ = iConfig.getUntrackedParameter<std::string>("hoInput");
   hfInput_ = iConfig.getUntrackedParameter<std::string>("hfInput");
   egammaJetTracks_ = iConfig.getUntrackedParameter<std::string>("egammaJetTrack"); 

}

ProducerAnalyzer::~ProducerAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void ProducerAnalyzer::beginJob( const edm::EventSetup& iSetup)
{
}

void ProducerAnalyzer::endJob()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ProducerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  std::vector<Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
  for( std::vector<Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(nameProd_,hbheInput_, hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());
   edm::Handle<HORecHitCollection> ho;
   iEvent.getByLabel(nameProd_,hoInput_, ho);
   const HORecHitCollection Hitho = *(ho.product());
   edm::Handle<HFRecHitCollection> hf;
   iEvent.getByLabel(nameProd_,hfInput_, hf);
   const HFRecHitCollection Hithf = *(hf.product());
   edm::Handle<EcalRecHitCollection> ecal;
   iEvent.getByLabel(nameProd_,ecalInput_, ecal);
   edm::Handle<reco::CaloJetCollection> jets;
   iEvent.getByLabel(nameProd_,jetCalo_, jets);
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(nameProd_,egammaJetTracks_, tracks);
   
   cout<<" Size of collections "<<(*ecal).size()<<" "<<Hithbhe.size()<<" "<<Hitho.size()<<" "<<
   Hithf.size()<<" "<<(*jets).size()<<" "<<(*tracks).size()<<endl;

}
//define this as a plug-in
//DEFINE_ANOTHER_FWK_MODULE(ProducerAnalyzer)
}
