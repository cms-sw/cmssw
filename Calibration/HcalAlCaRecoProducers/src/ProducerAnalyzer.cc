// system include files

// user include files

#include "Calibration/HcalAlCaRecoProducers/src/ProducerAnalyzer.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"


using namespace std;
using namespace reco;

namespace cms
{

//
// constructors and destructor
//
ProducerAnalyzer::ProducerAnalyzer(const edm::ParameterSet& iConfig)
{
  // get name of output file with histogramms  
   
   nameProd_ = iConfig.getUntrackedParameter<std::string>("nameProd");
   jetCalo_ = iConfig.getUntrackedParameter<std::string>("jetCalo","GammaJetJetBackToBackCollection");
   gammaClus_ = iConfig.getUntrackedParameter<std::string>("gammaClus","GammaJetGammaBackToBackCollection");
   ecalInput_=iConfig.getUntrackedParameter<std::string>("ecalInput","GammaJetEcalRecHitCollection");
   hbheInput_ = iConfig.getUntrackedParameter<std::string>("hbheInput");
   hoInput_ = iConfig.getUntrackedParameter<std::string>("hoInput");
   hfInput_ = iConfig.getUntrackedParameter<std::string>("hfInput");
   Tracks_ = iConfig.getUntrackedParameter<std::string>("Tracks","GammaJetTracksCollection"); 

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
   if(nameProd_ != "IsoProd")
   {
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(nameProd_,hbheInput_, hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());
   std::cout<<" Size of HBHE "<<(Hithbhe).size()<<std::endl;


   edm::Handle<HORecHitCollection> ho;
   iEvent.getByLabel(nameProd_,hoInput_, ho);
   const HORecHitCollection Hitho = *(ho.product());
   std::cout<<" Size of HO "<<(Hitho).size()<<std::endl;


   edm::Handle<HFRecHitCollection> hf;
   iEvent.getByLabel(nameProd_,hfInput_, hf);
   const HFRecHitCollection Hithf = *(hf.product());
   std::cout<<" Size of HF "<<(Hithf).size()<<std::endl;
   }
   if(nameProd_ == "IsoProd")
   {
   cout<<" We are here "<<endl;
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(nameProd_,Tracks_,tracks);
   std::cout<<" Tracks size "<<(*tracks).size()<<std::endl;

   edm::Handle<EcalRecHitCollection> ecal;
   iEvent.getByLabel(nameProd_,ecalInput_,ecal);
   const EcalRecHitCollection Hitecal = *(ecal.product());
   std::cout<<" Size of HO "<<(Hitecal).size()<<std::endl;


   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(nameProd_,hbheInput_,hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());
   std::cout<<" Size of HBHE "<<(Hithbhe).size()<<std::endl;
   
   edm::Handle<HORecHitCollection> ho;
   iEvent.getByLabel(nameProd_,hoInput_,ho);
   const HORecHitCollection Hitho = *(ho.product());
   std::cout<<" Size of HBHE "<<(Hitho).size()<<std::endl;



   }
   if(nameProd_ == "GammaJetProd" || nameProd_ == "DiJProd")
   {
    cout<<" we are in GammaJetProd area "<<endl;
   edm::Handle<EcalRecHitCollection> ecal;
   iEvent.getByLabel(nameProd_,ecalInput_, ecal);
   std::cout<<" Size of ECAL "<<(*ecal).size()<<std::endl;

   edm::Handle<reco::CaloJetCollection> jets;
   iEvent.getByLabel(nameProd_,jetCalo_, jets);
   std::cout<<" Jet size "<<(*jets).size()<<std::endl; 
   reco::CaloJetCollection::const_iterator jet = jets->begin ();
          for (; jet != jets->end (); jet++)
         {
           cout<<" Et jet "<<(*jet).et()<<" eta "<<(*jet).eta()<<" phi "<<(*jet).phi()<<endl;
         }  

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(nameProd_,Tracks_, tracks);
   std::cout<<" Tracks size "<<(*tracks).size()<<std::endl; 
   }
   if( nameProd_ == "GammaJetProd")
   {
   edm::Handle<reco::SuperClusterCollection> eclus;
   iEvent.getByLabel(nameProd_,gammaClus_, eclus);
   std::cout<<" GammaClus size "<<(*eclus).size()<<std::endl;
      reco::SuperClusterCollection::const_iterator iclus = eclus->begin ();
          for (; iclus != eclus->end (); iclus++)
         {
           cout<<" Et gamma "<<(*iclus).energy()/cosh((*iclus).eta())<<" eta "<<(*iclus).eta()<<" phi "<<(*iclus).phi()<<endl;
         }
   }

}
//define this as a plug-in
//DEFINE_ANOTHER_FWK_MODULE(ProducerAnalyzer)
}
