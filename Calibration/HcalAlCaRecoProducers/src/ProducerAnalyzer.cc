// system include files

// user include files

#include "Calibration/HcalAlCaRecoProducers/src/ProducerAnalyzer.h"
#include "FWCore/Common/interface/Provenance.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h" 
#include "Geometry/CaloGeometry/interface/CaloGeometry.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 
#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include <map>

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

   tok_hovar_ = consumes<HOCalibVariableCollection>( edm::InputTag(nameProd_,hoInput_) );
   tok_horeco_ = consumes<HORecHitCollection>( edm::InputTag("horeco") );
   tok_ho_ = consumes<HORecHitCollection>( edm::InputTag(hoInput_) );
   tok_hoProd_ = consumes<HORecHitCollection>( edm::InputTag(nameProd_,hoInput_) );

    tok_hf_ = consumes<HFRecHitCollection>( edm::InputTag(hfInput_) );

   tok_jets_ = consumes<reco::CaloJetCollection>( edm::InputTag(nameProd_,jetCalo_) );
   tok_gamma_ = consumes<reco::SuperClusterCollection>( edm::InputTag(nameProd_,gammaClus_) );
   tok_muons_ = consumes<reco::MuonCollection>(edm::InputTag(nameProd_,"SelectedMuons"));
   tok_ecal_ = consumes<EcalRecHitCollection>( edm::InputTag(nameProd_,ecalInput_) );
   tok_tracks_ = consumes<reco::TrackCollection>( edm::InputTag(nameProd_,Tracks_) );

   tok_hbheProd_ = consumes<HBHERecHitCollection>( edm::InputTag(nameProd_,hbheInput_) );
   tok_hbhe_ = consumes<HBHERecHitCollection>( edm::InputTag(hbheInput_) );

}

ProducerAnalyzer::~ProducerAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void ProducerAnalyzer::beginJob()
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

   const CaloGeometry* geo;
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   geo = pG.product();
   

  std::vector<Provenance const*> theProvenance;
  iEvent.getAllProvenance(theProvenance);
  for( std::vector<Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<moduleName(**ip)<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
  
  
  if(nameProd_ == "hoCalibProducer")
  {
     edm::Handle<HOCalibVariableCollection> ho;
     iEvent.getByToken(tok_hovar_, ho);
     const HOCalibVariableCollection Hitho = *(ho.product());
     std::cout<<" Size of HO "<<(Hitho).size()<<std::endl;
  }
  
   if(nameProd_ == "ALCARECOMuAlZMuMu" )
   {
   
   edm::Handle<HORecHitCollection> ho;
   iEvent.getByToken(tok_horeco_, ho);
   const HORecHitCollection Hitho = *(ho.product());
   std::cout<<" Size of HO "<<(Hitho).size()<<std::endl;
   edm::Handle<MuonCollection> mucand;
   iEvent.getByToken(tok_muons_, mucand);
   std::cout<<" Size of muon collection "<<mucand->size()<<std::endl;
   for(MuonCollection::const_iterator it =  mucand->begin(); it != mucand->end(); it++)
   {
      TrackRef mu = (*it).combinedMuon();
      std::cout<<" Pt muon "<<mu->innerMomentum()<<std::endl;
   }
   
   }  
  
   if(nameProd_ != "IsoProd" && nameProd_ != "ALCARECOMuAlZMuMu" && nameProd_ != "hoCalibProducer")
   {
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByToken(tok_hbhe_, hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());
   std::cout<<" Size of HBHE "<<(Hithbhe).size()<<std::endl;


   edm::Handle<HORecHitCollection> ho;
   iEvent.getByToken(tok_ho_, ho);
   const HORecHitCollection Hitho = *(ho.product());
   std::cout<<" Size of HO "<<(Hitho).size()<<std::endl;


   edm::Handle<HFRecHitCollection> hf;
   iEvent.getByToken(tok_hf_, hf);
   const HFRecHitCollection Hithf = *(hf.product());
   std::cout<<" Size of HF "<<(Hithf).size()<<std::endl;
   }
   if(nameProd_ == "IsoProd")
   {
   cout<<" We are here "<<endl;
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByToken(tok_tracks_,tracks);
 
   
   std::cout<<" Tracks size "<<(*tracks).size()<<std::endl;
   reco::TrackCollection::const_iterator track = tracks->begin ();

          for (; track != tracks->end (); track++)
         {
           cout<<" P track "<<(*track).p()<<" eta "<<(*track).eta()<<" phi "<<(*track).phi()<<" Outer "<<(*track).outerMomentum()<<" "<<
	   (*track).outerPosition()<<endl;
	   TrackExtraRef myextra = (*track).extra();
	   cout<<" Track extra "<<myextra->outerMomentum()<<" "<<myextra->outerPosition()<<endl;
         }  

   edm::Handle<EcalRecHitCollection> ecal;
   iEvent.getByToken(tok_ecal_,ecal);
   const EcalRecHitCollection Hitecal = *(ecal.product());
   std::cout<<" Size of Ecal "<<(Hitecal).size()<<std::endl;
   EcalRecHitCollection::const_iterator hite = (ecal.product())->begin ();

         double energyECAL = 0.;
         double energyHCAL = 0.;

          for (; hite != (ecal.product())->end (); hite++)
         {

//           cout<<" Energy ECAL "<<(*hite).energy()<<endl;


//	   " eta "<<(*hite).detid()<<" phi "<<(*hite).detid().getPosition().phi()<<endl;

	 GlobalPoint posE = geo->getPosition((*hite).detid());

           cout<<" Energy ECAL "<<(*hite).energy()<<
	   " eta "<<posE.eta()<<" phi "<<posE.phi()<<endl;

         energyECAL = energyECAL + (*hite).energy();
	 
         }

   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByToken(tok_hbheProd_,hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());
   std::cout<<" Size of HBHE "<<(Hithbhe).size()<<std::endl;
   HBHERecHitCollection::const_iterator hith = (hbhe.product())->begin ();

          for (; hith != (hbhe.product())->end (); hith++)
         {

	 GlobalPoint posH = geo->getPosition((*hith).detid());

           cout<<" Energy HCAL "<<(*hith).energy()<<
	   " eta "<<posH.eta()<<" phi "<<posH.phi()<<endl;

         energyHCAL = energyHCAL + (*hith).energy();
	 
         }
   
   cout<<" Energy ECAL "<< energyECAL<<" Energy HCAL "<< energyHCAL<<endl;
   
   edm::Handle<HORecHitCollection> ho;
   iEvent.getByToken(tok_hoProd_,ho);
   const HORecHitCollection Hitho = *(ho.product());
   std::cout<<" Size of HO "<<(Hitho).size()<<std::endl;
   HORecHitCollection::const_iterator hito = (ho.product())->begin ();

          for (; hito != (ho.product())->end (); hito++)
         {
//           cout<<" Energy HO    "<<(*hito).energy()<<endl;
//	   " eta "<<(*hite).eta()<<" phi "<<(*hite).phi()<<endl;
         }

   }
   
   
   if(nameProd_ == "GammaJetProd" || nameProd_ == "DiJProd")
   {
    cout<<" we are in GammaJetProd area "<<endl;
   edm::Handle<EcalRecHitCollection> ecal;
   iEvent.getByToken(tok_ecal_, ecal);
   std::cout<<" Size of ECAL "<<(*ecal).size()<<std::endl;

   edm::Handle<reco::CaloJetCollection> jets;
   iEvent.getByToken(tok_jets_, jets);
   std::cout<<" Jet size "<<(*jets).size()<<std::endl; 
   reco::CaloJetCollection::const_iterator jet = jets->begin ();
          for (; jet != jets->end (); jet++)
         {
           cout<<" Et jet "<<(*jet).et()<<" eta "<<(*jet).eta()<<" phi "<<(*jet).phi()<<endl;
         }  

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByToken(tok_tracks_, tracks);
   std::cout<<" Tracks size "<<(*tracks).size()<<std::endl; 
   }
   if( nameProd_ == "GammaJetProd")
   {
   edm::Handle<reco::SuperClusterCollection> eclus;
   iEvent.getByToken(tok_gamma_, eclus);
   std::cout<<" GammaClus size "<<(*eclus).size()<<std::endl;
      reco::SuperClusterCollection::const_iterator iclus = eclus->begin ();
          for (; iclus != eclus->end (); iclus++)
         {
           cout<<" Et gamma "<<(*iclus).energy()/cosh((*iclus).eta())<<" eta "<<(*iclus).eta()<<" phi "<<(*iclus).phi()<<endl;
         }
   }

}
//define this as a plug-in
//DEFINE_FWK_MODULE(ProducerAnalyzer)
}
