// -*- C++ -*-
//
// Package:    EopTreeWriter
// Class:      EopTreeWriter
// 
/**\class EopTreeWriter EopTreeWriter.cc Alignment/OfflineValidation/plugins/EopTreeWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Holger Enderle
//         Created:  Thu Dec  4 11:22:48 CET 2008
// $Id$
//
//

#include "EopTreeWriter.h"

//
// class decleration
//

class EopTreeWriter : public edm::EDAnalyzer {
   public:
      explicit EopTreeWriter(const edm::ParameterSet&);
      ~EopTreeWriter();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      edm::InputTag src_;

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
EopTreeWriter::EopTreeWriter(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src"))
{
   //now do what ever initialization is needed

   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );

   tree = fs->make<TTree>("EopTree","EopTree");
   treeMemPtr = new EopVariables;
   tree->Branch("EopVariables", &treeMemPtr); // address of pointer!
}


EopTreeWriter::~EopTreeWriter()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EopTreeWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get geometry
   edm::ESHandle<CaloGeometry> geometry;
   iSetup.get<CaloGeometryRecord>().get(geometry);
   const CaloGeometry* geo = geometry.product();
//    const CaloSubdetectorGeometry* towerGeometry = 
//      geo->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);

   // temporary collection of EB+EE recHits
   std::auto_ptr<EcalRecHitCollection> tmpEcalRecHitCollection(new EcalRecHitCollection);
   std::vector<edm::InputTag> ecalLabels_;

   edm::Handle<EcalRecHitCollection> tmpEc;
   bool ecalInAlca = iEvent.getByLabel(edm::InputTag("IsoProd","IsoTrackEcalRecHitCollection"),tmpEc);
   bool ecalInReco = iEvent.getByLabel(edm::InputTag("ecalRecHit","EcalRecHitsEB"),tmpEc)&& 
                     iEvent.getByLabel(edm::InputTag("ecalRecHit","EcalRecHitsEE"),tmpEc);
   if(ecalInAlca)ecalLabels_.push_back(edm::InputTag("IsoProd","IsoTrackEcalRecHitCollection"));
   else if(ecalInReco){
     ecalLabels_.push_back(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
     ecalLabels_.push_back(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
   }
   else throw cms::Exception("MissingProduct","can not find EcalRecHits");

   std::vector<edm::InputTag>::const_iterator i;
   for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) 
     {
       edm::Handle<EcalRecHitCollection> ec;
       iEvent.getByLabel(*i,ec);
       for(EcalRecHitCollection::const_iterator recHit = (*ec).begin(); recHit != (*ec).end(); ++recHit)
	 {
	   tmpEcalRecHitCollection->push_back(*recHit);
	 }
     }      
   
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(src_, tracks);

   edm::Handle<reco::IsolatedPixelTrackCandidateCollection> isoPixelTracks;
   edm::Handle<reco::IsolatedPixelTrackCandidateCollection> tmpPix;
   bool pixelInAlca = iEvent.getByLabel(edm::InputTag("IsoProd","HcalIsolatedTrackCollection"),tmpPix);
   if(pixelInAlca)iEvent.getByLabel(edm::InputTag("IsoProd","HcalIsolatedTrackCollection"),isoPixelTracks);

   Double_t trackemc1;
   Double_t trackemc3;
   Double_t trackemc5;
   Double_t trackhac1;
   Double_t trackhac3;
   Double_t trackhac5;
   Double_t maxPNearby;
   Double_t dist;
   Double_t EnergyIn;
   Double_t EnergyOut;

   parameters_.useMuon = false;

   if(pixelInAlca)
     if(isoPixelTracks->size()==0) return;

   for(reco::TrackCollection::const_iterator track = tracks->begin();track!=tracks->end();++track){

     bool noChargedTracks = true;

     if(track->p()<9.) continue;

     trackAssociator_.useDefaultPropagator();
     TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track), parameters_);

     trackemc1 = 0;
     trackemc3 = 0;
     trackemc5 = 0;
     trackhac1 = 0;
     trackhac3 = 0;
     trackhac5 = 0;
     
     trackemc1 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 0);
     trackemc3 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
     trackemc5 = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);
     trackhac1 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 0);
     trackhac3 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
     trackhac5 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 2);

     if(trackhac3<5.) continue;

     double etaecal=info.trkGlobPosAtEcal.eta();
     double phiecal=info.trkGlobPosAtEcal.phi();
       
     maxPNearby=-10;
     dist=50;
     for (reco::TrackCollection::const_iterator track1 = tracks->begin(); track1!=tracks->end(); track1++)
       {
	 if (track == track1) continue;
	 TrackDetMatchInfo info1 = trackAssociator_.associate(iEvent, iSetup, *track1, parameters_);
	 double etaecal1=info1.trkGlobPosAtEcal.eta();
	 double phiecal1=info1.trkGlobPosAtEcal.phi();

	 if (etaecal1==0&&phiecal1==0) continue;	

	 double ecDist=getDistInCM(etaecal,phiecal,etaecal1,phiecal1);

	 if( ecDist <  40. ) 
	   {
	     //calculate maximum P and sum P near seed track
	     if (track1->p()>maxPNearby)
	       {
		 maxPNearby=track1->p();
		 dist = ecDist;
	       }
	     
	     //apply loose isolation criteria
	     if (track1->p()>5.) 
	       {
		 noChargedTracks = false;
		 break;
	       }
	   }
       }
     EnergyIn=0;
     EnergyOut=0;
     if(noChargedTracks){
       for (std::vector<EcalRecHit>::const_iterator ehit=tmpEcalRecHitCollection->begin(); ehit!=tmpEcalRecHitCollection->end(); ehit++) 
	 {
	   ////////////////////// FIND ECAL CLUSTER ENERGY
	   // R-scheme of ECAL CLUSTERIZATION
	   GlobalPoint posH = geo->getPosition((*ehit).detid());
	   double phihit = posH.phi();
	   double etahit = posH.eta();
	   
	   double dHitCM=getDistInCM(etaecal,phiecal,etahit,phihit);
	   
	   if (dHitCM<9.0)
	     {
	       EnergyIn+=ehit->energy();
	     }
	   if (dHitCM>15.0&&dHitCM<35.0)
	     {
	       EnergyOut+=ehit->energy();
	     }
	   
	 }

       treeMemPtr->fillVariables(track->charge(), track->innerOk(), track->outerRadius(),
				 track->numberOfValidHits(), track->numberOfLostHits(),
				 track->chi2(), track->normalizedChi2(),
				 track->p(), track->pt(), track->ptError(), 
				 track->theta(), track->eta(), track->phi(),
				 trackemc1, trackemc3, trackemc5,
				 trackhac1, trackhac3, trackhac5,
				 maxPNearby, dist, EnergyIn, EnergyOut);
       
       tree->Fill();
       }
   }

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
EopTreeWriter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EopTreeWriter::endJob() {

  delete treeMemPtr; treeMemPtr = 0;

}

//define this as a plug-in
DEFINE_FWK_MODULE(EopTreeWriter);
