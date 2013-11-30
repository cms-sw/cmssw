// -*- C++ -*-
//
// Package:    DQMAnalyzerSTEP1
// Class:      DQMAnalyzerSTEP1
// 
/**\class DQMAnalyzerSTEP1 DQMAnalyzerSTEP1.cc DQMAnalyzerStep1/DQMAnalyzerSTEP1/plugins/DQMAnalyzerSTEP1.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Cesare Calabria
//         Created:  Wed, 06 Nov 2013 11:27:47 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
 
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "DataFormats/Provenance/interface/Timestamp.h"

#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace std;
using namespace edm;

//
// class declaration
//

class DQMAnalyzerSTEP1 : public edm::EDAnalyzer {
   public:
      explicit DQMAnalyzerSTEP1(const edm::ParameterSet&);
      ~DQMAnalyzerSTEP1();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      // Histograms

      MonitorElement * hNumSimTracks;
      MonitorElement * hNumMuonSimTracks;
      MonitorElement * hNumRecTracks;
      MonitorElement * hNumGEMSimHits;
      MonitorElement * hNumGEMRecHits;
      MonitorElement * hPtResVsPt;
      MonitorElement * hInvPtResVsPt;
      MonitorElement * hPtResVsEta;
      MonitorElement * hInvPtResVsEta;
      MonitorElement * hDenSimPt;
      MonitorElement * hDenSimEta;
      MonitorElement * hDenSimPhiPlus;
      MonitorElement * hDenSimPhiMinus;
      MonitorElement * hNumSimPt;
      MonitorElement * hNumSimEta;
      MonitorElement * hNumSimPhiPlus;
      MonitorElement * hNumSimPhiMinus;
      MonitorElement * hDR;
      MonitorElement * hDeltaCharge;
      MonitorElement * hSimTrackMatch;
      MonitorElement * hDRMatchVsPt;
      MonitorElement * hMatchedSimHits;

   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

      edm::InputTag staTrackLabel_;
      std::string theSeedCollectionLabel;
      bool debug_;
      std::string folderPath_;
      bool noGEMCase_;
      bool isGlobalMuon_;

      std::map<int, std::map<std::string, MonitorElement*> >  meCollection;

      bool EffSaveRootFile_;
      std::string EffRootFileName_;
      DQMStore * dbe;

};

bool isSimMatched(SimTrackContainer::const_iterator simTrack, edm::PSimHitContainer::const_iterator itHit)
{

  bool result = false;

  int trackId = simTrack->trackId();
  int trackId_sim = itHit->trackId();
  if(trackId == trackId_sim) result = true;

  //std::cout<<"ID: "<<trackId<<" "<<trackId_sim<<" "<<result<<std::endl;

  return result;

}

edm::PSimHitContainer isTrackMatched(SimTrackContainer::const_iterator simTrack, const Event & event, const EventSetup& eventSetup)
{

  edm::PSimHitContainer selectedGEMHits;

  edm::Handle<edm::PSimHitContainer> GEMHits;
  event.getByLabel(edm::InputTag("g4SimHits","MuonGEMHits"), GEMHits);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit){
							 
	DetId id = DetId(itHit->detUnitId());
	if (!(id.subdetId() == MuonSubdetId::GEM)) continue;
  	if(itHit->particleType() != (*simTrack).type()) continue;

	bool result = isSimMatched(simTrack, itHit);
	if(result) selectedGEMHits.push_back(*itHit);

  }

  //std::cout<<"Size: "<<selectedGEMHits.size()<<std::endl;
  return selectedGEMHits;

}

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DQMAnalyzerSTEP1::DQMAnalyzerSTEP1(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   staTrackLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("StandAloneTrackCollectionLabel");
   theSeedCollectionLabel = iConfig.getUntrackedParameter<string>("MuonSeedCollectionLabel");
   debug_ = iConfig.getUntrackedParameter<bool>("debug",false);
   folderPath_ = iConfig.getUntrackedParameter<std::string>("folderPath", "GEMBasicPlots/");
   noGEMCase_ = iConfig.getUntrackedParameter<bool>("NoGEMCase",true);
   isGlobalMuon_ = iConfig.getUntrackedParameter<bool>("isGlobalMuon",false);
   EffSaveRootFile_ = iConfig.getUntrackedParameter<bool>("EffSaveRootFile",true);
   EffRootFileName_ = iConfig.getUntrackedParameter<std::string>("EffRootFileName","GLBMuonAnalyzerWithGEMs_1step.root");

   dbe = edm::Service<DQMStore>().operator->();

   if(debug_) std::cout<<"booking Global histograms with "<<folderPath_<<std::endl;
   
   std::string folder;
   folder = folderPath_;
   dbe->setCurrentFolder(folder);

   hNumSimTracks = dbe->book1D("NumSimTracks","NumSimTracks",101,-0.5,100.5);
   hNumMuonSimTracks = dbe->book1D("NumMuonSimTracks","NumMuonSimTracks",11,-0.5,10.5);
   hNumRecTracks = dbe->book1D("NumRecTracks","NumRecTracks",11,-0.5,10.5);
   hNumGEMSimHits = dbe->book1D("NumGEMSimHits","NumGEMSimHits",11,-0.5,10.5);
   hNumGEMRecHits = dbe->book1D("NumGEMRecHits","NumGEMRecHits",11,-0.5,10.5);
   hPtResVsPt = dbe->book2D("PtResVsPt","p_{T} Resolution vs. Sim p_{T}",261,-2.5,1302.5,1000,-5,5);
   hInvPtResVsPt = dbe->book2D("InvPtResVsPt","1/p_{T} Resolution vs. Sim p_{T}",261,-2.5,1302.5,1000,-5,5);
   hPtResVsEta = dbe->book2D("PtResVsEta","p_{T} Resolution vs. Sim #eta",100,-2.5,2.5,1000,-5,5);
   hInvPtResVsEta = dbe->book2D("InvPtResVsEta","1/p_{T} Resolution vs. Sim #eta",100,-2.5,2.5,1000,-5,5);
   hDenSimPt = dbe->book1D("DenSimPt","DenSimPt",261,-2.5,1302.5);
   hDenSimEta = dbe->book1D("DenSimEta","DenSimEta",100,-2.5,2.5);
   hDenSimPhiPlus = dbe->book1D("DenSimPhiPlus","DenSimPhiMinus",360,0,180);
   hDenSimPhiMinus = dbe->book1D("DenSimPhiMinus","DenSimPhiMinus",360,0,180);
   hNumSimPt = dbe->book1D("NumSimPt","NumSimPt",261,-2.5,1302.5);
   hNumSimEta = dbe->book1D("NumSimEta","NumSimEta",100,-2.5,2.5);
   hNumSimPhiPlus = dbe->book1D("NumSimPhiPlus","NumSimPhiMinus",360,0,180);
   hNumSimPhiMinus = dbe->book1D("NumSimPhiMinus","NumSimPhiMinus",360,0,180);
   hDR = dbe->book1D("DR","#Delta R (SIM-RECO)",300,0,1);
   hDeltaCharge = dbe->book2D("DeltaCharge","#Delta q (SIM-RECO)",261,-2.5,1302.5,6,-3,3);
   hSimTrackMatch = dbe->book1D("SimTrackMatch", "SimTrackMatch",2,0.,2.);
   hDRMatchVsPt = dbe->book2D("DRMatchVsPt","DRMatchVsPt",261,-2.5,1302.5,10,0,10);
   hMatchedSimHits = dbe->book1D("MatchedSimHits","MatchedSimHits",6,-0.5,5.5);

}


DQMAnalyzerSTEP1::~DQMAnalyzerSTEP1()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
DQMAnalyzerSTEP1::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
  Handle<reco::TrackCollection> staTracks;
  iEvent.getByLabel(staTrackLabel_, staTracks);

  Handle<SimTrackContainer> simTracks;
  iEvent.getByLabel("g4SimHits",simTracks);

  ESHandle<MagneticField> theMGField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  edm::ESHandle<GEMGeometry> gemGeom;
  iSetup.get<MuonGeometryRecord>().get(gemGeom);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  edm::Handle<GEMRecHitCollection> gemRecHits; 
  iEvent.getByLabel("gemRecHits","",gemRecHits);

  edm::Handle<edm::PSimHitContainer> GEMHits;
  iEvent.getByLabel(edm::InputTag("g4SimHits","MuonGEMHits"), GEMHits);

  hNumRecTracks->Fill(staTracks->size());
  hNumSimTracks->Fill(simTracks->size());
  if(debug_) cout<<"Reconstructed tracks: " << staTracks->size() << endl;

  reco::TrackCollection::const_iterator staTrack;
  SimTrackContainer::const_iterator simTrack;

  int simCount = 0;

  for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){

		int countMatching = 0;

	      	if (abs((*simTrack).type()) != 13) continue;
  		if ((*simTrack).noVertex()) continue;
  		if ((*simTrack).noGenpart()) continue;

		double simEta = (*simTrack).momentum().eta();
		double simPhi = (*simTrack).momentum().phi();
		double simPt = (*simTrack).momentum().pt();

		if (abs(simEta) > 2.1 || abs(simEta) < 1.64) continue;
		simCount++;

		if(debug_) std::cout<<"SimEta "<<simEta<<" SimPhi "<<simPhi<<std::endl;

		edm::PSimHitContainer selGEMSimHits = isTrackMatched(simTrack, iEvent, iSetup);
		int size = selGEMSimHits.size();
		hMatchedSimHits->Fill(size);
		hSimTrackMatch->Fill(size > 0 ? 1 : 0);
		if(size == 0 && noGEMCase_) continue;

		for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){

			double recEta = staTrack->momentum().eta();
			double recPhi = staTrack->momentum().phi();
			if(debug_) cout<<"RecEta "<<recEta<<" recPhi "<<recPhi<<std::endl;
			double dR = sqrt(pow((simEta-recEta),2) + pow((simPhi-recPhi),2));
			if(debug_) cout<<"dR "<<dR<<std::endl;

			if(dR > 0.1) continue;
			countMatching++;
		    
			double recPt = staTrack->pt();
		    	//cout<<" chi2: "<<track.chi2()<<endl;
	    
		    	if(abs(simEta) > 2.1 || abs(simEta) < 1.64) continue;

			double phi_02pi_sim = simPhi < 0 ? simPhi + TMath::Pi() : simPhi;
			double phiDegSim = phi_02pi_sim * 180/ TMath::Pi();
			bool hasGemRecHits = false;
			int numGEMRecHits = 0;
			int numGEMSimHits = 0;

			hDenSimPt->Fill(simPt);
			hDenSimEta->Fill(simEta);

			if(simEta > 0) hDenSimPhiPlus->Fill(phiDegSim);
			else if(simEta < 0) hDenSimPhiMinus->Fill(phiDegSim);

			for(trackingRecHit_iterator recHit = staTrack->recHitsBegin(); recHit != staTrack->recHitsEnd(); ++recHit){

				if (!((*recHit)->geographicalId().det() == DetId::Muon)) continue;
				if (!((*recHit)->geographicalId().subdetId() == MuonSubdetId::GEM)) continue;

				numGEMRecHits++;
				hasGemRecHits = true;

		      	}

			hNumGEMRecHits->Fill(numGEMRecHits);
			hNumGEMSimHits->Fill(numGEMSimHits);

			if(noGEMCase_) hasGemRecHits = true;
			if(!hasGemRecHits) continue;

			int qGen = simTrack->charge();
			int qRec = staTrack->charge();

			hDeltaCharge->Fill(simPt, qGen-qRec);

			if(debug_) {

				cout<<"RecEta "<<recEta<<" recPhi "<<recPhi<<std::endl;
				cout<<"SimEta "<<simEta<<" SimPhi "<<simPhi<<std::endl;
				cout<<"dR "<<dR<<std::endl;

			}

			hDR->Fill(dR);

			hPtResVsPt->Fill(simPt, (recPt*qRec-simPt*qGen)/(simPt*qGen));
			hPtResVsEta->Fill(simEta, (recPt*qRec-simPt*qGen)/(simPt*qGen));
			hInvPtResVsPt->Fill(simPt, (qRec/recPt - qGen/simPt)/(qGen/simPt));
			hInvPtResVsEta->Fill(simEta, (qRec/recPt - qGen/simPt)/(qGen/simPt));

			hNumSimPt->Fill(simPt);
			hNumSimEta->Fill(simEta);

			if(simEta > 0) hNumSimPhiPlus->Fill(phiDegSim);
			else if(simEta < 0) hNumSimPhiMinus->Fill(phiDegSim);
    
  		}

	hDRMatchVsPt->Fill(simPt, countMatching);

  }

  hNumMuonSimTracks->Fill(simCount);

}


// ------------ method called once each job just before starting event loop  ------------
void 
DQMAnalyzerSTEP1::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DQMAnalyzerSTEP1::endJob() 
{

  dbe = 0;

}

// ------------ method called when starting to processes a run  ------------

void 
DQMAnalyzerSTEP1::beginRun(edm::Run const&, edm::EventSetup const&)
{
}


// ------------ method called when ending the processing of a run  ------------

void 
DQMAnalyzerSTEP1::endRun(edm::Run const&, edm::EventSetup const&)
{

  if (EffSaveRootFile_) dbe->save(EffRootFileName_);

}


// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
DQMAnalyzerSTEP1::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
DQMAnalyzerSTEP1::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DQMAnalyzerSTEP1::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMAnalyzerSTEP1);
