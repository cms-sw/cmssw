#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "Math/VectorUtil.h"
using namespace ROOT::Math;

using namespace reco;

TCTauAlgorithm::TCTauAlgorithm(){
	init();
}

TCTauAlgorithm::TCTauAlgorithm(const edm::ParameterSet& iConfig){
        init();
	inputConfig(iConfig);
}

TCTauAlgorithm::~TCTauAlgorithm(){}

void TCTauAlgorithm::init(){

	event = 0;
	setup = 0;

	trackAssociator = new TrackDetectorAssociator();
  	trackAssociator->useDefaultPropagator();

        all    = 0;
        passed = 0;
	prongs = -1;
	algoComponentUsed = 0;
}

void TCTauAlgorithm::inputConfig(const edm::ParameterSet& iConfig){

	etCaloOverTrackMin = iConfig.getParameter<double>("EtCaloOverTrackMin");
	etCaloOverTrackMax = iConfig.getParameter<double>("EtCaloOverTrackMax");
	etHcalOverTrackMin = iConfig.getParameter<double>("EtHcalOverTrackMin");
        etHcalOverTrackMax = iConfig.getParameter<double>("EtHcalOverTrackMax");

	signalCone         = iConfig.getParameter<double>("SignalConeSize");
	ecalCone	   = iConfig.getParameter<double>("EcalConeSize");

	EcalRecHitsEB_input= iConfig.getParameter<edm::InputTag>("EBRecHitCollection");
	EcalRecHitsEE_input= iConfig.getParameter<edm::InputTag>("EERecHitCollection");
	HBHERecHits_input  = iConfig.getParameter<edm::InputTag>("HBHERecHitCollection");
	HORecHits_input    = iConfig.getParameter<edm::InputTag>("HORecHitCollection");
	HFRecHits_input    = iConfig.getParameter<edm::InputTag>("HFRecHitCollection");

	edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  	trackAssociatorParameters.loadParameters( pset );

	dropCaloJets       = iConfig.getUntrackedParameter<bool>("DropCaloJets",false);
	dropRejected       = iConfig.getUntrackedParameter<bool>("DropRejectedJets",true);
}


double TCTauAlgorithm::efficiency(){
	return double(passed)/all;
}

int TCTauAlgorithm::statistics(){
	return passed;
}

int TCTauAlgorithm::allTauCandidates(){
        return all;
}

int TCTauAlgorithm::algoComponent(){
        return algoComponentUsed;
}

void TCTauAlgorithm::eventSetup(const edm::Event& iEvent,const edm::EventSetup& iSetup){

	event = &iEvent;
	setup = &iSetup;

	edm::ESHandle<TransientTrackBuilder> builder;
        iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
        transientTrackBuilder = builder.product();

        // geometry initialization
        edm::ESHandle<CaloGeometry> geometry;
        iSetup.get<CaloGeometryRecord>().get(geometry);


        EB = geometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
        EE = geometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
        HB = geometry->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
        HE = geometry->getSubdetectorGeometry(DetId::Hcal,HcalEndcap);
        HO = geometry->getSubdetectorGeometry(DetId::Hcal,HcalOuter);
        HF = geometry->getSubdetectorGeometry(DetId::Hcal,HcalForward);

        //hits
	iEvent.getByLabel( EcalRecHitsEB_input, EBRecHits );
        iEvent.getByLabel( EcalRecHitsEE_input, EERecHits );

        iEvent.getByLabel( HBHERecHits_input, HBHERecHits );
        iEvent.getByLabel( HORecHits_input, HORecHits );
        iEvent.getByLabel( HFRecHits_input, HFRecHits );
}


math::XYZTLorentzVector TCTauAlgorithm::recalculateEnergy(const reco::CaloTau& jet){

	const TrackRef& leadTk = jet.leadTrack();

	const TrackRefVector associatedTracks = jet.caloTauTagInfoRef()->Tracks();

	const CaloJet* cJet = jet.caloTauTagInfoRef()->calojetRef().get();
	CaloJet caloJet = *cJet;
	caloJet.setP4(jet.p4());

	return recalculateEnergy(caloJet,leadTk,associatedTracks);
}

math::XYZTLorentzVector TCTauAlgorithm::recalculateEnergy(const reco::CaloJet& caloJet,const TrackRef& leadTk,const TrackRefVector& associatedTracks){

        all++;

        math::XYZTLorentzVector p4(0,0,0,0);
	algoComponentUsed = TCAlgoUndetermined;

	//if(!dropRejected) 
	p4 = caloJet.p4();

        if(leadTk.isNull()) return p4;

	XYZVector momentum(0,0,0);
	int prongCounter = 0;
        edm::RefVector<TrackCollection>::const_iterator iTrack;
        for(iTrack = associatedTracks.begin(); iTrack!= associatedTracks.end(); ++iTrack){
		double DR = ROOT::Math::VectorUtil::DeltaR(leadTk->momentum(),(*iTrack)->momentum());
		if(DR < signalCone) {
			momentum+=(*iTrack)->momentum();
			prongCounter++;
		}
	}
        if(momentum.Rho() == 0) return p4;

	XYZVector ltrackEcalHitPoint = trackEcalHitPoint(*leadTk);

	if(! (ltrackEcalHitPoint.Rho() > 0 && ltrackEcalHitPoint.Rho() < 9999) ) return p4;

        std::pair<XYZVector,XYZVector> caloClusters = getClusterEnergy(caloJet,ltrackEcalHitPoint,signalCone);
	XYZVector EcalCluster = caloClusters.first;
        XYZVector HcalCluster = caloClusters.second;

	double eCaloOverTrack = (EcalCluster.R()+HcalCluster.R()-momentum.R())/momentum.R();

        std::pair<XYZVector,XYZVector> caloClustersPhoton = getClusterEnergy(caloJet,ltrackEcalHitPoint,ecalCone);
        XYZVector EcalClusterPhoton = caloClustersPhoton.first;

	math::XYZTLorentzVector p4photons(0,0,0,EcalClusterPhoton.R() - EcalCluster.R());

        if( eCaloOverTrack > etCaloOverTrackMin  && eCaloOverTrack < etCaloOverTrackMax ) {

                double eHcalOverTrack = (HcalCluster.R()-momentum.R())/momentum.R();

                if ( eHcalOverTrack  > etHcalOverTrackMin  && eHcalOverTrack  < etHcalOverTrackMax ) {
                  p4.SetXYZT(EcalCluster.X()   + momentum.X(),
                             EcalCluster.Y()   + momentum.Y(),
                             EcalCluster.Z()   + momentum.Z(),
                             EcalCluster.R()   + momentum.R());
                  p4 += p4photons;
		  algoComponentUsed = TCAlgoMomentumECAL;
                }else{
	          p4.SetXYZT(momentum.X(),
	                     momentum.Y(),
        	             momentum.Z(),
                	     momentum.R());
                  algoComponentUsed = TCAlgoMomentum;
		}
        }
        if( eCaloOverTrack  > etCaloOverTrackMax ) {
                double eHcalOverTrack = (HcalCluster.R()-momentum.R())/momentum.R();

                if ( eHcalOverTrack  > etHcalOverTrackMin  && eHcalOverTrack  < etHcalOverTrackMax ) {
                  p4.SetXYZT(EcalCluster.X()   + momentum.X(),
                             EcalCluster.Y()   + momentum.Y(),
                             EcalCluster.Z()   + momentum.Z(),
                             EcalCluster.R()   + momentum.R());
                  p4 += p4photons;
                  algoComponentUsed = TCAlgoMomentumECAL;
                }
                if ( eHcalOverTrack  < etHcalOverTrackMin ) {
                  if(!dropCaloJets) p4.SetXYZT(caloJet.px(),caloJet.py(),caloJet.pz(),caloJet.energy());
                  else p4.SetXYZT(0,0,0,0);
                  algoComponentUsed = TCAlgoCaloJet;
                }
		if ( eHcalOverTrack  > etHcalOverTrackMax ) {
		  algoComponentUsed = TCAlgoHadronicJet; // reject
		  if(!dropRejected) p4.SetXYZT(caloJet.px(),caloJet.py(),caloJet.pz(),caloJet.energy());
		  else p4.SetXYZT(0,0,0,0);
		}
        }
	if( eCaloOverTrack  < etCaloOverTrackMin ) {
	          algoComponentUsed = TCAlgoTrackProblem; // reject
		  if(!dropRejected) p4.SetXYZT(caloJet.px(),caloJet.py(),caloJet.pz(),caloJet.energy());
	}

	if(p4.Et() > 0) passed++;

	return p4;
}


XYZVector TCTauAlgorithm::trackEcalHitPoint(const TransientTrack& transientTrack,const CaloJet& caloJet){


        GlobalPoint ecalHitPosition(0,0,0);

        double maxTowerEt = 0;
	std::vector<CaloTowerPtr> towers = caloJet.getCaloConstituents();
        for(std::vector<CaloTowerPtr>::const_iterator iTower = towers.begin();
                                                 iTower!= towers.end(); ++iTower){
                if((*iTower)->et() > maxTowerEt){
                        maxTowerEt = (*iTower)->et();
                        ecalHitPosition = (*iTower)->emPosition();
                }
        }


        XYZVector ecalHitPoint(0,0,0);

		GlobalPoint trackEcalHitPoint = transientTrack.stateOnSurface(ecalHitPosition).globalPosition();

                ecalHitPoint.SetXYZ(trackEcalHitPoint.x(),
                                    trackEcalHitPoint.y(),
                                    trackEcalHitPoint.z());

        return ecalHitPoint;
}

XYZVector TCTauAlgorithm::trackEcalHitPoint(const Track& track){

      	const FreeTrajectoryState fts = trackAssociator->getFreeTrajectoryState(*setup,track);
      	TrackDetMatchInfo info = trackAssociator->associate(*event, *setup, fts, trackAssociatorParameters);
      	if( info.isGoodEcal != 0 ) {
          return XYZVector(info.trkGlobPosAtEcal.x(),info.trkGlobPosAtEcal.y(),info.trkGlobPosAtEcal.z());
	}
	return XYZVector(0,0,0);
}

std::pair<XYZVector,XYZVector> TCTauAlgorithm::getClusterEnergy(const CaloJet& caloJet,XYZVector& trackEcalHitPoint,double cone){

        XYZVector ecalCluster(0,0,0);
        XYZVector hcalCluster(0,0,0);

        std::vector<CaloTowerPtr> towers = caloJet.getCaloConstituents();

        for(std::vector<CaloTowerPtr>::const_iterator iTower = towers.begin();
                                                 iTower!= towers.end(); ++iTower){
                std::vector<XYZVector> ECALCells;
                std::vector<XYZVector> HCALCells;

                size_t numRecHits = (**iTower).constituentsSize();

                // access CaloRecHits
                for(size_t j = 0; j < numRecHits; j++) {
                        DetId recHitDetID = (**iTower).constituent(j);
                        //DetId::Detector detNum=recHitDetID.det();
                        if( recHitDetID.det() == DetId::Ecal ){
                          if( recHitDetID.subdetId() == 1 ){ // Ecal Barrel
                                EBDetId ecalID = recHitDetID;
                                EBRecHitCollection::const_iterator theRecHit = EBRecHits->find(ecalID);
                                if(theRecHit != EBRecHits->end()){
                                  DetId id = theRecHit->detid();
                                  const CaloCellGeometry* this_cell = EB->getGeometry(id);
                                  double energy = theRecHit->energy();
                                  ECALCells.push_back(getCellMomentum(this_cell,energy));
                                }
                          }
                          if( recHitDetID.subdetId() == 2 ){ // Ecal Endcap
                                EEDetId ecalID = recHitDetID;
                                EERecHitCollection::const_iterator theRecHit = EERecHits->find(ecalID);
                                if(theRecHit != EERecHits->end()){
                                  DetId id = theRecHit->detid();
                                  const CaloCellGeometry* this_cell = EE->getGeometry(id);
                                  double energy = theRecHit->energy();
                                  ECALCells.push_back(getCellMomentum(this_cell,energy));
                                }
                          }
                        }
                        if( recHitDetID.det() == DetId::Hcal ){
                          HcalDetId hcalID = recHitDetID;
                          if( recHitDetID.subdetId() == HcalBarrel ){
                            //int depth = hcalID.depth();
                            //if (depth==1){
                                HBHERecHitCollection::const_iterator theRecHit=HBHERecHits->find(hcalID);
                                if(theRecHit != HBHERecHits->end()){
                                  DetId id = theRecHit->detid();
                                  const CaloCellGeometry* this_cell = HB->getGeometry(id);
                                  double energy = theRecHit->energy();
                                  HCALCells.push_back(getCellMomentum(this_cell,energy));
                                }
                            //}
                          }
                          if( recHitDetID.subdetId() == HcalEndcap ){
                            //int depth = hcalID.depth();
                            //if (depth==1){
                                HBHERecHitCollection::const_iterator theRecHit=HBHERecHits->find(hcalID);
                                if(theRecHit != HBHERecHits->end()){
                                  DetId id = theRecHit->detid();
                                  const CaloCellGeometry* this_cell = HE->getGeometry(id);
                                  double energy = theRecHit->energy();
                                  HCALCells.push_back(getCellMomentum(this_cell,energy));
                                }
                            //}
                          }
                          if( recHitDetID.subdetId() == HcalOuter ){
                                HORecHitCollection::const_iterator theRecHit=HORecHits->find(hcalID);
                                if(theRecHit != HORecHits->end()){
                                  DetId id = theRecHit->detid();
                                  const CaloCellGeometry* this_cell = HO->getGeometry(id);
                                  double energy = theRecHit->energy();
                                  HCALCells.push_back(getCellMomentum(this_cell,energy));
                                }
                          }
                          if( recHitDetID.subdetId() == HcalForward ){
                                HFRecHitCollection::const_iterator theRecHit=HFRecHits->find(hcalID);
                                if(theRecHit != HFRecHits->end()){
                                  DetId id = theRecHit->detid();
                                  const CaloCellGeometry* this_cell = HF->getGeometry(id);
                                  double energy = theRecHit->energy();
                                  HCALCells.push_back(getCellMomentum(this_cell,energy));
                                }
                          }
                        }
                }

		std::vector<XYZVector>::const_iterator i;
                for(i = ECALCells.begin(); i != ECALCells.end(); ++i) {
			double DR = ROOT::Math::VectorUtil::DeltaR(trackEcalHitPoint,*i);
                        if( DR < cone ) ecalCluster += *i;
                }
                for(i = HCALCells.begin(); i != HCALCells.end(); ++i) {
			double DR = ROOT::Math::VectorUtil::DeltaR(trackEcalHitPoint,*i);
                        if( DR < cone ) hcalCluster += *i;
                }
	}
        return std::pair<XYZVector,XYZVector> (ecalCluster,hcalCluster);
}

XYZVector TCTauAlgorithm::getCellMomentum(const CaloCellGeometry* cell,double& energy){
        XYZVector momentum(0,0,0);
        if(cell){
                GlobalPoint hitPosition = cell->getPosition();

                double phi   = hitPosition.phi();
                double theta = hitPosition.theta();
                if(theta > 3.14159) theta = 2*3.14159 - theta;
                double px = energy * sin(theta)*cos(phi);
                double py = energy * sin(theta)*sin(phi);
                double pz = energy * cos(theta);

                momentum = XYZVector(px,py,pz);
        }
        return momentum;
}
