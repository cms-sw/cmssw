#include "JetMETCorrections/TauJet/interface/HardTauAlgorithm.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "Math/VectorUtil.h"
using namespace ROOT::Math;

HardTauAlgorithm::HardTauAlgorithm(){
	init();
}

HardTauAlgorithm::HardTauAlgorithm(const edm::ParameterSet& iConfig){
        init();
	inputConfig(iConfig);
}

HardTauAlgorithm::~HardTauAlgorithm(){}

void HardTauAlgorithm::init(){

	etCaloOverTrackMin = -0.9;
	etCaloOverTrackMax = 0.0;
	etHcalOverTrackMin = -0.5;
	etHcalOverTrackMax = 0.5;

        signalCone         = 0.2;
	ecalCone           = 0.5;
	matchingCone       = 0.1;
	tkptmin            = 1.0;

	tkmaxipt	   = 0.03;
	tkmaxChi2	   = 100;
	tkminPixelHitsn    = 2;
	tkminTrackerHitsn  = 8; 

	trackInput          = InputTag("generalTracks");
	vertexInput         = InputTag("offlinePrimaryVertices");

	EcalRecHitsEB_input = InputTag("ecalRecHit:EcalRecHitsEB");
	EcalRecHitsEE_input = InputTag("ecalRecHit:EcalRecHitsEE");
	HBHERecHits_input   = InputTag("hbhereco");
	HORecHits_input     = InputTag("horeco");
	HFRecHits_input     = InputTag("hfreco");

        all    = 0;
        passed = 0;
	prongs = -1;
}

void HardTauAlgorithm::inputConfig(const edm::ParameterSet& iConfig){

	etCaloOverTrackMin = iConfig.getUntrackedParameter<double>("EtCaloOverTrackMin",etCaloOverTrackMin);
	etCaloOverTrackMax = iConfig.getUntrackedParameter<double>("EtCaloOverTrackMax",etCaloOverTrackMax);
	etHcalOverTrackMin = iConfig.getUntrackedParameter<double>("EtHcalOverTrackMin",etHcalOverTrackMin);
        etHcalOverTrackMax = iConfig.getUntrackedParameter<double>("EtHcalOverTrackMax",etHcalOverTrackMax);

	signalCone         = iConfig.getUntrackedParameter<double>("SignalConeSize",signalCone);
	ecalCone	   = iConfig.getUntrackedParameter<double>("EcalConeSize",ecalCone);
	matchingCone       = iConfig.getUntrackedParameter<double>("MatchingConeSize",matchingCone);
	tkptmin            = iConfig.getUntrackedParameter<double>("Track_minPt",tkptmin);

	tkmaxipt           = iConfig.getUntrackedParameter<double>("tkmaxipt",tkmaxipt);
	tkmaxChi2          = iConfig.getUntrackedParameter<double>("tkmaxChi2",tkmaxChi2);
	tkminPixelHitsn    = iConfig.getUntrackedParameter<int>("tkminPixelHitsn",tkminPixelHitsn);
	tkminTrackerHitsn  = iConfig.getUntrackedParameter<int>("tkminTrackerHitsn",tkminTrackerHitsn);

	trackInput         = iConfig.getUntrackedParameter<InputTag>("TrackCollection",trackInput);
	vertexInput        = iConfig.getUntrackedParameter<InputTag>("PVProducer",vertexInput);

	EcalRecHitsEB_input= iConfig.getUntrackedParameter<InputTag>("EBRecHitCollection",EcalRecHitsEB_input);
	EcalRecHitsEE_input= iConfig.getUntrackedParameter<InputTag>("EERecHitCollection",EcalRecHitsEE_input);
	HBHERecHits_input  = iConfig.getUntrackedParameter<InputTag>("HBHERecHitCollection",HBHERecHits_input);
	HORecHits_input    = iConfig.getUntrackedParameter<InputTag>("HORecHitCollection",HORecHits_input);
	HFRecHits_input    = iConfig.getUntrackedParameter<InputTag>("HFRecHitCollection",HFRecHits_input);

}


double HardTauAlgorithm::efficiency(){
	return double(passed)/all;
}


void HardTauAlgorithm::eventSetup(const edm::Event& iEvent,const edm::EventSetup& iSetup){
	edm::ESHandle<TransientTrackBuilder> builder;
        iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
        transientTrackBuilder = builder.product();

        // geometry initialization
        ESHandle<CaloGeometry> geometry;
        iSetup.get<IdealGeometryRecord>().get(geometry);

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

	//tracks and PV (in case they are needed)
	iEvent.getByLabel(trackInput,tracks);
	iEvent.getByLabel(vertexInput,thePVs);
}


TLorentzVector HardTauAlgorithm::recalculateEnergy(const reco::CaloTau& jet){

	const TrackRef& leadTk = jet.leadTrack();

	const TrackRefVector associatedTracks = jet.caloTauTagInfoRef()->Tracks();

	const CaloJet* cJet = jet.caloTauTagInfoRef()->calojetRef().get();
	CaloJet caloJet = *cJet;
	caloJet.setP4(jet.p4());

	return recalculateEnergy(caloJet,leadTk,associatedTracks);
}

TLorentzVector HardTauAlgorithm::recalculateEnergy(const reco::CaloJet& caloJet){

	TrackRef leadTk;
        TrackRefVector associatedTracks;

	double ptmax = 0;

	if(tracks.isValid()) {

		//const TrackCollection tracks = *(trackHandle.product());
		TrackCollection::const_iterator iTrack;
		for(unsigned int i = 0; i < tracks->size(); ++i){
			TrackRef trackRef(tracks,i);
			double DR = ROOT::Math::VectorUtil::DeltaR(caloJet.momentum(),trackRef->momentum());
			if(DR < 0.5) associatedTracks.push_back(trackRef);
		}
	}

	Vertex thePV = *(thePVs->begin());
	TrackRefVector theFilteredTracks = TauTagTools::filteredTracks(associatedTracks,
								       tkptmin,
								       tkminPixelHitsn,
								       tkminTrackerHitsn,
								       tkmaxipt,
								       tkmaxChi2,
								       thePV);

	for(TrackRefVector::const_iterator i = theFilteredTracks.begin();
					   i!= theFilteredTracks.end(); ++i){
		double DR = ROOT::Math::VectorUtil::DeltaR(caloJet.momentum(),(*i)->momentum());
		if(DR < matchingCone && (*i)->pt() > ptmax){
                                leadTk = *i;
                                ptmax = (*i)->pt();
                }
	}

	if(ptmax > 0) return recalculateEnergy(caloJet,leadTk,theFilteredTracks);

	return TLorentzVector(0,0,0,0);
}

TLorentzVector HardTauAlgorithm::recalculateEnergy(const reco::Jet& tau){

	cout << "HardTauAlgorithm::recalculateEnergy(const reco::Jet&) "
             << "is not working. " << endl;
	cout << "Please use CaloJet or CaloTau instead. Exiting..." << endl;
	exit(0);

        const CaloJet& cJet = dynamic_cast<const CaloJet&>(tau);
	CaloJet caloJet = cJet;
        caloJet.setP4(tau.p4());

        return recalculateEnergy(caloJet);
}

TLorentzVector HardTauAlgorithm::recalculateEnergy(const reco::IsolatedTauTagInfo& tau){

	const TrackRef& leadTk = tau.leadingSignalTrack(matchingCone,tkptmin);

	const TrackRefVector associatedTracks = tau.allTracks();

        const CaloJet& cJet = dynamic_cast<const CaloJet&>(*(tau.jet()));
        CaloJet caloJet = cJet;
        caloJet.setP4(tau.jet().get()->p4());

        return recalculateEnergy(caloJet,leadTk,associatedTracks);
}

TLorentzVector HardTauAlgorithm::recalculateEnergy(const reco::CaloJet& caloJet,const TrackRef& leadTk,const TrackRefVector& associatedTracks){

        all++;

        TLorentzVector p4(0,0,0,0);

        if(leadTk->pt() == 0) return p4;

	XYZVector momentum(0,0,0);
	int prongCounter = 0;
        RefVector<TrackCollection>::const_iterator iTrack;
        for(iTrack = associatedTracks.begin(); iTrack!= associatedTracks.end(); ++iTrack){
		double DR = ROOT::Math::VectorUtil::DeltaR(leadTk->momentum(),(*iTrack)->momentum());
		if(DR < signalCone) {
			momentum+=(*iTrack)->momentum();
			prongCounter++;
		}
	}
        if(momentum.Rho() == 0) return p4;

	const TransientTrack transientTrack = transientTrackBuilder->build(*leadTk);
	XYZVector ltrackEcalHitPoint = trackEcalHitPoint(transientTrack,caloJet);
	if(! (ltrackEcalHitPoint.Rho() > 0 && ltrackEcalHitPoint.Rho() < 9999) ) return p4;

        pair<XYZVector,XYZVector> caloClusters = getClusterEnergy(caloJet,ltrackEcalHitPoint,signalCone);
	XYZVector EcalCluster = caloClusters.first;
        XYZVector HcalCluster = caloClusters.second;

        double etCaloOverTrack = (EcalCluster.Rho()+HcalCluster.Rho()-momentum.Rho())/momentum.Rho();

        pair<XYZVector,XYZVector> caloClustersPhoton = getClusterEnergy(caloJet,ltrackEcalHitPoint,ecalCone);
        XYZVector EcalClusterPhoton = caloClustersPhoton.first;
	TLorentzVector p4photons(EcalClusterPhoton.X() - EcalCluster.X(),
                                 EcalClusterPhoton.Y() - EcalCluster.Y(),
                                 EcalClusterPhoton.Z() - EcalCluster.Z(),
                                 EcalClusterPhoton.R() - EcalCluster.R());

        if( etCaloOverTrack > etCaloOverTrackMin  && etCaloOverTrack < etCaloOverTrackMax ) {
                p4.SetXYZT(momentum.X(),
                           momentum.Y(),
                           momentum.Z(),
                           momentum.R());
		p4 += p4photons;
        }
        if( etCaloOverTrack  > etCaloOverTrackMax ) {
                double etHcalOverTrack = (HcalCluster.Rho()-momentum.Rho())/momentum.Rho();

		if ( etHcalOverTrack  > etHcalOverTrackMin  && etHcalOverTrack  < etHcalOverTrackMax ) {
                  p4.SetXYZT(EcalCluster.X()   + momentum.X(),
                             EcalCluster.Y()   + momentum.Y(),
                             EcalCluster.Z()   + momentum.Z(),
                             EcalCluster.R()   + momentum.R());
		  p4 += p4photons;
                }
                if ( etHcalOverTrack  < etHcalOverTrackMin ) {
                  p4.SetXYZT(caloJet.px(),caloJet.py(),caloJet.pz(),caloJet.energy());
                }
        }

	if(p4.Et() > 0) passed++;

	return p4;
}


XYZVector HardTauAlgorithm::trackEcalHitPoint(const TransientTrack& transientTrack,const CaloJet& caloJet){


        GlobalPoint ecalHitPosition(0,0,0);

        double maxTowerEt = 0;
        vector<CaloTowerRef> towers = caloJet.getConstituents();
        for(vector<CaloTowerRef>::const_iterator iTower = towers.begin();
                                                 iTower != towers.end(); iTower++){
                if((*iTower)->et() > maxTowerEt){
                        maxTowerEt = (*iTower)->et();
                        ecalHitPosition = GlobalPoint((*iTower)->momentum().x(),
                                                      (*iTower)->momentum().y(),
                                                      (*iTower)->momentum().z());
                }
        }


        XYZVector ecalHitPoint(0,0,0);

        try{
                TrajectoryStateClosestToPoint TSCP = transientTrack.trajectoryStateClosestToPoint(ecalHitPosition);
                GlobalPoint trackEcalHitPoint = TSCP.position();

                ecalHitPoint.SetXYZ(trackEcalHitPoint.x(),
                                    trackEcalHitPoint.y(),
                                    trackEcalHitPoint.z());
        }catch(...) {;}

        return ecalHitPoint;
}

pair<XYZVector,XYZVector> HardTauAlgorithm::getClusterEnergy(const CaloJet& caloJet,XYZVector& trackEcalHitPoint,double cone){

        XYZVector ecalCluster(0,0,0);
        XYZVector hcalCluster(0,0,0);

        vector<CaloTowerRef> towers = caloJet.getConstituents();

        for(vector<CaloTowerRef>::const_iterator iTower = towers.begin();
                                                 iTower != towers.end(); iTower++){
                vector<XYZVector> ECALCells;
                vector<XYZVector> HCALCells;

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

		vector<XYZVector>::const_iterator i;
                for(i = ECALCells.begin(); i != ECALCells.end(); ++i) {
			double DR = ROOT::Math::VectorUtil::DeltaR(trackEcalHitPoint,*i);
                        if( DR < cone ) ecalCluster += *i;
                }
                for(i = HCALCells.begin(); i != HCALCells.end(); ++i) {
			double DR = ROOT::Math::VectorUtil::DeltaR(trackEcalHitPoint,*i);
                        if( DR < cone ) hcalCluster += *i;
                }
	}
        return pair<XYZVector,XYZVector> (ecalCluster,hcalCluster);
}

XYZVector HardTauAlgorithm::getCellMomentum(const CaloCellGeometry* cell,double& energy){
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
