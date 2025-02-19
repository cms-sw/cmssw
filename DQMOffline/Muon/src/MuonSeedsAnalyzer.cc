
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/25
 18:37:05 $
 *  $Revision: 1.15 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonSeedsAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



MuonSeedsAnalyzer::MuonSeedsAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {
  parameters = pSet;
}


MuonSeedsAnalyzer::~MuonSeedsAnalyzer() { }


void MuonSeedsAnalyzer::beginJob(DQMStore * dbe) {

  metname = "seedsAnalyzer";

  LogTrace(metname)<<"[MuonSeedsAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/MuonSeedsAnalyzer");

  seedHitBin = parameters.getParameter<int>("RecHitBin");
  seedHitMin = parameters.getParameter<double>("RecHitMin");
  seedHitMax = parameters.getParameter<double>("RecHitMax");
  string histname = "NumberOfRecHitsPerSeed_";
  NumberOfRecHitsPerSeed = dbe->book1D(histname, "Number of seed recHits", seedHitBin, seedHitMin, seedHitMax);
 
  PhiBin = parameters.getParameter<int>("PhiBin");
  PhiMin = parameters.getParameter<double>("PhiMin");
  PhiMax = parameters.getParameter<double>("PhiMax");
  histname = "seedPhi_";
  seedPhi = dbe->book1D(histname, "Seed #phi", PhiBin, PhiMin, PhiMax);
  seedPhi->setAxisTitle("rad");
   
  EtaBin = parameters.getParameter<int>("EtaBin");
  EtaMin = parameters.getParameter<double>("EtaMin");
  EtaMax = parameters.getParameter<double>("EtaMax");
  histname = "seedEta_";
  seedEta = dbe->book1D(histname, "Seed #eta", EtaBin, EtaMin, EtaMax);
    
  ThetaBin = parameters.getParameter<int>("ThetaBin");
  ThetaMin = parameters.getParameter<double>("ThetaMin");
  ThetaMax = parameters.getParameter<double>("ThetaMax");
  histname = "seedTheta_";
  seedTheta = dbe->book1D(histname, "Seed #theta", ThetaBin, ThetaMin, ThetaMax);
  seedTheta->setAxisTitle("rad");
 
  seedPtBin = parameters.getParameter<int>("seedPtBin");
  seedPtMin = parameters.getParameter<double>("seedPtMin");
  seedPtMax = parameters.getParameter<double>("seedPtMax");
  histname = "seedPt_";
  seedPt = dbe->book1D(histname, "Seed p_{t}", seedPtBin, seedPtMin, seedPtMax);
  seedPt->setAxisTitle("GeV");

  seedPxyzBin = parameters.getParameter<int>("seedPxyzBin");
  seedPxyzMin = parameters.getParameter<double>("seedPxyzMin");
  seedPxyzMax = parameters.getParameter<double>("seedPxyzMax");
  histname = "seedPx_";
  seedPx = dbe->book1D(histname, "Seed p_{x}", seedPxyzBin, seedPxyzMin, seedPxyzMax);
  seedPx->setAxisTitle("GeV");
  histname = "seedPy_";
  seedPy = dbe->book1D(histname, "Seed p_{y}", seedPxyzBin, seedPxyzMin, seedPxyzMax);
  seedPy->setAxisTitle("GeV");
  histname = "seedPz_";
  seedPz = dbe->book1D(histname, "Seed p_{z}", seedPxyzBin, seedPxyzMin, seedPxyzMax);
  seedPz->setAxisTitle("GeV");

  pErrBin = parameters.getParameter<int>("pErrBin");
  pErrMin = parameters.getParameter<double>("pErrMin");
  pErrMax = parameters.getParameter<double>("pErrMax");
  histname = "seedPtErrOverPt_";
  seedPtErr = dbe->book1D(histname, "Seed p_{t}Err/p_{t}", pErrBin, pErrMin, pErrMax);
  histname = "seedPtErrOverPtVsPhi_";
  seedPtErrVsPhi = dbe->book2D(histname, "Seed p_{t}Err/p_{t} vs #phi", PhiBin, PhiMin, PhiMax, pErrBin, pErrMin, pErrMax);
  seedPtErrVsPhi->setAxisTitle("rad",2);
  histname = "seedPtErrOverPtVsEta_";
  seedPtErrVsEta = dbe->book2D(histname, "Seed p_{t}Err/p_{t} vs #eta", EtaBin, EtaMin, EtaMax, pErrBin, pErrMin, pErrMax);
  histname = "seedPtErrOverPtVsPt_";
  seedPtErrVsPt = dbe->book2D(histname, "Seed p_{t}Err/p_{t} vs p_{t}", seedPtBin/5, seedPtMin, seedPtMax, pErrBin, pErrMin, pErrMax);
  seedPtErrVsPt->setAxisTitle("GeV",2);
  histname = "seedPErrOverP_";
  seedPErr = dbe->book1D(histname, "Seed pErr/p", pErrBin, pErrMin, pErrMax);
  histname = "seedPErrOverPVsPhi_";
  seedPErrVsPhi = dbe->book2D(histname, "Seed pErr/p vs #phi", PhiBin, PhiMin, PhiMax, pErrBin, pErrMin, pErrMax);
  seedPErrVsPhi->setAxisTitle("rad",2);
  histname = "seedPErrOverPVsEta_";
  seedPErrVsEta = dbe->book2D(histname, "Seed pErr/p vs #eta", EtaBin, EtaMin, EtaMax, pErrBin, pErrMin, pErrMax);
  histname = "seedPErrOverPVsPt_";
  seedPErrVsPt = dbe->book2D(histname, "Seed pErr/p vs p_{t}", seedPtBin/5, seedPtMin, seedPtMax, pErrBin, pErrMin, pErrMax);
  seedPErrVsPt->setAxisTitle("GeV",2);
  
  pxyzErrBin = parameters.getParameter<int>("pxyzErrBin");
  pxyzErrMin = parameters.getParameter<double>("pxyzErrMin");
  pxyzErrMax = parameters.getParameter<double>("pxyzErrMax");
  histname = "seedPxErrOverPx_";
  seedPxErr = dbe->book1D(histname, "Seed p_{x}Err/p_{x}", pxyzErrBin, pxyzErrMin, pxyzErrMax);
  histname = "seedPyErrOverPy_";
  seedPyErr = dbe->book1D(histname, "Seed p_{y}Err/p_{y}", pxyzErrBin, pxyzErrMin, pxyzErrMax);
   histname = "seedPzErrOverPz_";
  seedPzErr = dbe->book1D(histname, "Seed p_{z}Err/p_{z}", pxyzErrBin, pxyzErrMin, pxyzErrMax);
 
  phiErrBin = parameters.getParameter<int>("phiErrBin");
  phiErrMin = parameters.getParameter<double>("phiErrMin");
  phiErrMax = parameters.getParameter<double>("phiErrMax");
  histname = "seedPhiErr_";
  seedPhiErr = dbe->book1D(histname, "Seed #phi error", phiErrBin, phiErrMin, phiErrMax);

  etaErrBin = parameters.getParameter<int>("etaErrBin");
  etaErrMin = parameters.getParameter<double>("etaErrMin");
  etaErrMax = parameters.getParameter<double>("etaErrMax");
  histname = "seedEtaErr_";
  seedEtaErr = dbe->book1D(histname, "Seed #eta error", etaErrBin, etaErrMin, etaErrMax);

}


void MuonSeedsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrajectorySeed& seed) {

  TrajectoryStateOnSurface seedTSOS = getSeedTSOS(seed);
  AlgebraicSymMatrix66 errors = seedTSOS.cartesianError().matrix();
  double partialPterror = errors(3,3)*pow(seedTSOS.globalMomentum().x(),2) + errors(4,4)*pow(seedTSOS.globalMomentum().y(),2);

  LogTrace(metname)<<"[MuonSeedAnalyzer] Filling the histos";

  // nhits
  LogTrace(metname)<<"Number od recHits per seed: "<<seed.nHits();
  NumberOfRecHitsPerSeed->Fill(seed.nHits());
  
  // pt
  LogTrace(metname)<<"seed momentum: "<<seedTSOS.globalMomentum().perp();
  seedPt->Fill(seedTSOS.globalMomentum().perp());

  // px
  LogTrace(metname)<<"seed px: "<<seedTSOS.globalMomentum().x();
  seedPx->Fill(seedTSOS.globalMomentum().x());

  // py
  LogTrace(metname)<<"seed py: "<<seedTSOS.globalMomentum().y();
  seedPy->Fill(seedTSOS.globalMomentum().y());

  // pz 
  LogTrace(metname)<<"seed pz: "<<seedTSOS.globalMomentum().z();
  seedPz->Fill(seedTSOS.globalMomentum().z());

  // phi
  LogTrace(metname)<<"seed phi: "<<seedTSOS.globalMomentum().phi();
  seedPhi->Fill(seedTSOS.globalMomentum().phi());

  // theta
  LogTrace(metname)<<"seed theta: "<<seedTSOS.globalMomentum().theta();
  seedTheta->Fill(seedTSOS.globalMomentum().theta());

  // eta
  LogTrace(metname)<<"seed eta: "<<seedTSOS.globalMomentum().eta();
  seedEta->Fill(seedTSOS.globalMomentum().eta());

  // pt err
  LogTrace(metname)<<"seed pt error: "<<sqrt(partialPterror)/seedTSOS.globalMomentum().perp();
  seedPtErr->Fill(sqrt(partialPterror)/seedTSOS.globalMomentum().perp());

  // ptErr/pt Vs phi
  seedPtErrVsPhi->Fill(seedTSOS.globalMomentum().phi(),
		       sqrt(partialPterror)/seedTSOS.globalMomentum().perp());
  // ptErr/pt Vs eta
  seedPtErrVsEta->Fill(seedTSOS.globalMomentum().eta(),
		       sqrt(partialPterror)/seedTSOS.globalMomentum().perp());
  // ptErr/pt Vs pt
  seedPtErrVsPt->Fill(seedTSOS.globalMomentum().perp(),
		       sqrt(partialPterror)/seedTSOS.globalMomentum().perp());

  // px err
  LogTrace(metname)<<"seed px error: "<<sqrt(errors(3,3))/seedTSOS.globalMomentum().x();
  seedPxErr->Fill(sqrt(errors(3,3))/seedTSOS.globalMomentum().x());
  
  // py err
  LogTrace(metname)<<"seed py error: "<<sqrt(errors(4,4))/seedTSOS.globalMomentum().y();
  seedPyErr->Fill(sqrt(errors(4,4))/seedTSOS.globalMomentum().y());

  // pz err
  LogTrace(metname)<<"seed pz error: "<<sqrt(errors(5,5))/seedTSOS.globalMomentum().z();
  seedPzErr->Fill(sqrt(errors(5,5))/seedTSOS.globalMomentum().z());

  // p err
  LogTrace(metname)<<"seed p error: "<<sqrt(partialPterror+errors(5,5)*pow(seedTSOS.globalMomentum().z(),2))/seedTSOS.globalMomentum().mag();
  seedPErr->Fill(sqrt(partialPterror+errors(5,5)*pow(seedTSOS.globalMomentum().z(),2))/seedTSOS.globalMomentum().mag());

  // pErr/p Vs phi
  seedPErrVsPhi->Fill(seedTSOS.globalMomentum().phi(),
		      sqrt(partialPterror+errors(5,5)*pow(seedTSOS.globalMomentum().z(),2))/seedTSOS.globalMomentum().mag());
   // pErr/p Vs eta
  seedPErrVsEta->Fill(seedTSOS.globalMomentum().eta(),
		      sqrt(partialPterror+errors(5,5)*pow(seedTSOS.globalMomentum().z(),2))/seedTSOS.globalMomentum().mag());
  // pErr/p Vs pt
  seedPErrVsPt->Fill(seedTSOS.globalMomentum().perp(),
		     sqrt(partialPterror+errors(5,5)*pow(seedTSOS.globalMomentum().z(),2))/seedTSOS.globalMomentum().mag());

  // phi err
  LogTrace(metname)<<"seed phi error: "<<sqrt(seedTSOS.curvilinearError().matrix()(2,2));
  seedPhiErr->Fill(sqrt(seedTSOS.curvilinearError().matrix()(2,2)));

  // eta err
  LogTrace(metname)<<"seed eta error: "<<sqrt(seedTSOS.curvilinearError().matrix()(1,1))*abs(sin(seedTSOS.globalMomentum().theta()));
  seedEtaErr->Fill(sqrt(seedTSOS.curvilinearError().matrix()(1,1))*abs(sin(seedTSOS.globalMomentum().theta())));
  
}


TrajectoryStateOnSurface MuonSeedsAnalyzer::getSeedTSOS(const TrajectorySeed& seed){

  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();
  // Transform it in a TrajectoryStateOnSurface
  
  DetId seedDetId(pTSOD.detId());
  const GeomDet* gdet = service()->trackingGeometry()->idToDet( seedDetId );
  TrajectoryStateOnSurface initialState = trajectoryStateTransform::transientState(pTSOD, &(gdet->surface()), &*(service())->magneticField());

  return initialState;

}

  
