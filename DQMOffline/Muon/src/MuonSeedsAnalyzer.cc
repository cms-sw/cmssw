
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/25 18:37:05 $
 *  $Revision: 1.2 $
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


#include <string>
using namespace std;
using namespace edm;



MuonSeedsAnalyzer::MuonSeedsAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {

  cout<<"[MuonSeedsAnalyzer] Constructor called!"<<endl;
  parameters = pSet;
  // Set the verbosity
  debug = parameters.getParameter<bool>("debug");

}


MuonSeedsAnalyzer::~MuonSeedsAnalyzer() { }


void MuonSeedsAnalyzer::beginJob(edm::EventSetup const& iSetup, DaqMonitorBEInterface * dbe) {

  cout<<"[MuonSeedsAnalyzer] Parameters initialization"<<endl;
  dbe->setCurrentFolder("Muons/MuonSeedsAnalyzer");

  seedHitBin = parameters.getParameter<int>("RecHitBin");
  seedHitMin = parameters.getParameter<double>("RecHitMin");
  seedHitMax = parameters.getParameter<double>("RecHitMax");
  string histname = "NumberOfRecHitsPerSeed_";
  NumberOfRecHitsPerSeed = dbe->book1D(histname, histname, seedHitBin, seedHitMin, seedHitMax);
  NumberOfRecHitsPerSeed ->setAxisTitle("Number of RecHits of each seed");

  PhiBin = parameters.getParameter<int>("PhiBin");
  PhiMin = parameters.getParameter<double>("PhiMin");
  PhiMax = parameters.getParameter<double>("PhiMax");
  histname = "seedPhi_";
  seedPhi = dbe->book1D(histname, histname, PhiBin, PhiMin, PhiMax);
  seedPhi->setAxisTitle("Seed azimuthal angle");
  
  EtaBin = parameters.getParameter<int>("EtaBin");
  EtaMin = parameters.getParameter<double>("EtaMin");
  EtaMax = parameters.getParameter<double>("EtaMax");
  histname = "seedEta_";
  seedEta = dbe->book1D(histname, histname, EtaBin, EtaMin, EtaMax);
  seedEta->setAxisTitle("Seed pseudorapidity");
  
  ThetaBin = parameters.getParameter<int>("ThetaBin");
  ThetaMin = parameters.getParameter<double>("ThetaMin");
  ThetaMax = parameters.getParameter<double>("ThetaMax");
  histname = "seedTheta_";
  seedTheta = dbe->book1D(histname, histname, ThetaBin, ThetaMin, ThetaMax);
  seedTheta->setAxisTitle("Seed polar angle");

  seedPtBin = parameters.getParameter<int>("seedPtBin");
  seedPtMin = parameters.getParameter<double>("seedPtMin");
  seedPtMax = parameters.getParameter<double>("seedPtMax");
  histname = "seedPt_";
  seedPt = dbe->book1D(histname, histname, seedPtBin, seedPtMin, seedPtMax);
  seedPt->setAxisTitle("Transverse seed momentum");

  seedPxBin = parameters.getParameter<int>("seedPxBin");
  seedPxMin = parameters.getParameter<double>("seedPxMin");
  seedPxMax = parameters.getParameter<double>("seedPxMax");
  histname = "seedPx_";
  seedPx = dbe->book1D(histname, histname, seedPxBin, seedPxMin, seedPxMax);
  seedPx->setAxisTitle("x component of seed momentum");

  seedPyBin = parameters.getParameter<int>("seedPyBin");
  seedPyMin = parameters.getParameter<double>("seedPyMin");
  seedPyMax = parameters.getParameter<double>("seedPyMax");
  histname = "seedPy_";
  seedPy = dbe->book1D(histname, histname, seedPyBin, seedPyMin, seedPyMax);
  seedPy->setAxisTitle("y component of seed momentum");

  seedPzBin = parameters.getParameter<int>("seedPzBin");
  seedPzMin = parameters.getParameter<double>("seedPzMin");
  seedPzMax = parameters.getParameter<double>("seedPzMax");
  histname = "seedPz_";
  seedPz = dbe->book1D(histname, histname, seedPzBin, seedPzMin, seedPzMax);
  seedPz->setAxisTitle("z component of seed momentum");

  ptErrBin = parameters.getParameter<int>("ptErrBin");
  ptErrMin = parameters.getParameter<double>("ptErrMin");
  ptErrMax = parameters.getParameter<double>("ptErrMax");
  histname = "seedPtErrOverPt_";
  seedPtErr = dbe->book1D(histname, histname, ptErrBin, ptErrMin, ptErrMax);
  seedPtErr->setAxisTitle("ptErr/pt");
  
  pxErrBin = parameters.getParameter<int>("pxErrBin");
  pxErrMin = parameters.getParameter<double>("pxErrMin");
  pxErrMax = parameters.getParameter<double>("pxErrMax");
  histname = "seedPxErrOverPx_";
  seedPxErr = dbe->book1D(histname, histname, pxErrBin, pxErrMin, pxErrMax);
  seedPxErr->setAxisTitle("pxErr/px");

  pyErrBin = parameters.getParameter<int>("pyErrBin");
  pyErrMin = parameters.getParameter<double>("pyErrMin");
  pyErrMax = parameters.getParameter<double>("pyErrMax");
  histname = "seedPyErrOverPy_";
  seedPyErr = dbe->book1D(histname, histname, pyErrBin, pyErrMin, pyErrMax);
  seedPyErr->setAxisTitle("pyErr/py");

  pzErrBin = parameters.getParameter<int>("pzErrBin");
  pzErrMin = parameters.getParameter<double>("pzErrMin");
  pzErrMax = parameters.getParameter<double>("pzErrMax");
  histname = "seedPzErrOverPz_";
  seedPzErr = dbe->book1D(histname, histname, pzErrBin, pzErrMin, pzErrMax);
  seedPzErr->setAxisTitle("pzErr/pz");

  pErrBin = parameters.getParameter<int>("pErrBin");
  pErrMin = parameters.getParameter<double>("pErrMin");
  pErrMax = parameters.getParameter<double>("pErrMax");
  histname = "seedPErrOverP_";
  seedPErr = dbe->book1D(histname, histname, pErrBin, pErrMin, pErrMax);
  seedPErr->setAxisTitle("pErr/p");

  phiErrBin = parameters.getParameter<int>("phiErrBin");
  phiErrMin = parameters.getParameter<double>("phiErrMin");
  phiErrMax = parameters.getParameter<double>("phiErrMax");
  histname = "seedPhiErr_";
  seedPhiErr = dbe->book1D(histname, histname, phiErrBin, phiErrMin, phiErrMax);
  seedPhiErr->setAxisTitle("phiErr");

  etaErrBin = parameters.getParameter<int>("etaErrBin");
  etaErrMin = parameters.getParameter<double>("etaErrMin");
  etaErrMax = parameters.getParameter<double>("etaErrMax");
  histname = "seedEtaErr_";
  seedEtaErr = dbe->book1D(histname, histname, etaErrBin, etaErrMin, etaErrMax);
  seedEtaErr->setAxisTitle("etaErr");
  

}


void MuonSeedsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrajectorySeed& seed) {

  TrajectoryStateOnSurface seedTSOS = getSeedTSOS(seed);
  AlgebraicSymMatrix66 errors = seedTSOS.cartesianError().matrix();
  double partialPterror = errors(3,3)*pow(seedTSOS.globalMomentum().x(),2) + errors(4,4)*pow(seedTSOS.globalMomentum().y(),2);

  cout<<"[MuonSeedAnalyzer] Filling the histos"<<endl;

  // nhits
  NumberOfRecHitsPerSeed->Fill(seed.nHits());
  
  // pt
  seedPt->Fill(seedTSOS.globalMomentum().perp());

  // px
  seedPx->Fill(seedTSOS.globalMomentum().x());

  // py
  seedPy->Fill(seedTSOS.globalMomentum().y());

  // pz 
  seedPz->Fill(seedTSOS.globalMomentum().z());

  // phi
  seedPhi->Fill(seedTSOS.globalMomentum().phi());

  // theta
  seedTheta->Fill(seedTSOS.globalMomentum().theta());

  // eta
  seedEta->Fill(seedTSOS.globalMomentum().eta());

  // pt err
  seedPtErr->Fill(sqrt(partialPterror)/seedTSOS.globalMomentum().perp());

  // px err
  seedPxErr->Fill(sqrt(errors(3,3))/seedTSOS.globalMomentum().x());

  // py err
  seedPyErr->Fill(sqrt(errors(4,4))/seedTSOS.globalMomentum().y());

  // pz err
  seedPzErr->Fill(sqrt(errors(5,5))/seedTSOS.globalMomentum().z());

  // p err
  seedPErr->Fill(sqrt(partialPterror+errors(5,5)*pow(seedTSOS.globalMomentum().z(),2))/seedTSOS.globalMomentum().mag());

  // phi err
  seedPhiErr->Fill(sqrt(seedTSOS.curvilinearError().matrix()(2,2)));

  // eta err
  seedEtaErr->Fill(sqrt(seedTSOS.curvilinearError().matrix()(1,1))*abs(sin(seedTSOS.globalMomentum().theta())));

}


TrajectoryStateOnSurface MuonSeedsAnalyzer::getSeedTSOS(const TrajectorySeed& seed){

  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();
  // Transform it in a TrajectoryStateOnSurface
  TrajectoryStateTransform tsTransform;
  DetId seedDetId(pTSOD.detId());
  const GeomDet* gdet = service()->trackingGeometry()->idToDet( seedDetId );
  TrajectoryStateOnSurface initialState = tsTransform.transientState(pTSOD, &(gdet->surface()), &*(service())->magneticField());

  return initialState;

}

  
