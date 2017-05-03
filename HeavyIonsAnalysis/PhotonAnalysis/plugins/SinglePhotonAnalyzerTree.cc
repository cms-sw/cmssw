
/** \class SinglePhotonAnalyzerTree  CmsHi/PhotonAnalysis/plugins/SinglePhotonAnalyzerTree.cc
 *
 * Description:
 * Analysis code of the QCD Photons group;
 * Data Analyzer for the single photon (inclusive isolated photons and photon plus jets) cross section measurement;
 * Makes ntuple of the leading photon and array of jets in each event;
 * Store in ntuple run conditions and missing ET information for each event;
 * Store in ntuple trigger and vertexing information for each event;
 * Makes ntuple for the generator prompt photon kinematics and genealogy;
 * Perform the MC truth matching for the reconstructed photon;
 * Fill number of DQM histograms
 *
 * \author Serguei Ganjour,     CEA-Saclay/IRFU, FR
 * \author Ted Ritchie Kolberg, University of Notre Dame, US
 * \author Michael B. Anderson, University of Wisconsin Madison, US
 * \author Laurent Millischer,  CEA-Saclay/IRFU, FR
 * \author Vasundhara Chetluru, FNAL, US
 * \author Vladimir Litvin,     Caltech, US
 * \author Yen-Jie Lee,         MIT, US
 * \author Abraham DeBenedetti, University of Minnesota, US
 * \author Pasquale Musella,    LIP, PT
 * \author Shin-Shan Eiko Yu,   National Central University, TW
 * \author Rong-Shyang Lu,      National Taiwan University, TW
 *
 * \version $Id: SinglePhotonAnalyzerTree.cc,v 1.10 2011/10/25 13:37:15 yjlee Exp $
 *
 */
// This was modified to fit with Heavy Ion collsion by Yongsun Kim ( MIT)

#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/SinglePhotonAnalyzerTree.h"

#include <memory>
#include <iostream>
#include <algorithm>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"

//Trigger DataFormats
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTrigger/plugins/L1GlobalTrigger.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CommonTools/Utils/interface/PtComparator.h"
//#include "RecoEgamma/EgammaTools/interface/ConversionLikelihoodCalculator.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

// for conversion truth
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"



// Histograms, ntuples not used anymore!!
//#include "UserCode/HafHistogram/interface/HTupleManager.h"
//#include "UserCode/HafHistogram/interface/HHistogram.h"
//#include "UserCode/HafHistogram/interface/HTuple.h"

//ROOT includes
#include <Math/VectorUtil.h>
#include <TLorentzVector.h>

using namespace pat;
using namespace edm;
using namespace std;
using namespace ROOT::Math::VectorUtil;

bool operator < (const TLorentzVector & a, const TLorentzVector & b) { return a.Pt() < b.Pt(); }
bool operator < (const TVector3 & a,       const TVector3 & b      ) { return a.Pt() < b.Pt(); }


SinglePhotonAnalyzerTree::SinglePhotonAnalyzerTree(const edm::ParameterSet& ps)
{

  verbose_                         = ps.getUntrackedParameter<bool>("verbose", false);
  fillMCNTuple_                    = ps.getUntrackedParameter<bool>("FillMCNTuple", true);
  doL1Objects_                     = ps.getUntrackedParameter<bool>("DoL1Objects",  false);
  isMC_                            = kFALSE;//Set by checking if generator block is valid
  storePhysVectors_                = ps.getUntrackedParameter<bool>("StorePhysVectors",  false);
  outputFile_                      = ps.getParameter<string>("OutputFile");

  hlTriggerResults_                = ps.getParameter<InputTag>("HltTriggerResults");
  l1gtReadout_                     = ps.getParameter<InputTag>("L1gtReadout");
  l1IsolTag_                       = ps.getParameter<InputTag>("L1IsolTag");
  l1NonIsolTag_                    = ps.getParameter<InputTag>("L1NonIsolTag");
  triggerPathsToStore_             = ps.getParameter<vector<string> >("TriggerPathsToStore");

  genParticleProducer_             = ps.getParameter<InputTag>("GenParticleProducer");
  hepMCProducer_                   = ps.getParameter<InputTag>("HepMCProducer");
  genEventScale_                   = ps.getParameter<InputTag>("GenEventScale");
  photonProducer_                  = ps.getParameter<InputTag>("PhotonProducer");
  compPhotonProducer_              = ps.getParameter<InputTag>("compPhotonProducer");

  trackProducer_                   = ps.getParameter<InputTag>("TrackProducer");
  jetProducer_                     = ps.getParameter<InputTag>("JetProducer");
  metProducer_                     = ps.getParameter<InputTag>("METProducer");
  vertexProducer_                  = ps.getParameter<InputTag>("VertexProducer");
  beamSpotProducer_                = ps.getParameter<edm::InputTag>("BeamSpotProducer");

  ebReducedRecHitCollection_       = consumes<EcalRecHitCollection>( ps.getParameter<edm::InputTag>("ebReducedRecHitCollection"));
  eeReducedRecHitCollection_       = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("eeReducedRecHitCollection"));
  srcTowers_                       = ps.getParameter<edm::InputTag>("caloTowers");
  //heavy ion
  basicClusterBarrel_              = ps.getParameter<edm::InputTag>("basicClusterBarrel");
  basicClusterEndcap_              = ps.getParameter<edm::InputTag>("basicClusterEndcap");
  hbhe_                            = ps.getParameter<edm::InputTag>("hbhe");
  hf_                              = ps.getParameter<edm::InputTag>("hf");
  ho_                              = ps.getParameter<edm::InputTag>("ho");



  //event plance
  evtPlaneLabel                    =  ps.getParameter<edm::InputTag>("hiEvtPlane_");

  // pf candidate and voronoi
  pfCandidateLabel_                 = ps.getParameter<edm::InputTag>("pfCandidateLabel");
  srcPfVor_ =                          ps.getParameter<edm::InputTag>("voronoiBkg");

  // tower hits and voronoi
  towerCandidateLabel_               = ps.getParameter<edm::InputTag>("towerCandidateLabel");
  srcTowerVor_ =                       ps.getParameter<edm::InputTag>("towerVoronoiBkg");

  // for July exercise
  isMC_                        = ps.getUntrackedParameter<bool>("isMC_",false);

  ptMin_                           = ps.getUntrackedParameter<double>("GammaPtMin", 15);
  etaMax_                          = ps.getUntrackedParameter<double>("GammaEtaMax",3);
  ecalBarrelMaxEta_                = ps.getUntrackedParameter<double>("EcalBarrelMaxEta",1.45);
  ecalEndcapMinEta_                = ps.getUntrackedParameter<double>("EcalEndcapMinEta",1.55);

  ptJetMin_                        = ps.getUntrackedParameter<double>("JetPtMin", 20);
  ptTrackMin_                      = ps.getUntrackedParameter<double>("TrackPtMin",1.5);
  etaTrackMax_                     = ps.getUntrackedParameter<double>("TrackEtaMax",1.75);

  pdgId_                           = ps.getUntrackedParameter<int>("pdgId", 22);
  otherPdgIds_                     = ps.getUntrackedParameter<vector<int> >("OtherPdgIds", vector<int>(1,11) );
  mcPtMin_                         = ps.getUntrackedParameter<double>("McPtMin", 15);
  mcEtaMax_                        = ps.getUntrackedParameter<double>("McEtaMax",1.7);

  etCutGenMatch_                   = ps.getUntrackedParameter<double>("etCutGenMatch",13);
  etaCutGenMatch_                  = ps.getUntrackedParameter<double>("etaCutGenMatch",3);

  doStoreL1Trigger_                = ps.getUntrackedParameter<bool>("doStoreL1Trigger",false);
  doStoreHLT_                      = ps.getUntrackedParameter<bool>("doStoreHLT",false);
  doStoreHF_                       = ps.getUntrackedParameter<bool>("doStoreHF",false);
  doStoreVertex_                   = ps.getUntrackedParameter<bool>("doStoreVertex",false);
  doStoreMET_                      = ps.getUntrackedParameter<bool>("doStoreMET",false);
  doStoreJets_                     = ps.getUntrackedParameter<bool>("doStoreJets",false);
  doStoreCompCone_                 = ps.getUntrackedParameter<bool>("doStoreCompCone",false);
  doStoreConversions_              = ps.getUntrackedParameter<bool>("doStoreConversions",false);

  doStoreTracks_                   = ps.getUntrackedParameter<bool>("doStoreTracks",false);

  // electorn collection
  EleTag_                          = ps.getUntrackedParameter<edm::InputTag>("gsfElectronCollection");

  trackQuality_                    = ps.getUntrackedParameter<string>("trackQuality","highPurity");


  // book ntuples; columns are defined dynamically later
  // tplmgr = new HTupleManager(outputFile_.c_str(),"RECREATE");

  // tplmgr->SetDir("1D-Spectra");
  //  _ptHist    = tplmgr->MomentumHistogram("GenPt"  ,"p_{T} MC photon (GeV/c);p_{T} (GeV/c)",100,0,50);
  // _ptHatHist = tplmgr->MomentumHistogram("GenPtHat"  ,"p_{T} Hat MC Events (GeV/c);p_{T} (GeV/c)",500,0,500);

  //  _etaHist   = tplmgr->MomentumHistogram("GenEta" ,"#eta MC photon;#eta"  ,100,-3,3);
  //  _vtxX      = tplmgr->MomentumHistogram("GenVtxX","Generated Vertex X"   ,100, 0.01,0.06);
  // _vtxY      = tplmgr->MomentumHistogram("GenVtxY","Generated Vertex Y"   ,100,-0.02,0.02);
  //  _vtxZ      = tplmgr->MomentumHistogram("GenVtxZ","Generated Vertex X"   ,100,-10,10);
  //  _gammaPtHist  = tplmgr->MomentumHistogram("GammaPt" ,"p_{T} leading photon candidate (GeV/c);p_{T} (GeV/c)",100,0,50);
  //  _gammaEtaHist = tplmgr->MomentumHistogram("GammaEta","#eta leading photon candidate;#eta"        ,100,-3,3);
  // note 0.1745329 = 2*pi/360 (there are 360 ecal crystals in circle in phi)
  //  _gammaPhiModHist=tplmgr->MomentumHistogram("GammaPhiMod","#phi_{mod} leading photon candidate (Barrel only);#phi_{mod}" , 42, (-1.-1./20)*0.1745329, (1.+1./20.)*0.1745329);
  //  _metHist      = tplmgr->MomentumHistogram("MET"     ,"MET (GeV);MET (GeV)"                      ,100,0,100);
  //  _nVtxHist     = tplmgr->Histogram("NumVtx",20,0,20);
  //  _primVtxX     = tplmgr->MomentumHistogram("PrimVtxX","Primary Vertex X"   ,100, 0.01,0.06);
  // _primVtxY     = tplmgr->MomentumHistogram("PrimVtxY","Primary Vertex Y"   ,100,-0.02,0.02);
  // _primVtxZ     = tplmgr->MomentumHistogram("PrimVtxZ","Primary Vertex Z"   ,100,-10,10);

  // _nPhotonsHist   = tplmgr->Histogram("NumPhotons",10,0,10);
  // _nJetsHist    = tplmgr->Histogram("NumJets",20,-0.5,19.5);

  // tplmgr->SetDir("NTuples");
  //  _ntuple     = tplmgr->Ntuple("Analysis");
  //  _ntupleMC   = tplmgr->Ntuple("Generator");

  //#if MPA_VERSION < 2
  // theLikelihoodCalc_ = new ConversionLikelihoodCalculator();
  // edm::FileInPath path_mvaWeightFile("RecoEgamma/EgammaTools/data/TMVAnalysis_Likelihood.weights.txt");
  //  theLikelihoodCalc_->setWeightsFile(path_mvaWeightFile.fullPath().c_str());
  //  #endif



}

SinglePhotonAnalyzerTree::~SinglePhotonAnalyzerTree() {

  //#if MPA_VERSION < 2
  //  delete theLikelihoodCalc_;
  //#endif

}

void SinglePhotonAnalyzerTree::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {

  analyzeMC(e,iSetup);

}

void SinglePhotonAnalyzerTree::beginJob() {
  theTree  = fs->make<TTree>("photon","v1");
  int run;
  //int evt;
  int bunchCrossing;
  int luminosityBlock;
  theTree->Branch("run",&run,"run/I");
  theTree->Branch("event",&event,"event/I");
  theTree->Branch("bunchCrossing",&bunchCrossing,"bunchCrossing/I");
  theTree->Branch("luminosityBlock",&luminosityBlock,"luminosityBlock/I");

  theTree->Branch("nPhotons",&nPho,"nPhotons/I");
  theTree->Branch("pt",pt,"pt[nPhotons]/F");
  theTree->Branch("energy",energy,"energy[nPhotons]/F");
  theTree->Branch("rawEnergy",rawEnergy,"rawEnergy[nPhotons]/F");
/*
  theTree->Branch("px",px,"px[nPhotons]/F");
  theTree->Branch("py",py,"py[nPhotons]/F");
  theTree->Branch("pz",pz,"pz[nPhotons]/F");
*/
  theTree->Branch("eta",eta,"eta[nPhotons]/F");
  theTree->Branch("phi",phi,"phi[nPhotons]/F");
  theTree->Branch("r9",r9,"r9[nPhotons]/F");
  theTree->Branch("isEB",isEB,"isEB[nPhotons]/O");
  theTree->Branch("isEBGap",isEBGap,"isEBGap[nPhotons]/O");
  theTree->Branch("isEEGap",isEEGap,"isEEGap[nPhotons]/O");
  theTree->Branch("isEBEEGap",isEBEEGap,"isEBEEGap[nPhotons]/O");
  theTree->Branch("isTransGap",isTransGap,"isTransGap[nPhotons]/O");
  theTree->Branch("preshowerEnergy",preshowerEnergy,"preshowerEnergy[nPhotons]/F");
  theTree->Branch("numOfPreshClusters",numOfPreshClusters,"numOfPreshClusters[nPhotons]/F");
  theTree->Branch("ESRatio",ESRatio,"ESRatio[nPhotons]/F");
  theTree->Branch("clustersSize",clustersSize,"clustersSize[nPhotons]/F");
  theTree->Branch("scSize",scSize,"scSize[nPhotons]/F");
  theTree->Branch("phiWidth",phiWidth,"phiWidth[nPhotons]/F");
  theTree->Branch("etaWidth",etaWidth,"etaWidth[nPhotons]/F");
  theTree->Branch("scEta",scEta,"scEta[nPhotons]/F");
  theTree->Branch("scPhi",scPhi,"scPhi[nPhotons]/F");
  theTree->Branch("sigmaEtaEta",sigmaEtaEta,"sigmaEtaEta[nPhotons]/F");
  theTree->Branch("sigmaIetaIeta",sigmaIetaIeta,"sigmaIetaIeta[nPhotons]/F");
  theTree->Branch("sigmaIphiIphi",sigmaIphiIphi,"sigmaIphiIphi[nPhotons]/F");
  theTree->Branch("sieie50",sieie50,"sieie50[nPhotons]/F");
  theTree->Branch("sieie45",sieie45,"sieie45[nPhotons]/F");
  theTree->Branch("sieie47",sieie47,"sieie47[nPhotons]/F");
  theTree->Branch("sieie42",sieie42,"sieie42[nPhotons]/F");
  theTree->Branch("sieie39",sieie39,"sieie39[nPhotons]/F");
  theTree->Branch("covPhiPhi",covPhiPhi,"covPhiPhi[nPhotons]/F");
  theTree->Branch("covEtaPhi",covEtaPhi,"covEtaPhi[nPhotons]/F");
  theTree->Branch("covEtaEta",covEtaEta,"covEtaEta[nPhotons]/F");
  theTree->Branch("r1x5",r1x5,"r1x5[nPhotons]/F");
  theTree->Branch("r2x5",r2x5,"r2x5[nPhotons]/F");
  theTree->Branch("e1x5",e1x5,"e1x5[nPhotons]/F");
  theTree->Branch("e2x5",e2x5,"e2x5[nPhotons]/F");
  theTree->Branch("eMax",eMax,"eMax[nPhotons]/F");
  theTree->Branch("e2nd",e2nd,"e2nd[nPhotons]/F");
  theTree->Branch("e2x2",e2x2,"e2x2[nPhotons]/F");
  theTree->Branch("e3x3",e3x3,"e3x3[nPhotons]/F");
  theTree->Branch("e3x2",e3x2,"e3x2[nPhotons]/F");
  theTree->Branch("e4x4",e4x4,"e4x4[nPhotons]/F");
  theTree->Branch("e5x5",e5x5,"e5x5[nPhotons]/F");
  theTree->Branch("e2overe8",e2overe8,"e2overe8[nPhotons]/F");
  theTree->Branch("eRight",eRight,"eRight[nPhotons]/F");
  theTree->Branch("eLeft",eLeft,"eLeft[nPhotons]/F");
  theTree->Branch("eTop",eTop,"eTop[nPhotons]/F");
  theTree->Branch("eBottom",eBottom,"eBottom[nPhotons]/F");
  theTree->Branch("e2x5Right",e2x5Right,"e2x5Right[nPhotons]/F");
  theTree->Branch("e2x5Left",e2x5Left,"e2x5Left[nPhotons]/F");
  theTree->Branch("e2x5Top",e2x5Top,"e2x5Top[nPhotons]/F");
  theTree->Branch("e2x5Bottom",e2x5Bottom,"e2x5Bottom[nPhotons]/F");
  theTree->Branch("seedTime",seedTime,"seedTime[nPhotons]/F");
  theTree->Branch("seedChi2",seedChi2,"seedChi2[nPhotons]/F");
  theTree->Branch("seedOutOfTimeChi2",seedOutOfTimeChi2,"seedOutOfTimeChi2[nPhotons]/F");
  theTree->Branch("seedRecoFlag",seedRecoFlag,"seedRecoFlag[nPhotons]/F");
  theTree->Branch("seedSeverity",seedSeverity,"seedSeverity[nPhotons]/F");
//   theTree->Branch("tRight",tRight,"tRight[nPhotons]/F");
//   theTree->Branch("tLeft",tLeft,"tLeft[nPhotons]/F");
//   theTree->Branch("tTop",tTop,"tTop[nPhotons]/F");
//   theTree->Branch("tBottom",tBottom,"tBottom[nPhotons]/F");
  theTree->Branch("swissCrx",swissCrx,"swissCrx[nPhotons]/F");
  theTree->Branch("hadronicOverEm",hadronicOverEm,"hadronicOverEm[nPhotons]/F");
  theTree->Branch("hadronicDepth1OverEm",hadronicDepth1OverEm,"hadronicDepth1OverEm[nPhotons]/F");
  theTree->Branch("hadronicDepth2OverEm",hadronicDepth2OverEm,"hadronicDepth2OverEm[nPhotons]/F");
  theTree->Branch("ecalRecHitSumEtConeDR04",ecalRecHitSumEtConeDR04,"ecalRecHitSumEtConeDR04[nPhotons]/F");
  theTree->Branch("hcalTowerSumEtConeDR04",hcalTowerSumEtConeDR04,"hcalTowerSumEtConeDR04[nPhotons]/F");
  theTree->Branch("hcalDepth1TowerSumEtConeDR04",hcalDepth1TowerSumEtConeDR04,"hcalDepth1TowerSumEtConeDR04[nPhotons]/F");
  theTree->Branch("hcalDepth2TowerSumEtConeDR04",hcalDepth2TowerSumEtConeDR04,"hcalDepth2TowerSumEtConeDR04[nPhotons]/F");
  theTree->Branch("trkSumPtHollowConeDR04",trkSumPtHollowConeDR04,"trkSumPtHollowConeDR04[nPhotons]/F");
  theTree->Branch("trkSumPtSolidConeDR04",trkSumPtSolidConeDR04,"trkSumPtSolidConeDR04[nPhotons]/F");

  theTree->Branch("ecalRecHitSumEtConeDR03",ecalRecHitSumEtConeDR03,"ecalRecHitSumEtConeDR03[nPhotons]/F");
  theTree->Branch("hcalTowerSumEtConeDR03",hcalTowerSumEtConeDR03,"hcalTowerSumEtConeDR03[nPhotons]/F");
  theTree->Branch("hcalDepth1TowerSumEtConeDR03",hcalDepth1TowerSumEtConeDR03,"hcalDepth1TowerSumEtConeDR03[nPhotons]/F");
  theTree->Branch("hcalDepth2TowerSumEtConeDR03",hcalDepth2TowerSumEtConeDR03,"hcalDepth2TowerSumEtConeDR03[nPhotons]/F");
  theTree->Branch("trkSumPtHollowConeDR03",trkSumPtHollowConeDR03,"trkSumPtHollowConeDR03[nPhotons]/F");
  theTree->Branch("trkSumPtSolidConeDR03",trkSumPtSolidConeDR03,"trkSumPtSolidConeDR03[nPhotons]/F");

  theTree->Branch("isEle",isEle,"isEle[nPhotons]/I");
  theTree->Branch("hasPixelSeed",hasPixelSeed,"hasPixelSeed[nPhotons]/I");

  theTree->Branch("detaEle",detaEle,"detaEle[nPhotons]/F");
  theTree->Branch("dphiEle",dphiEle,"dphiEle[nPhotons]/F");
  theTree->Branch("eleCharge",eleCharge,"eleCharge[nPhotons]/F");
  theTree->Branch("eleEoverP",eleEoverP,"eleEoverP[nPhotons]/F");

  theTree->Branch("c1",c1,"c1[nPhotons]/F");
  theTree->Branch("c2",c2,"c2[nPhotons]/F");
  theTree->Branch("c3",c3,"c3[nPhotons]/F");
  theTree->Branch("c4",c4,"c4[nPhotons]/F");
  theTree->Branch("c5",c5,"c5[nPhotons]/F");
  theTree->Branch("r1",r1,"r1[nPhotons]/F");
  theTree->Branch("r2",r2,"r2[nPhotons]/F");
  theTree->Branch("r3",r3,"r3[nPhotons]/F");
  theTree->Branch("r4",r4,"r4[nPhotons]/F");
  theTree->Branch("r5",r5,"r5[nPhotons]/F");
  theTree->Branch("t1",t1,"t1[nPhotons]/F");
  theTree->Branch("t2",t2,"t2[nPhotons]/F");
  theTree->Branch("t3",t3,"t3[nPhotons]/F");
  theTree->Branch("t4",t4,"t4[nPhotons]/F");
  theTree->Branch("t5",t5,"t5[nPhotons]/F");

  theTree->Branch("t1PtCut",t1PtCut,"t1PtCut[nPhotons]/F");
  theTree->Branch("t2PtCut",t2PtCut,"t2PtCut[nPhotons]/F");
  theTree->Branch("t3PtCut",t3PtCut,"t3PtCut[nPhotons]/F");
  theTree->Branch("t4PtCut",t4PtCut,"t4PtCut[nPhotons]/F");
  theTree->Branch("t5PtCut",t5PtCut,"t5PtCut[nPhotons]/F");
  theTree->Branch("cc1",cc1,"cc1[nPhotons]/F");
  theTree->Branch("cc2",cc2,"cc2[nPhotons]/F");
  theTree->Branch("cc3",cc3,"cc3[nPhotons]/F");
  theTree->Branch("cc4",cc4,"cc4[nPhotons]/F");
  theTree->Branch("cc4j",cc4j,"cc4j[nPhotons]/F");
  theTree->Branch("cc5",cc5,"cc5[nPhotons]/F");
  theTree->Branch("cc05",cc05,"cc05[nPhotons]/F");

  theTree->Branch("pfcIso1",pfcIso1,"pfcIso1[nPhotons]/F");
  theTree->Branch("pfcIso2",pfcIso2,"pfcIso2[nPhotons]/F");
  theTree->Branch("pfcIso3",pfcIso3,"pfcIso3[nPhotons]/F");
  theTree->Branch("pfcIso4",pfcIso4,"pfcIso4[nPhotons]/F");
  theTree->Branch("pfcIso5",pfcIso5,"pfcIso5[nPhotons]/F");

  theTree->Branch("pfpIso1",pfpIso1,"pfpIso1[nPhotons]/F");
  theTree->Branch("pfpIso2",pfpIso2,"pfpIso2[nPhotons]/F");
  theTree->Branch("pfpIso3",pfpIso3,"pfpIso3[nPhotons]/F");
  theTree->Branch("pfpIso4",pfpIso4,"pfpIso4[nPhotons]/F");
  theTree->Branch("pfpIso5",pfpIso5,"pfpIso5[nPhotons]/F");

  theTree->Branch("pfnIso1",pfnIso1,"pfnIso1[nPhotons]/F");
  theTree->Branch("pfnIso2",pfnIso2,"pfnIso2[nPhotons]/F");
  theTree->Branch("pfnIso3",pfnIso3,"pfnIso3[nPhotons]/F");
  theTree->Branch("pfnIso4",pfnIso4,"pfnIso4[nPhotons]/F");
  theTree->Branch("pfnIso5",pfnIso5,"pfnIso5[nPhotons]/F");

  theTree->Branch("pfsumIso1",pfsumIso1,"pfsumIso1[nPhotons]/F");
  theTree->Branch("pfsumIso2",pfsumIso2,"pfsumIso2[nPhotons]/F");
  theTree->Branch("pfsumIso3",pfsumIso3,"pfsumIso3[nPhotons]/F");
  theTree->Branch("pfsumIso4",pfsumIso4,"pfsumIso4[nPhotons]/F");
  theTree->Branch("pfsumIso5",pfsumIso5,"pfsumIso5[nPhotons]/F");

  theTree->Branch("pfcVsIso1",pfcVsIso1,"pfcVsIso1[nPhotons]/F");
  theTree->Branch("pfcVsIso2",pfcVsIso2,"pfcVsIso2[nPhotons]/F");
  theTree->Branch("pfcVsIso3",pfcVsIso3,"pfcVsIso3[nPhotons]/F");
  theTree->Branch("pfcVsIso4",pfcVsIso4,"pfcVsIso4[nPhotons]/F");
  theTree->Branch("pfcVsIso5",pfcVsIso5,"pfcVsIso5[nPhotons]/F");
  theTree->Branch("pfcVsIso1th1",pfcVsIso1th1,"pfcVsIso1th1[nPhotons]/F");
  theTree->Branch("pfcVsIso2th1",pfcVsIso2th1,"pfcVsIso2th1[nPhotons]/F");
  theTree->Branch("pfcVsIso3th1",pfcVsIso3th1,"pfcVsIso3th1[nPhotons]/F");
  theTree->Branch("pfcVsIso4th1",pfcVsIso4th1,"pfcVsIso4th1[nPhotons]/F");
  theTree->Branch("pfcVsIso5th1",pfcVsIso5th1,"pfcVsIso5th1[nPhotons]/F");
  theTree->Branch("pfcVsIso1th2",pfcVsIso1th2,"pfcVsIso1th2[nPhotons]/F");
  theTree->Branch("pfcVsIso2th2",pfcVsIso2th2,"pfcVsIso2th2[nPhotons]/F");
  theTree->Branch("pfcVsIso3th2",pfcVsIso3th2,"pfcVsIso3th2[nPhotons]/F");
  theTree->Branch("pfcVsIso4th2",pfcVsIso4th2,"pfcVsIso4th2[nPhotons]/F");
  theTree->Branch("pfcVsIso5th2",pfcVsIso5th2,"pfcVsIso5th2[nPhotons]/F");

  theTree->Branch("pfnVsIso1",pfnVsIso1,"pfnVsIso1[nPhotons]/F");
  theTree->Branch("pfnVsIso2",pfnVsIso2,"pfnVsIso2[nPhotons]/F");
  theTree->Branch("pfnVsIso3",pfnVsIso3,"pfnVsIso3[nPhotons]/F");
  theTree->Branch("pfnVsIso4",pfnVsIso4,"pfnVsIso4[nPhotons]/F");
  theTree->Branch("pfnVsIso5",pfnVsIso5,"pfnVsIso5[nPhotons]/F");
  theTree->Branch("pfnVsIso1th1",pfnVsIso1th1,"pfnVsIso1th1[nPhotons]/F");
  theTree->Branch("pfnVsIso2th1",pfnVsIso2th1,"pfnVsIso2th1[nPhotons]/F");
  theTree->Branch("pfnVsIso3th1",pfnVsIso3th1,"pfnVsIso3th1[nPhotons]/F");
  theTree->Branch("pfnVsIso4th1",pfnVsIso4th1,"pfnVsIso4th1[nPhotons]/F");
  theTree->Branch("pfnVsIso5th1",pfnVsIso5th1,"pfnVsIso5th1[nPhotons]/F");
  theTree->Branch("pfnVsIso1th2",pfnVsIso1th2,"pfnVsIso1th2[nPhotons]/F");
  theTree->Branch("pfnVsIso2th2",pfnVsIso2th2,"pfnVsIso2th2[nPhotons]/F");
  theTree->Branch("pfnVsIso3th2",pfnVsIso3th2,"pfnVsIso3th2[nPhotons]/F");
  theTree->Branch("pfnVsIso4th2",pfnVsIso4th2,"pfnVsIso4th2[nPhotons]/F");
  theTree->Branch("pfnVsIso5th2",pfnVsIso5th2,"pfnVsIso5th2[nPhotons]/F");

  theTree->Branch("pfpVsIso1",pfpVsIso1,"pfpVsIso1[nPhotons]/F");
  theTree->Branch("pfpVsIso2",pfpVsIso2,"pfpVsIso2[nPhotons]/F");
  theTree->Branch("pfpVsIso3",pfpVsIso3,"pfpVsIso3[nPhotons]/F");
  theTree->Branch("pfpVsIso4",pfpVsIso4,"pfpVsIso4[nPhotons]/F");
  theTree->Branch("pfpVsIso5",pfpVsIso5,"pfpVsIso5[nPhotons]/F");
  theTree->Branch("pfpVsIso1th1",pfpVsIso1th1,"pfpVsIso1th1[nPhotons]/F");
  theTree->Branch("pfpVsIso2th1",pfpVsIso2th1,"pfpVsIso2th1[nPhotons]/F");
  theTree->Branch("pfpVsIso3th1",pfpVsIso3th1,"pfpVsIso3th1[nPhotons]/F");
  theTree->Branch("pfpVsIso4th1",pfpVsIso4th1,"pfpVsIso4th1[nPhotons]/F");
  theTree->Branch("pfpVsIso5th1",pfpVsIso5th1,"pfpVsIso5th1[nPhotons]/F");
  theTree->Branch("pfpVsIso1th2",pfpVsIso1th2,"pfpVsIso1th2[nPhotons]/F");
  theTree->Branch("pfpVsIso2th2",pfpVsIso2th2,"pfpVsIso2th2[nPhotons]/F");
  theTree->Branch("pfpVsIso3th2",pfpVsIso3th2,"pfpVsIso3th2[nPhotons]/F");
  theTree->Branch("pfpVsIso4th2",pfpVsIso4th2,"pfpVsIso4th2[nPhotons]/F");
  theTree->Branch("pfpVsIso5th2",pfpVsIso5th2,"pfpVsIso5th2[nPhotons]/F");

  theTree->Branch("pfsumVsIso1",pfsumVsIso1,"pfsumVsIso1[nPhotons]/F");
  theTree->Branch("pfsumVsIso2",pfsumVsIso2,"pfsumVsIso2[nPhotons]/F");
  theTree->Branch("pfsumVsIso3",pfsumVsIso3,"pfsumVsIso3[nPhotons]/F");
  theTree->Branch("pfsumVsIso4",pfsumVsIso4,"pfsumVsIso4[nPhotons]/F");
  theTree->Branch("pfsumVsIso5",pfsumVsIso5,"pfsumVsIso5[nPhotons]/F");
  theTree->Branch("pfsumVsIso1th1",pfsumVsIso1th1,"pfsumVsIso1th1[nPhotons]/F");
  theTree->Branch("pfsumVsIso2th1",pfsumVsIso2th1,"pfsumVsIso2th1[nPhotons]/F");
  theTree->Branch("pfsumVsIso3th1",pfsumVsIso3th1,"pfsumVsIso3th1[nPhotons]/F");
  theTree->Branch("pfsumVsIso4th1",pfsumVsIso4th1,"pfsumVsIso4th1[nPhotons]/F");
  theTree->Branch("pfsumVsIso5th1",pfsumVsIso5th1,"pfsumVsIso5th1[nPhotons]/F");
  theTree->Branch("pfsumVsIso1th2",pfsumVsIso1th2,"pfsumVsIso1th2[nPhotons]/F");
  theTree->Branch("pfsumVsIso2th2",pfsumVsIso2th2,"pfsumVsIso2th2[nPhotons]/F");
  theTree->Branch("pfsumVsIso3th2",pfsumVsIso3th2,"pfsumVsIso3th2[nPhotons]/F");
  theTree->Branch("pfsumVsIso4th2",pfsumVsIso4th2,"pfsumVsIso4th2[nPhotons]/F");
  theTree->Branch("pfsumVsIso5th2",pfsumVsIso5th2,"pfsumVsIso5th2[nPhotons]/F");


  theTree->Branch("pfVsSubIso1",pfVsSubIso1,"pfVsSubIso1[nPhotons]/F");
  theTree->Branch("pfVsSubIso2",pfVsSubIso2,"pfVsSubIso2[nPhotons]/F");
  theTree->Branch("pfVsSubIso3",pfVsSubIso3,"pfVsSubIso3[nPhotons]/F");
  theTree->Branch("pfVsSubIso4",pfVsSubIso4,"pfVsSubIso4[nPhotons]/F");
  theTree->Branch("pfVsSubIso5",pfVsSubIso5,"pfVsSubIso5[nPhotons]/F");


  theTree->Branch("towerIso1",towerIso1,"towerIso1[nPhotons]/F");
  theTree->Branch("towerIso2",towerIso2,"towerIso2[nPhotons]/F");
  theTree->Branch("towerIso3",towerIso3,"towerIso3[nPhotons]/F");
  theTree->Branch("towerIso4",towerIso4,"towerIso4[nPhotons]/F");
  theTree->Branch("towerIso5",towerIso5,"towerIso5[nPhotons]/F");
  theTree->Branch("towerVsIso1",towerVsIso1,"towerVsIso1[nPhotons]/F");
  theTree->Branch("towerVsIso2",towerVsIso2,"towerVsIso2[nPhotons]/F");
  theTree->Branch("towerVsIso3",towerVsIso3,"towerVsIso3[nPhotons]/F");
  theTree->Branch("towerVsIso4",towerVsIso4,"towerVsIso4[nPhotons]/F");
  theTree->Branch("towerVsIso5",towerVsIso5,"towerVsIso5[nPhotons]/F");
  theTree->Branch("towerVsSubIso1",towerVsSubIso1,"towerVsSubIso1[nPhotons]/F");
  theTree->Branch("towerVsSubIso2",towerVsSubIso2,"towerVsSubIso2[nPhotons]/F");
  theTree->Branch("towerVsSubIso3",towerVsSubIso3,"towerVsSubIso3[nPhotons]/F");
  theTree->Branch("towerVsSubIso4",towerVsSubIso4,"towerVsSubIso4[nPhotons]/F");
  theTree->Branch("towerVsSubIso5",towerVsSubIso5,"towerVsSubIso5[nPhotons]/F");


  theTree->Branch("cr1",cr1,"cr1[nPhotons]/F");
  theTree->Branch("cr2",cr2,"cr2[nPhotons]/F");
  theTree->Branch("cr3",cr3,"cr3[nPhotons]/F");
  theTree->Branch("cr4",cr4,"cr4[nPhotons]/F");
  theTree->Branch("cr5",cr5,"cr5[nPhotons]/F");
  theTree->Branch("cr4j",cr4j,"cr4j[nPhotons]/F");
  theTree->Branch("ct1",ct1,"ct1[nPhotons]/F");
  theTree->Branch("ct2",ct2,"ct2[nPhotons]/F");
  theTree->Branch("ct3",ct3,"ct3[nPhotons]/F");
  theTree->Branch("ct4",ct4,"ct4[nPhotons]/F");
  theTree->Branch("ct5",ct5,"ct5[nPhotons]/F");
  theTree->Branch("ct1PtCut20",ct1PtCut20,"ct1PtCut20[nPhotons]/F");
  theTree->Branch("ct2PtCut20",ct2PtCut20,"ct2PtCut20[nPhotons]/F");
  theTree->Branch("ct3PtCut20",ct3PtCut20,"ct3PtCut20[nPhotons]/F");
  theTree->Branch("ct4PtCut20",ct4PtCut20,"ct4PtCut20[nPhotons]/F");
  theTree->Branch("ct5PtCut20",ct5PtCut20,"ct5PtCut20[nPhotons]/F");
  theTree->Branch("ct4j20",ct4j20,"ct4j20[nPhotons]/F");
  theTree->Branch("ct4j10",ct4j10,"ct4j10[nPhotons]/F");
  theTree->Branch("ct4j15",ct4j15,"ct4j15[nPhotons]/F");
  theTree->Branch("ct4j",ct4j,"ct4j[nPhotons]/F");
  theTree->Branch("dr11",dr11,"dr11[nPhotons]/F");
  theTree->Branch("dr21",dr21,"dr21[nPhotons]/F");
  theTree->Branch("dr31",dr31,"dr31[nPhotons]/F");
  theTree->Branch("dr41",dr41,"dr41[nPhotons]/F");
  theTree->Branch("dr12",dr12,"dr12[nPhotons]/F");
  theTree->Branch("dr22",dr22,"dr22[nPhotons]/F");
  theTree->Branch("dr32",dr32,"dr32[nPhotons]/F");
  theTree->Branch("dr42",dr42,"dr42[nPhotons]/F");
  theTree->Branch("dr13",dr13,"dr13[nPhotons]/F");
  theTree->Branch("dr23",dr23,"dr23[nPhotons]/F");
  theTree->Branch("dr33",dr33,"dr33[nPhotons]/F");
  theTree->Branch("dr43",dr43,"dr43[nPhotons]/F");
  theTree->Branch("dr14",dr14,"dr14[nPhotons]/F");
  theTree->Branch("dr24",dr24,"dr24[nPhotons]/F");
  theTree->Branch("dr34",dr34,"dr34[nPhotons]/F");
  theTree->Branch("dr44",dr44,"dr44[nPhotons]/F");
  theTree->Branch("t11",t11,"t11[nPhotons]/F");
  theTree->Branch("t21",t21,"t21[nPhotons]/F");
  theTree->Branch("t31",t31,"t31[nPhotons]/F");
  theTree->Branch("t41",t41,"t41[nPhotons]/F");
  theTree->Branch("t12",t12,"t12[nPhotons]/F");
  theTree->Branch("t22",t22,"t22[nPhotons]/F");
  theTree->Branch("t32",t32,"t32[nPhotons]/F");
  theTree->Branch("t42",t42,"t42[nPhotons]/F");
  theTree->Branch("t13",t13,"t13[nPhotons]/F");
  theTree->Branch("t23",t23,"t23[nPhotons]/F");
  theTree->Branch("t33",t33,"t33[nPhotons]/F");
  theTree->Branch("t43",t43,"t43[nPhotons]/F");
  theTree->Branch("t14",t14,"t14[nPhotons]/F");
  theTree->Branch("t24",t24,"t24[nPhotons]/F");
  theTree->Branch("t34",t34,"t34[nPhotons]/F");
  theTree->Branch("t44",t44,"t44[nPhotons]/F");


  theTree->Branch("isGenMatched",isGenMatched,"isGenMatched[nPhotons]/I");
  theTree->Branch("genMatchedCollId",genMatchedCollId,"genMatchedCollId[nPhotons]/I");
  theTree->Branch("genMatchedPt",genMatchedPt,"genMatchedPt[nPhotons]/F");
  theTree->Branch("genMatchedEta",genMatchedEta,"genMatchedEta[nPhotons]/F");
  theTree->Branch("genMatchedPhi",genMatchedPhi,"genMatchedPhi[nPhotons]/F");
  theTree->Branch("genMomId",genMomId,"genMomId[nPhotons]/I");
  theTree->Branch("genGrandMomId",genGrandMomId,"genGrandMomId[nPhotons]/I");
  theTree->Branch("genNSiblings",genNSiblings,"genNSiblings[nPhotons]/I");
  theTree->Branch("genCalIsoDR03",genCalIsoDR03,"genCalIsoDR03[nPhotons]/F");
  theTree->Branch("genCalIsoDR04",genCalIsoDR04,"genCalIsoDR04[nPhotons]/F");
  theTree->Branch("genTrkIsoDR03",genTrkIsoDR03,"genTrkIsoDR03[nPhotons]/F");
  theTree->Branch("genTrkIsoDR04",genTrkIsoDR04,"genTrkIsoDR04[nPhotons]/F");


  theTree->Branch("nGp",&nGp,"nGp/I");
  theTree->Branch("simVtxX",&simVtxX,"simVtxX/F");
  theTree->Branch("simVtxY",&simVtxY,"simVtxY/F");
  theTree->Branch("simVtxZ",&simVtxZ,"simVtxZ/F");
  theTree->Branch("ptHat",&ptHat,"ptHat/F");
  theTree->Branch("gpEt",gpEt,"gpEt[nGp]/F");
  theTree->Branch("gpEta",gpEta,"gpEta[nGp]/F");
  theTree->Branch("gpCalIsoDR04",gpCalIsoDR04,"gpCalIsoDR04[nGp]/F");
  theTree->Branch("gpCalIsoDR03",gpCalIsoDR03,"gpCalIsoDR03[nGp]/F");
  theTree->Branch("gpTrkIsoDR03",gpTrkIsoDR03,"gpTrkIsoDR03[nGp]/F");
  theTree->Branch("gpTrkIsoDR04",gpTrkIsoDR04,"gpTrkIsoDR04[nGp]/F");
  theTree->Branch("gpStatus",gpStatus,"gpStatus[nGp]/I");
  theTree->Branch("gpCollId",gpCollId,"gpCollId[nGp]/I");
  theTree->Branch("gpId",gpId,"gpId[nGp]/I");
  theTree->Branch("gpMomId",gpMomId,"gpMomId[nGp]/I");



}

void SinglePhotonAnalyzerTree::endJob() {
  //  tplmgr->Store();
  // tplmgr->SetDir("Info");
  //  TObjString codeVersion = "$Name: HiForest_V02_01 $";
  //  codeVersion.Write("CodeVersion");
  //  delete tplmgr;

}



bool SinglePhotonAnalyzerTree::analyzeMC(const edm::Event& e, const edm::EventSetup& iSetup){

  /////////////////////////////////////////////////////////
  // Generator Section: Analyzing Monte Carlo Truth Info //
  /////////////////////////////////////////////////////////

  simVtxX=-100000;
  simVtxY=-100000;
  simVtxZ=-100000;
  ptHat=-100000;
  nGp=0;

  Handle<HepMCProduct> evtMC;
  e.getByLabel(hepMCProducer_,evtMC);
  if (evtMC.isValid())  isMC_=kTRUE;
  edm::Handle<reco::GenParticleCollection> genParticles;

  if (isMC_) {
    // get simulated vertex and store in ntuple
    Float_t simVertexX(0), simVertexY(0), simVertexZ(0);
    if(evtMC->GetEvent()->signal_process_vertex() != NULL) {
      simVertexX = evtMC->GetEvent()->signal_process_vertex()->position().x();
      simVertexY = evtMC->GetEvent()->signal_process_vertex()->position().y();
      simVertexZ = evtMC->GetEvent()->signal_process_vertex()->position().z();
    }


    // get pthat value and store in ntuple
    edm::Handle<GenEventInfoProduct>    genEventScale;
    e.getByLabel(genEventScale_, genEventScale);   // hi style


    simVtxX = simVertexX;
    simVtxY = simVertexY;
    simVtxZ = simVertexZ;
    ptHat = genEventScale->qScale();

    //  get generated particles and store generator ntuple
    try { e.getByLabel( genParticleProducer_,      genParticles );} catch (...) {;}
    const int nMaxGenPar = 90;

    nGp=0;
    for (reco::GenParticleCollection::const_iterator it_gen =
	   genParticles->begin(); it_gen!= genParticles->end(); it_gen++){
      const reco::GenParticle &p = (*it_gen);
      if ( p.pt() < mcPtMin_ ||  fabs(p.p4().eta()) > mcEtaMax_ ) continue;

      gpEt    [nGp] = p.et();
      gpEta   [nGp] = p.eta();
      gpPhi   [nGp] = p.phi();
      gpCalIsoDR04 [nGp] =  getGenCalIso(genParticles, it_gen, 0.4);
      gpCalIsoDR03 [nGp] =  getGenCalIso(genParticles, it_gen, 0.3);

      gpTrkIsoDR03[nGp] = getGenTrkIso(genParticles, it_gen, 0.3);
      gpTrkIsoDR04[nGp] = getGenTrkIso(genParticles, it_gen, 0.4);

      gpStatus [nGp] = p.status();
      gpCollId [nGp] = p.collisionId();
      gpId     [nGp] = p.pdgId();
      gpMomId  [nGp] = 0;
      if( p.numberOfMothers() > 0 )
	gpMomId[nGp] = p.mother()->pdgId();

      nGp++;
      if (nGp> nMaxGenPar-1) break;
    }
  }
  return (isMC_);
}





Int_t SinglePhotonAnalyzerTree::getNumOfPreshClusters(Photon *photon, const edm::Event& e) {

  // ES clusters in X plane
  edm::Handle<reco::PreshowerClusterCollection> esClustersX;
  e.getByLabel(InputTag("multi5x5SuperClustersWithPreshower:preshowerXClusters"), esClustersX);
  const reco::PreshowerClusterCollection *ESclustersX = esClustersX.product();

  // ES clusters in Y plane
  edm::Handle<reco::PreshowerClusterCollection> esClustersY;
  e.getByLabel(InputTag("multi5x5SuperClustersWithPreshower:preshowerYClusters"),esClustersY);
  const reco::PreshowerClusterCollection *ESclustersY = esClustersY.product();


  Int_t numOfPreshClusters(-1);

  // Is the photon in region of Preshower?
  if (fabs(photon->eta())>1.62) {
    numOfPreshClusters=0;

    // Loop over all ECAL Basic clusters in the supercluster
    for (reco::CaloCluster_iterator ecalBasicCluster = photon->superCluster()->clustersBegin();
	 ecalBasicCluster!=photon->superCluster()->clustersEnd(); ecalBasicCluster++) {
      const reco::CaloClusterPtr ecalBasicClusterPtr = *(ecalBasicCluster);

      for (reco::PreshowerClusterCollection::const_iterator iESClus = ESclustersX->begin(); iESClus != ESclustersX->end(); ++iESClus) {
	const reco::CaloClusterPtr preshBasicCluster = iESClus->basicCluster();
	//const reco::PreshowerCluster *esCluster = &*iESClus;
	if (preshBasicCluster == ecalBasicClusterPtr) {
	  numOfPreshClusters++;
	  //  cout << esCluster->energy() <<"\t" << esCluster->x() << "\t" << esCluster->y() << endl;
	}
      }

      for (reco::PreshowerClusterCollection::const_iterator iESClus = ESclustersY->begin(); iESClus != ESclustersY->end(); ++iESClus) {
	const reco::CaloClusterPtr preshBasicCluster = iESClus->basicCluster();
	//const reco::PreshowerCluster *esCluster = &*iESClus;
	if (preshBasicCluster == ecalBasicClusterPtr) {
	  numOfPreshClusters++;
	  //  cout << esCluster->energy() <<"\t" << esCluster->x() << "\t" << esCluster->y() << endl;
	}
      }
    }
  }

  return numOfPreshClusters;

}

Float_t SinglePhotonAnalyzerTree::getESRatio(Photon *photon, const edm::Event& e, const edm::EventSetup& iSetup){

  //get Geometry
  ESHandle<CaloGeometry> caloGeometry;
  iSetup.get<CaloGeometryRecord>().get(caloGeometry);
  const CaloSubdetectorGeometry *geometry = caloGeometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;

  // Get ES rechits
  edm::Handle<EcalRecHitCollection> PreshowerRecHits;
  e.getByLabel(InputTag("ecalPreshowerRecHit","EcalRecHitsES"), PreshowerRecHits);
  if( PreshowerRecHits.isValid() ) EcalRecHitCollection preshowerHits(*PreshowerRecHits);

  Float_t esratio=-1.;

  if (fabs(photon->eta())>1.62) {

    const reco::CaloClusterPtr seed = (*photon).superCluster()->seed();
    reco::CaloCluster cluster = (*seed);
    const GlobalPoint phopoint(cluster.x(), cluster.y(), cluster.z());

    DetId photmp1 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(phopoint, 1);
    DetId photmp2 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(phopoint, 2);
    ESDetId esfid = (photmp1 == DetId(0)) ? ESDetId(0) : ESDetId(photmp1);
    ESDetId esrid = (photmp2 == DetId(0)) ? ESDetId(0) : ESDetId(photmp2);

    int gs_esfid = -99;
    int gs_esrid = -99;
    gs_esfid = esfid.six()*32+esfid.strip();
    gs_esrid = esrid.siy()*32+esrid.strip();

    float esfe3 = 0.;
    float esfe21 = 0.;
    float esre3 = 0.;
    float esre21 = 0.;

    const ESRecHitCollection *ESRH = PreshowerRecHits.product();
    EcalRecHitCollection::const_iterator esrh_it;
    for ( esrh_it = ESRH->begin(); esrh_it != ESRH->end(); esrh_it++) {
      ESDetId esdetid = ESDetId(esrh_it->id());
      if ( esdetid.plane()==1 ) {
	if ( esdetid.zside() == esfid.zside() &&
	     esdetid.siy() == esfid.siy() ) {
	  int gs_esid = esdetid.six()*32+esdetid.strip();
	  int ss = gs_esid-gs_esfid;
	  if ( TMath::Abs(ss)<=10) {
	    esfe21 += esrh_it->energy();
	  }
	  if ( TMath::Abs(ss)<=1) {
	    esfe3 += esrh_it->energy();
	  }
	}
      }
      if (esdetid.plane()==2 ){
	if ( esdetid.zside() == esrid.zside() &&
	     esdetid.six() == esrid.six() ) {
	  int gs_esid = esdetid.siy()*32+esdetid.strip();
	  int ss = gs_esid-gs_esrid;
	  if ( TMath::Abs(ss)<=10) {
	    esre21 += esrh_it->energy();
	  }
	  if ( TMath::Abs(ss)<=1) {
	    esre3 += esrh_it->energy();
	  }
	}
      }
    }

    if( (esfe21+esre21) == 0.) {
      esratio = 1.;
    }else{
      esratio = (esfe3+esre3) / (esfe21+esre21);
    }


    if ( esratio>1.) {
      cout << "es numbers " << esfe3 << " " << esfe21 << " " << esre3 << " " << esre21 << endl;
    }

  }

  return esratio;

}


// get amount of generator isolation
// default cut value of etMin is 0.0
// return number of particles and sumEt surrounding candidate

Float_t SinglePhotonAnalyzerTree::getGenCalIso(edm::Handle<reco::GenParticleCollection> handle,
					       reco::GenParticleCollection::const_iterator thisPho,                                            const Float_t dRMax)
{
  const Float_t etMin = 0.0;
  Float_t genCalIsoSum = 0.0;
  if(!isMC_)return genCalIsoSum;
  if(!handle.isValid())return genCalIsoSum;

  for (reco::GenParticleCollection::const_iterator it_gen =
	 handle->begin(); it_gen!=handle->end(); it_gen++){

    if(it_gen == thisPho)continue;      // can't be the original photon
    if(it_gen->status()!=1)continue;    // need to be a stable particle
    if (thisPho->collisionId() != it_gen->collisionId())  // has to come from the same collision
      continue;

    Int_t pdgCode = abs(it_gen->pdgId());
    if(pdgCode>11 && pdgCode < 20)continue;     // we should not count neutrinos, muons

    Float_t et = it_gen->et();
    if(et < etMin) continue; // pass a minimum et threshold, default 0

    Float_t dR = reco::deltaR(thisPho->momentum(),
			      it_gen->momentum());
    if(dR > dRMax) continue; // within deltaR cone
    genCalIsoSum += et;

  }// end of loop over gen particles

  return genCalIsoSum;
}


//=============================================================================
// default cut value of ptMin is 0.0

Float_t SinglePhotonAnalyzerTree::getGenTrkIso(edm::Handle<reco::GenParticleCollection> handle,
					       reco::GenParticleCollection::const_iterator thisPho,                                            const Float_t dRMax)
{
  const Float_t ptMin = 0.0;
  Float_t genTrkIsoSum = 0.0;
  if(!isMC_)return genTrkIsoSum;
  if(!handle.isValid())return genTrkIsoSum;

  for (reco::GenParticleCollection::const_iterator it_gen =
	 handle->begin(); it_gen!=handle->end(); it_gen++){

    if(it_gen == thisPho)continue;      // can't be the original photon
    if(it_gen->status()!=1)continue;    // need to be a stable particle
    if (thisPho->collisionId() != it_gen->collisionId())  // has to come from the same collision
      continue;

    if(it_gen->charge()==0)continue;    // we should not count neutral particles

    Float_t pt = it_gen->pt();
    if(pt < ptMin) continue; // pass a minimum pt threshold, default 0

    Float_t dR = reco::deltaR(thisPho->momentum(),
			      it_gen->momentum());
    if(dR > dRMax) continue; // within deltaR cone
    genTrkIsoSum += pt;

  }// end of loop over gen particles

  return genTrkIsoSum;
}






DEFINE_FWK_MODULE(SinglePhotonAnalyzerTree);
