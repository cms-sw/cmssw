/** \class MultiPhotonAnalyzerTree MultiPhotonAnalyzerTree.cc CmsHi/PhotonAnalysis/plugin/MultiPhotonAnalyzerTree.cc
 *
 * Description:
 * Analysis code of the QCD Photons group;
 * Data Analyzer for the single photon (inclusive isolated photons and photon plus jets) cross section measurement;
 * Store in ntuple run conditions and missing ET information for each event;
 * Makes ntuple of the photon and jet arrays in each event;
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
 * \author Yongsun Kim,         MIT, US
 * \author Pasquale Musella,    LIP, PT
 * \author Shin-Shan Eiko Yu,   National Central University, TW
 * \author Abe DeBenedetti,     University of Minnesota, US
 * \author Rong-Shyang Lu,      National Taiwan University, TW
 * \version $Id: MultiPhotonAnalyzerTree.cc,v 1.14 2011/10/24 22:08:47 yjlee Exp $
 * j
 */


// This MultiphotonAnalyzer was modified to fit with Heavy Ion collsion by Yongsun Kim ( MIT)


#include <memory>
#include <iostream>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Common/interface/TriggerNames.h"

//Trigger DataFormats
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTrigger/plugins/L1GlobalTrigger.h"

//#include "DataFormats/Common/plugins/TriggerResults.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/PatCandidates/interface/Photon.h"
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

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//ROOT includes
#include <Math/VectorUtil.h>
#include <TLorentzVector.h>

//Include the Single Photon Analyzer
#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/MultiPhotonAnalyzerTree.h"

//Include Heavy Ion isolation variable calculator
#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/CxCalculator.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/RxCalculator.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/TxCalculator.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/TxyCalculator.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/dRxyCalculator.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/dRxyCalculator.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/src/pfIsoCalculator.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/src/towerIsoCalculator.h"

// Electron
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

//particeFlow Canddiate
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"


// Voronoi algorithm
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"

using namespace pat;
using namespace edm;
using namespace std;
using namespace ROOT::Math::VectorUtil;


MultiPhotonAnalyzerTree::MultiPhotonAnalyzerTree(const edm::ParameterSet& ps):
  SinglePhotonAnalyzerTree(ps)
  //  kMaxPhotons(ps.getUntrackedParameter<int>("MaxPhotons", 50))
  // already defined as 50
{
}

MultiPhotonAnalyzerTree::~MultiPhotonAnalyzerTree(){
}



void MultiPhotonAnalyzerTree::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {

  //  if (doStoreGeneral_) 	storeGeneral(e, iSetup);
  //  if (doStoreL1Trigger_) 	storeL1Trigger(e);
  //   if (doStoreHLT_) 	storeHLT(e);
  //  if (doStoreHF_)		storeHF(e);
  //  if (doStoreVertex_)	storeVertex(e);
  //  if (doStoreMET_)	storeMET(e);
  //  if (doStoreJets_)	storeJets(e);
  //  if (doStoreTracks_)     storeTracks(e);

  //   storeEvtPlane(e);

  analyzeMC(e,iSetup);
  //int foundPhotons = selectStorePhotons(e,iSetup,"");
  selectStorePhotons(e,iSetup,"");
  //cout <<"Found photons? "<<foundPhotons<<endl;
  theTree->Fill();
}



int MultiPhotonAnalyzerTree::selectStorePhotons(const edm::Event& e,const edm::EventSetup& iSetup, const char* prefx){

  Handle<HepMCProduct> evtMC;
  e.getByLabel(hepMCProducer_,evtMC);
  isMC_ = false;
  if (evtMC.isValid())  isMC_=kTRUE;



  /////////////////////////////////////////////////////////////////////////////
  // Photon Section: store kMaxPhotons in the events as an array in the tree //
  /////////////////////////////////////////////////////////////////////////////
  // Get photon details
  Handle<pat::PhotonCollection> photons;
  e.getByLabel(photonProducer_, photons);

  pat::PhotonCollection myphotons;
  for (PhotonCollection::const_iterator phoItr = photons->begin(); phoItr != photons->end(); ++phoItr) {
    myphotons.push_back(*phoItr);
  }

  reco::PhotonCollection myCompPhotons;
  if (doStoreCompCone_) {
    Handle<reco::PhotonCollection> compPhotons;
    e.getByLabel(compPhotonProducer_, compPhotons);
    for (reco::PhotonCollection::const_iterator phoItr = compPhotons->begin(); phoItr != compPhotons->end(); ++phoItr){
      myCompPhotons.push_back(*phoItr);
    }
  }

  GreaterByPt<Photon> pTComparator_;

  // Sort photons according to pt
  std::sort(myphotons.begin(), myphotons.end(), pTComparator_);
  std::sort(myCompPhotons.begin(), myCompPhotons.end(), pTComparator_);

  TString pfx(prefx);

  // Tools to get cluster shapes

  edm::Handle<EcalRecHitCollection> EBReducedRecHits;
  e.getByToken(ebReducedRecHitCollection_, EBReducedRecHits);
  edm::Handle<EcalRecHitCollection> EEReducedRecHits;
  e.getByToken(eeReducedRecHitCollection_, EEReducedRecHits);
  // get the channel status from the DB
  edm::ESHandle<EcalChannelStatus> chStatus;
  iSetup.get<EcalChannelStatusRcd>().get(chStatus);

  EcalClusterLazyTools lazyTool(e, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_ );

  // Tools to get electron informations.
  edm::Handle<reco::GsfElectronCollection> EleHandle ;
  e.getByLabel (EleTag_.label(),EleHandle) ;
  reco::GsfElectronCollection myEle;

  bool isEleRecoed = false;
  if (EleHandle.isValid()) {
    //    cout << " electron was reconstructed! " << endl;
    isEleRecoed = true;
  }

  if ( isEleRecoed) {
    for (reco::GsfElectronCollection::const_iterator eleItr = EleHandle->begin(); eleItr != EleHandle->end(); ++eleItr) {
      myEle.push_back(*eleItr);
    }
  }


  // Heavy Ion variable calculator
  CxCalculator CxC(e,iSetup, basicClusterBarrel_, basicClusterEndcap_);
  RxCalculator RxC(e,iSetup, hbhe_, hf_, ho_);
  TxCalculator TxC(e,iSetup, trackProducer_,trackQuality_);
  TxyCalculator Txy(e,iSetup,trackProducer_,trackQuality_);
  dRxyCalculator dRxy(e,iSetup,trackProducer_,trackQuality_);
  pfIsoCalculator pfIso(e,iSetup,pfCandidateLabel_, srcPfVor_, vertexProducer_);
  towerIsoCalculator towerIso(e,iSetup,towerCandidateLabel_, srcTowerVor_, vertexProducer_);

  // store general
  run = (Int_t)e.id().run();
  event = (Int_t)e.id().event();
  bunchCrossing = (Int_t)e.bunchCrossing();
  luminosityBlock = (Int_t)e.luminosityBlock();

  int nphotonscounter(0);
  for (PhotonCollection::const_iterator phoItr = myphotons.begin(); phoItr != myphotons.end(); ++phoItr) {
    if(phoItr->pt() < ptMin_ || fabs(phoItr->p4().eta()) > etaMax_) continue;
    // Dump photon kinematics and AOD
    Photon photon = Photon(*phoItr);

    pt[nphotonscounter] = photon.p4().pt();
    px[nphotonscounter] = photon.px();
    py[nphotonscounter] = photon.py();
    pz[nphotonscounter] = photon.pz();
    energy[nphotonscounter] = photon.energy();
    rawEnergy[nphotonscounter] =  photon.superCluster()->rawEnergy();

    eta[nphotonscounter] =  photon.p4().eta();
    phi[nphotonscounter] =  photon.p4().phi();
    r9[nphotonscounter]    =  photon.r9();

    isEB[nphotonscounter]    =  photon.isEB()? 1:0;
    isEBGap[nphotonscounter]    =  photon.isEBGap()? 1:0;
    isEEGap[nphotonscounter]    =  photon.isEEGap()? 1:0;
    isEBEEGap[nphotonscounter]  =  photon.isEBEEGap()? 1:0;
    isTransGap[nphotonscounter] =  (fabs(photon.eta()) > ecalBarrelMaxEta_ && fabs(photon.eta()) < ecalEndcapMinEta_) ? 1:0;

    preshowerEnergy[nphotonscounter]    =  photon.superCluster()->preshowerEnergy();
    numOfPreshClusters[nphotonscounter] =  getNumOfPreshClusters(&photon, e);
    clustersSize[nphotonscounter]   =  photon.superCluster()->clustersSize();
    phiWidth  [nphotonscounter]  =  photon.superCluster()->phiWidth();
    etaWidth  [nphotonscounter]   =  photon.superCluster()->etaWidth();
    scEta     [nphotonscounter]   =  photon.superCluster()->eta();
    scPhi     [nphotonscounter]  =  photon.superCluster()->phi();
    scSize    [nphotonscounter]   =  photon.superCluster()->size();

    //ES Ratio
    ESRatio   [nphotonscounter]   =  getESRatio(&photon, e, iSetup);

    // Cluster shape variables

    const reco::CaloClusterPtr  seed = photon.superCluster()->seed();

    const DetId &id = lazyTool.getMaximum(*seed).first;
    float time  = -999., /*outOfTimeChi2 = -999.,*/ chi2 = -999.;
    int   flags = -1, severity = -1;
    //EcalSeverityLevel::SeverityLevel severityFlag;
    const EcalRecHitCollection & rechits = ( photon.isEB() ? *EBReducedRecHits : *EEReducedRecHits);
    EcalRecHitCollection::const_iterator it = rechits.find( id );
    if( it != rechits.end() ) {
      time = it->time();
      //outOfTimeChi2 = it->outOfTimeChi2();
      chi2 = it->chi2();
      flags = it->recoFlag();
      //severityFlag = EcalSeverityLevelAlgo::severityLevel(id, rechits);
    }
    // Yen-Jie: Not used, need to be fixed
    //severity = -1;
    //if (severityFlag = EcalSeverityLevelAlgo::SeverityLevel::kGood) severity = 0;
/*

  float tleft = -999., tright=-999., ttop=-999., tbottom=-999.;
  std::vector<DetId> left   = lazyTool.matrixDetId(id,-1,-1, 0, 0);
  std::vector<DetId> right  = lazyTool.matrixDetId(id, 1, 1, 0, 0);
  std::vector<DetId> top    = lazyTool.matrixDetId(id, 0, 0, 1, 1);
  std::vector<DetId> bottom = lazyTool.matrixDetId(id, 0, 0,-1,-1);

  float *times[4] = {&tleft,&tright,&ttop,&tbottom};
  std::vector<DetId> ids[4]  = {left,right,top,bottom};
  int nt = sizeof(times)/sizeof(float);
  for(int ii=0; ii<nt; ++ii) {
  if( ids[ii].empty() ) { continue; }
  it = rechits.find( ids[ii][0] );
  if( it != rechits.end() ) { *(times[ii]) = it->time(); }
  }*/

    seedTime              [nphotonscounter]  = time;
    //seedOutOfTimeChi2     [nphotonscounter]  = outOfTimeChi2;
    seedChi2              [nphotonscounter]  = chi2;
    seedRecoFlag          [nphotonscounter] = flags;
    seedSeverity          [nphotonscounter]  = severity;

/*
  tLeft         [nphotonscounter] = tleft;
  tRight        [nphotonscounter] = tright;
  tTop        [nphotonscounter] = ttop;
  tBottom        [nphotonscounter] = tbottom;
*/

    eMax         [nphotonscounter] =  lazyTool.eMax(*seed);
    e2nd         [nphotonscounter] =  lazyTool.e2nd(*seed);
    e2x2         [nphotonscounter] =  lazyTool.e2x2(*seed);
    e3x2         [nphotonscounter] =  lazyTool.e3x2(*seed);
    e3x3         [nphotonscounter] =  lazyTool.e3x3(*seed);
    e4x4         [nphotonscounter] =  lazyTool.e4x4(*seed);
    e5x5         [nphotonscounter] =  lazyTool.e5x5(*seed);
    e2overe8     [nphotonscounter] =  ( lazyTool.e3x3(*seed)-lazyTool.eMax(*seed) ==0 )? 0: lazyTool.e2nd(*seed)/( lazyTool.e3x3(*seed)-lazyTool.eMax(*seed) );

    e2x5Right    [nphotonscounter] =  lazyTool.e2x5Right(*seed);
    e2x5Left     [nphotonscounter] =  lazyTool.e2x5Left(*seed);
    e2x5Top      [nphotonscounter] =  lazyTool.e2x5Top(*seed);
    e2x5Bottom   [nphotonscounter] =  lazyTool.e2x5Bottom(*seed);
    eRight       [nphotonscounter] =  lazyTool.eRight(*seed);
    eLeft        [nphotonscounter] =  lazyTool.eLeft(*seed);
    eTop         [nphotonscounter] =  lazyTool.eTop(*seed);
    eBottom      [nphotonscounter] =  lazyTool.eBottom(*seed);
    swissCrx     [nphotonscounter] =  1 - ( eRight[nphotonscounter] + eLeft[nphotonscounter] + eTop[nphotonscounter] + eBottom[nphotonscounter] )/eMax[nphotonscounter]  ;


    hadronicOverEm      [nphotonscounter]   =  photon.hadronicOverEm();
    hadronicDepth1OverEm[nphotonscounter]   =  photon.hadronicDepth1OverEm();
    hadronicDepth2OverEm[nphotonscounter]   =  photon.hadronicDepth2OverEm();
    //  trackIso            (nphotonscounter)   =  photon.trackIso();
    //  caloIso             (nphotonscounter)   =  photon.caloIso();
    //   ecalIso             (nphotonscounter)   =  photon.ecalIso();
    //  hcalIso             (nphotonscounter)   =  photon.hcalIso();

    vector<float> vCov;
    vCov = lazyTool.covariances(*seed);
    covPhiPhi    [nphotonscounter] = vCov[0];
    covEtaPhi    [nphotonscounter] = vCov[1];
    covEtaEta    [nphotonscounter] = vCov[2];

    // Photon shower shape parameters

    //  maxEnergyXtal(nphotonscounter) =  photon.maxEnergyXtal();
    sigmaEtaEta  [nphotonscounter] =  photon.sigmaEtaEta();
    sigmaIetaIeta[nphotonscounter] =  photon.sigmaIetaIeta();


    // see http://cmslxr.fnal.gov/lxr/source/RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h#076
    // other sieie values;
    vector<float> lCov39 = lazyTool.localCovariances(*seed, 3.9 );
    sieie39      [nphotonscounter] = sqrt(lCov39[0]);

    vector<float> lCov42 = lazyTool.localCovariances(*seed, 4.2 );
    sieie42      [nphotonscounter] = sqrt(lCov42[0]);

    vector<float> lCov45 = lazyTool.localCovariances(*seed, 4.5 );
    sieie45      [nphotonscounter] = sqrt(lCov45[0]);

    vector<float> lCov47 = lazyTool.localCovariances(*seed, 4.7 );
    sieie47      [nphotonscounter] = sqrt(lCov47[0]);
    sigmaIphiIphi[nphotonscounter] =  sqrt(lCov47[2]);

    vector<float> lCov50 = lazyTool.localCovariances(*seed, 5.0 );
    sieie50      [nphotonscounter] = sqrt(lCov50[0]);


    // absolute cuts.   cut = eMax * exp(-v)   so, v = -log(cut/eMax)
    //  float  theSeedE = eMax[nphotonscounter];
    //  vector<float> lCov02a = lazyTool.localCovariances(*seed, -log(0.2/theSeedE) );
    // sieie02a(nphotonscounter) = sqrt(lCov02a[0]);
    //    cout << "theSeedE = " << theSeedE << "    and cut = 0.2 " << "log (cut/theSeedE) = " << -log( 0.2 / theSeedE )  << endl;
    // vector<float> lCov03a = lazyTool.localCovariances(*seed, -log( 0.3 / theSeedE ) );
    // sieie03a(nphotonscounter) = sqrt(lCov03a[0]);

    r1x5         [nphotonscounter] =  photon.r1x5();
    r2x5         [nphotonscounter] =  photon.r2x5();
    e1x5         [nphotonscounter] =  photon.e1x5();
    e2x5         [nphotonscounter] =  photon.e2x5();




    // electron id
    bool isEleTemp = false;
    float dphiTemp(100), detaTemp(100);
    //float deltaPhiEleCTTemp(100), deltaEtaEleCTTemp(100);
    int eleChargeTemp(100);
    float eleEpTemp(100);

    //    cout << "photon et = " << photon.et() << "  eta =" << photon.eta() << endl;

    if ( isEleRecoed ) {
      // We will find the smallest e/p electron canddiates.
      for ( reco::GsfElectronCollection::const_iterator eleItr = myEle.begin(); eleItr != myEle.end(); ++eleItr) {
	//	if ( eleItr->superCluster()->energy() < 10 ) continue;
	if ( eleItr->energy() < 10 ) continue;
	if ( abs( eleItr->eta() - photon.eta() ) > 0.03 ) continue;

	//	float dphi = eleItr->superCluster()->phi() - photon.superCluster()->phi();
	float dphi = eleItr->phi() - photon.phi();
	if ( dphi >  3.141592 ) dphi = dphi - 2* 3.141592;
	if ( dphi < -3.141592 ) dphi = dphi + 2* 3.141592;
	if ( abs(dphi) > 0.03 )  continue;

	float iEp = eleItr->eSuperClusterOverP() ;

	if ( eleEpTemp < iEp )  continue;
	eleEpTemp = iEp;
	eleChargeTemp =  eleItr->charge();
	//deltaPhiEleCTTemp =  eleItr->deltaPhiEleClusterTrackAtCalo() ;
	//deltaEtaEleCTTemp =  eleItr->deltaEtaEleClusterTrackAtCalo() ;
	dphiTemp = dphi;
	detaTemp = eleItr->eta() - photon.eta() ;
	//	detaTemp = eleItr->superCluster()->eta() - photon.superCluster()->eta() ;

	isEleTemp = true;
      }

      //   if ( isEleTemp == false)
    }

    isEle         [nphotonscounter]    =  isEleTemp;
    detaEle            [nphotonscounter]    =  detaTemp;
    dphiEle            [nphotonscounter]    =  dphiTemp;
    // deltaEtaEleCT      [nphotonscounter]    =  deltaEtaEleCTTemp;
    // deltaPhiEleCT      [nphotonscounter]    =  deltaPhiEleCTTemp;
    eleCharge          [nphotonscounter]    =  eleChargeTemp;
    eleEoverP          [nphotonscounter]    =  eleEpTemp;




    // comp cones;
    //  int nComp = 0;
    //  int allcomps = 0;
    //  float sumCompEIso=0;
    //  float sumCompHIso=0;
    //  float sumCompTIso=0;

    /* ///////////////// comp photon
       for (reco::PhotonCollection::const_iterator compItr = myCompPhotons.begin(); compItr != myCompPhotons.end(); ++compItr) {
       allcomps++;
       if(compItr->pt() < ptMin_ || fabs(compItr->p4().eta()) > etaMax_) continue;

       if(compItr->superCluster()->energy() != photon.superCluster()->energy() ) continue;


       nComp++;
       Photon compPhoton = Photon(*compItr);
       compPhoton.setVertex(vtx_);
       sumCompEIso =  sumCompEIso + compItr->ecalRecHitSumEtConeDR04();
       sumCompHIso =  sumCompHIso + compItr->hcalTowerSumEtConeDR04();
       sumCompTIso =  sumCompTIso + compItr->trkSumPtHollowConeDR04();

       }

       //    cout << " Number of matched compl cones = " << nComp << endl;
       if ( nComp > 0 ) {
       compTrackIso[nphotonscounter) = sumCompTIso/(double)nComp;
       compEcalIso [nphotonscounter) = sumCompEIso/(double)nComp;
       compHcalIso [nphotonscounter) = sumCompHIso/(double)nComp;
       }
       else {
       compTrackIso(nphotonscounter) = -100;
       compEcalIso (nphotonscounter) = -100;
       compHcalIso (nphotonscounter) = -100;
       }
    */   //////////////// comp photon


    // Delta R= 0.4

    ecalRecHitSumEtConeDR04     [nphotonscounter]   =  photon.ecalRecHitSumEtConeDR04();
    hcalTowerSumEtConeDR04      [nphotonscounter]   =  photon.hcalTowerSumEtConeDR04();
    hcalDepth1TowerSumEtConeDR04[nphotonscounter]   =  photon.hcalDepth1TowerSumEtConeDR04();
    hcalDepth2TowerSumEtConeDR04[nphotonscounter]   =  photon.hcalDepth2TowerSumEtConeDR04();
    trkSumPtSolidConeDR04       [nphotonscounter]   =  photon.trkSumPtSolidConeDR04();
    trkSumPtHollowConeDR04      [nphotonscounter]   =  photon.trkSumPtHollowConeDR04();

    //    nTrkSolidConeDR04           (nphotonscounter)   =  photon.nTrkSolidConeDR04();
    //  nTrkHollowConeDR04          (nphotonscounter)   =  photon.nTrkHollowConeDR04();

    // Delta R= 0.3

    ecalRecHitSumEtConeDR03     [nphotonscounter]   =  photon.ecalRecHitSumEtConeDR03();
    hcalTowerSumEtConeDR03      [nphotonscounter]   =  photon.hcalTowerSumEtConeDR03();
    hcalDepth1TowerSumEtConeDR03[nphotonscounter]   =  photon.hcalDepth1TowerSumEtConeDR03();
    hcalDepth2TowerSumEtConeDR03[nphotonscounter]   =  photon.hcalDepth2TowerSumEtConeDR03();
    trkSumPtSolidConeDR03       [nphotonscounter]   =  photon.trkSumPtSolidConeDR03();
    trkSumPtHollowConeDR03      [nphotonscounter]   =  photon.trkSumPtHollowConeDR03();
    //  nTrkSolidConeDR03           (nphotonscounter)   =  photon.nTrkSolidConeDR03();
    //   nTrkHollowConeDR03          (nphotonscounter)   =  photon.nTrkHollowConeDR03();


    c1                          [nphotonscounter]   =  CxC.getCx(photon.superCluster(),1,0);
    c2                          [nphotonscounter]   =  CxC.getCx(photon.superCluster(),2,0);
    c3                          [nphotonscounter]   =  CxC.getCx(photon.superCluster(),3,0);
    c4                          [nphotonscounter]   =  CxC.getCx(photon.superCluster(),4,0);
    c5                          [nphotonscounter]   =  CxC.getCx(photon.superCluster(),5,0);

    t1                          [nphotonscounter]   =  TxC.getTx(photon,1,0);
    t2                          [nphotonscounter]   =  TxC.getTx(photon,2,0);
    t3                          [nphotonscounter]   =  TxC.getTx(photon,3,0);
    t4                          [nphotonscounter]   =  TxC.getTx(photon,4,0);
    t5                          [nphotonscounter]   =  TxC.getTx(photon,5,0);

    r1                          [nphotonscounter]   =  RxC.getRx(photon.superCluster(),1,0);
    r2                          [nphotonscounter]   =  RxC.getRx(photon.superCluster(),2,0);
    r3                          [nphotonscounter]   =  RxC.getRx(photon.superCluster(),3,0);
    r4                          [nphotonscounter]   =  RxC.getRx(photon.superCluster(),4,0);
    r5                          [nphotonscounter]   =  RxC.getRx(photon.superCluster(),5,0);

    t1PtCut                     [nphotonscounter]   =  TxC.getTx(photon,1,2); // 2 GeV cut
    t2PtCut                     [nphotonscounter]   =  TxC.getTx(photon,2,2);
    t3PtCut                     [nphotonscounter]   =  TxC.getTx(photon,3,2);
    t4PtCut                     [nphotonscounter]   =  TxC.getTx(photon,4,2);
    t5PtCut                     [nphotonscounter]   =  TxC.getTx(photon,5,2);


    cc1                         [nphotonscounter]   =  CxC.getCCx(photon.superCluster(),1,0);
    cc2                          [nphotonscounter]   =  CxC.getCCx(photon.superCluster(),2,0);
    cc3                          [nphotonscounter]   =  CxC.getCCx(photon.superCluster(),3,0);
    cc4                          [nphotonscounter]   =  CxC.getCCx(photon.superCluster(),4,0);
    cc5                          [nphotonscounter]   =  CxC.getCCx(photon.superCluster(),5,0);
    cc05                         [nphotonscounter]   =  CxC.getCCx(photon.superCluster(),0.5,0);


    // jurassic cone;
    cc4j                         [nphotonscounter]   =  CxC.getJcc(photon.superCluster(),0.4,0.06,0.04,0);


    // particle flow isolation
    pfcIso1                       [nphotonscounter]    =  pfIso.getPfIso(photon, 1, 0.1, 0.02, 0.0, 0 );
    pfcIso2                       [nphotonscounter]    =  pfIso.getPfIso(photon, 1, 0.2, 0.02, 0.0, 0 );
    pfcIso3                       [nphotonscounter]    =  pfIso.getPfIso(photon, 1, 0.3, 0.02, 0.0, 0 );
    pfcIso4                       [nphotonscounter]    =  pfIso.getPfIso(photon, 1, 0.4, 0.02, 0.0, 0 );
    pfcIso5                       [nphotonscounter]    =  pfIso.getPfIso(photon, 1, 0.5, 0.02, 0.0, 0 );

    pfnIso1                       [nphotonscounter]    =  pfIso.getPfIso(photon, 5, 0.1, 0.0, 0.0, 0 );
    pfnIso2                       [nphotonscounter]    =  pfIso.getPfIso(photon, 5, 0.2, 0.0, 0.0, 0 );
    pfnIso3                       [nphotonscounter]    =  pfIso.getPfIso(photon, 5, 0.3, 0.0, 0.0, 0 );
    pfnIso4                       [nphotonscounter]    =  pfIso.getPfIso(photon, 5, 0.4, 0.0, 0.0, 0 );
    pfnIso5                       [nphotonscounter]    =  pfIso.getPfIso(photon, 5, 0.5, 0.0, 0.0, 0 );

    pfpIso1                       [nphotonscounter]    =  pfIso.getPfIso(photon, 4, 0.1, 0.0, 0.015, 0 );
    pfpIso2                       [nphotonscounter]    =  pfIso.getPfIso(photon, 4, 0.2, 0.0, 0.015, 0 );
    pfpIso3                       [nphotonscounter]    =  pfIso.getPfIso(photon, 4, 0.3, 0.0, 0.015, 0 );
    pfpIso4                       [nphotonscounter]    =  pfIso.getPfIso(photon, 4, 0.4, 0.0, 0.015, 0 );
    pfpIso5                       [nphotonscounter]    =  pfIso.getPfIso(photon, 4, 0.5, 0.0, 0.015, 0 );

    pfsumIso1                       [nphotonscounter]    =  pfcIso1[nphotonscounter] + pfnIso1[nphotonscounter] + pfpIso1[nphotonscounter] ;
    pfsumIso2                       [nphotonscounter]    =  pfcIso2[nphotonscounter] + pfnIso2[nphotonscounter] + pfpIso2[nphotonscounter] ;
    pfsumIso3                       [nphotonscounter]    =  pfcIso3[nphotonscounter] + pfnIso3[nphotonscounter] + pfpIso3[nphotonscounter] ;
    pfsumIso4                       [nphotonscounter]    =  pfcIso4[nphotonscounter] + pfnIso4[nphotonscounter] + pfpIso4[nphotonscounter] ;
    pfsumIso5                       [nphotonscounter]    =  pfcIso5[nphotonscounter] + pfnIso5[nphotonscounter] + pfpIso5[nphotonscounter] ;

    pfVsSubIso1                       [nphotonscounter]    =  0;  //pfIso.getVsPfIso(photon, 0.2, 0.06, 0.04, 0, false );
    pfVsSubIso2                       [nphotonscounter]    =  0; //pfIso.getVsPfIso(photon, 0.2, 0.06, 0.04, 0, false );
    pfVsSubIso3                       [nphotonscounter]    =  0;  //pfIso.getVsPfIso(photon, 0.3, 0.06, 0.04, 0, false );
    pfVsSubIso4                       [nphotonscounter]    =  0;  //pfIso.getVsPfIso(photon, 0.4, 0.06, 0.04, 0, false );
    pfVsSubIso5                   [nphotonscounter]    =  0;  //pfIso.getVsPfIso(photon, 0.5, 0.06, 0.04, 0, false );

    pfcVsIso1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.1, 0.02, 0.0, 0, true);
    pfcVsIso2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.2, 0.02, 0.0, 0, true );
    pfcVsIso3                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.3, 0.02, 0.0, 0, true );
    pfcVsIso4                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.4, 0.02, 0.0, 0, true );
    pfcVsIso5                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.5, 0.02, 0.0, 0, true );
    pfcVsIso1th1                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.1, 0.02, 0.0, 1, true);
    pfcVsIso2th1                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.2, 0.02, 0.0, 1, true );
    pfcVsIso3th1                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.3, 0.02, 0.0, 1, true );
    pfcVsIso4th1                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.4, 0.02, 0.0, 1, true );
    pfcVsIso5th1                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.5, 0.02, 0.0, 1, true );
    pfcVsIso1th2                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.1, 0.02, 0.0, 2, true);
    pfcVsIso2th2                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.2, 0.02, 0.0, 2, true );
    pfcVsIso3th2                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.3, 0.02, 0.0, 2, true );
    pfcVsIso4th2                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.4, 0.02, 0.0, 2, true );
    pfcVsIso5th2                 [nphotonscounter]    =  pfIso.getVsPfIso(photon, 1, 0.5, 0.02, 0.0, 2, true );


    pfnVsIso1                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.1, 0.0, 0.0, 0, true);
    pfnVsIso2                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.2, 0.0, 0.0, 0, true );
    pfnVsIso3                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.3, 0.0, 0.0, 0, true );
    pfnVsIso4                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.4, 0.0, 0.0, 0, true );
    pfnVsIso5                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.5, 0.0, 0.0, 0, true );
    pfnVsIso1th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.1, 0.0, 0.0, 1, true);
    pfnVsIso2th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.2, 0.0, 0.0, 1, true );
    pfnVsIso3th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.3, 0.0, 0.0, 1, true );
    pfnVsIso4th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.4, 0.0, 0.0, 1, true );
    pfnVsIso5th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.5, 0.0, 0.0, 1, true );
    pfnVsIso1th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.1, 0.0, 0.0, 2, true);
    pfnVsIso2th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.2, 0.0, 0.0, 2, true );
    pfnVsIso3th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.3, 0.0, 0.0, 2, true );
    pfnVsIso4th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.4, 0.0, 0.0, 2, true );
    pfnVsIso5th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 5, 0.5, 0.0, 0.0, 2, true );

    pfpVsIso1                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.1,  0.0, 0.015, 0, true);
    pfpVsIso2                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.2,  0.0, 0.015, 0, true );
    pfpVsIso3                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.3,  0.0, 0.015, 0, true );
    pfpVsIso4                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.4,  0.0, 0.015, 0, true );
    pfpVsIso5                       [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.5,  0.0, 0.015, 0, true );
    pfpVsIso1th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.1,  0.0, 0.015, 1, true);
    pfpVsIso2th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.2,  0.0, 0.015, 1, true );
    pfpVsIso3th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.3,  0.0, 0.015, 1, true );
    pfpVsIso4th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.4,  0.0, 0.015, 1, true );
    pfpVsIso5th1                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.5,  0.0, 0.015, 1, true );
    pfpVsIso1th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.1,  0.0, 0.015, 2, true);
    pfpVsIso2th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.2,  0.0, 0.015, 2, true );
    pfpVsIso3th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.3,  0.0, 0.015, 2, true );
    pfpVsIso4th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.4,  0.0, 0.015, 2, true );
    pfpVsIso5th2                    [nphotonscounter]    =  pfIso.getVsPfIso(photon, 4, 0.5,  0.0, 0.015, 2, true );


    pfsumVsIso1       [nphotonscounter]    =  pfcVsIso1[nphotonscounter] + pfnVsIso1[nphotonscounter] + pfpVsIso1[nphotonscounter] ;
    pfsumVsIso2       [nphotonscounter]    =  pfcVsIso2[nphotonscounter] + pfnVsIso2[nphotonscounter] + pfpVsIso2[nphotonscounter] ;
    pfsumVsIso3       [nphotonscounter]    =  pfcVsIso3[nphotonscounter] + pfnVsIso3[nphotonscounter] + pfpVsIso3[nphotonscounter] ;
    pfsumVsIso4       [nphotonscounter]    =  pfcVsIso4[nphotonscounter] + pfnVsIso4[nphotonscounter] + pfpVsIso4[nphotonscounter] ;
    pfsumVsIso5       [nphotonscounter]    =  pfcVsIso5[nphotonscounter] + pfnVsIso5[nphotonscounter] + pfpVsIso5[nphotonscounter] ;
    pfsumVsIso1th1    [nphotonscounter]    =  pfcVsIso1th1[nphotonscounter] + pfnVsIso1th1[nphotonscounter] + pfpVsIso1th1[nphotonscounter] ;
    pfsumVsIso2th1    [nphotonscounter]    =  pfcVsIso2th1[nphotonscounter] + pfnVsIso2th1[nphotonscounter] + pfpVsIso2th1[nphotonscounter] ;
    pfsumVsIso3th1    [nphotonscounter]    =  pfcVsIso3th1[nphotonscounter] + pfnVsIso3th1[nphotonscounter] + pfpVsIso3th1[nphotonscounter] ;
    pfsumVsIso4th1    [nphotonscounter]    =  pfcVsIso4th1[nphotonscounter] + pfnVsIso4th1[nphotonscounter] + pfpVsIso4th1[nphotonscounter] ;
    pfsumVsIso5th1    [nphotonscounter]    =  pfcVsIso5th1[nphotonscounter] + pfnVsIso5th1[nphotonscounter] + pfpVsIso5th1[nphotonscounter] ;
    pfsumVsIso1th2    [nphotonscounter]    =  pfcVsIso1th2[nphotonscounter] + pfnVsIso1th2[nphotonscounter] + pfpVsIso1th2[nphotonscounter] ;
    pfsumVsIso2th2    [nphotonscounter]    =  pfcVsIso2th2[nphotonscounter] + pfnVsIso2th2[nphotonscounter] + pfpVsIso2th2[nphotonscounter] ;
    pfsumVsIso3th2    [nphotonscounter]    =  pfcVsIso3th2[nphotonscounter] + pfnVsIso3th2[nphotonscounter] + pfpVsIso3th2[nphotonscounter] ;
    pfsumVsIso4th2    [nphotonscounter]    =  pfcVsIso4th2[nphotonscounter] + pfnVsIso4th2[nphotonscounter] + pfpVsIso4th2[nphotonscounter] ;
    pfsumVsIso5th2    [nphotonscounter]    =  pfcVsIso5th2[nphotonscounter] + pfnVsIso5th2[nphotonscounter] + pfpVsIso5th2[nphotonscounter] ;

    // tower iso
    towerIso1                       [nphotonscounter]    =  towerIso.getTowerIso(photon, 0.1, 0.06, 0.04, 0 );
    towerIso2                       [nphotonscounter]    =  towerIso.getTowerIso(photon, 0.2, 0.06, 0.04, 0 );
    towerIso3                       [nphotonscounter]    =  towerIso.getTowerIso(photon, 0.3, 0.06, 0.04, 0 );
    towerIso4                       [nphotonscounter]    =  towerIso.getTowerIso(photon, 0.4, 0.06, 0.04, 0 );
    towerIso5                       [nphotonscounter]    =  towerIso.getTowerIso(photon, 0.5, 0.06, 0.04, 0 );

    towerVsSubIso1                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.1, 0.06, 0.04, 0, false );
    towerVsSubIso2                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.2, 0.06, 0.04, 0, false );
    towerVsSubIso3                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.3, 0.06, 0.04, 0, false );
    towerVsSubIso4                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.4, 0.06, 0.04, 0, false );
    towerVsSubIso5                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.5, 0.06, 0.04, 0, false );

    towerVsIso1                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.1, 0.06, 0.04, 0, true );
    towerVsIso2                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.2, 0.06, 0.04, 0, true );
    towerVsIso3                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.3, 0.06, 0.04, 0, true );
    towerVsIso4                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.4, 0.06, 0.04, 0, true );
    towerVsIso5                       [nphotonscounter]    =  towerIso.getVsTowerIso(photon, 0.5, 0.06, 0.04, 0, true );






    ct1                          [nphotonscounter]   =  TxC.getCTx(photon,1,0);
    ct2                          [nphotonscounter]   =  TxC.getCTx(photon,2,0);
    ct3                          [nphotonscounter]   =  TxC.getCTx(photon,3,0);
    ct4                          [nphotonscounter]   =  TxC.getCTx(photon,4,0);
    ct5                          [nphotonscounter]   =  TxC.getCTx(photon,5,0);
    //  ct05                         (nphotonscounter)   =  TxC.getCTx(photon,0.5,0);

    cr1                          [nphotonscounter]   =  RxC.getCRx(photon.superCluster(),1,0);
    cr2                          [nphotonscounter]   =  RxC.getCRx(photon.superCluster(),2,0);
    cr3                          [nphotonscounter]   =  RxC.getCRx(photon.superCluster(),3,0);
    cr4                          [nphotonscounter]   =  RxC.getCRx(photon.superCluster(),4,0);
    cr5                          [nphotonscounter]   =  RxC.getCRx(photon.superCluster(),5,0);
    cr4j                         [nphotonscounter]   =  RxC.getCRx(photon.superCluster(),4,0,0.15);
    // cr05                         (nphotonscounter)   =  RxC.getCRx(photon.superCluster(),0.5,0);

    ct1PtCut20                     [nphotonscounter]   =  TxC.getCTx(photon,1,2); // 2 GeV cut
    ct2PtCut20                     [nphotonscounter]   =  TxC.getCTx(photon,2,2);
    ct3PtCut20                     [nphotonscounter]   =  TxC.getCTx(photon,3,2);
    ct4PtCut20                     [nphotonscounter]   =  TxC.getCTx(photon,4,2);
    ct5PtCut20                     [nphotonscounter]   =  TxC.getCTx(photon,5,2);
    // ct05PtCut                    (nphotonscounter)   =  TxC.getCTx(photon,0.5,2);

    //    trackIsohi                   (nphotonscounter)   =  TxC.getTx(photon,4, 0.0, 0.04);
    //  trackIsohi10                 (nphotonscounter)   =  TxC.getTx(photon,4, 1.0, 0.04);
    //   trackIsohi15                 (nphotonscounter)   =  TxC.getTx(photon,4, 1.5, 0.04);
    //  trackIsohi20                 (nphotonscounter)   =  TxC.getTx(photon,4, 2.0, 0.04);

    //   trackIsohij                   (nphotonscounter)   =  TxC.getJt(photon, 0.4, 0.04, 0.015,0.0);
    //  trackIsohi10j                 (nphotonscounter)   =  TxC.getJt(photon, 0.4, 0.04, 0.015,1.0);
    //  trackIsohi15j                 (nphotonscounter)   =  TxC.getJt(photon, 0.4, 0.04, 0.015,1.5);
    //  trackIsohi20j                 (nphotonscounter)   =  TxC.getJt(photon, 0.4, 0.04, 0.015,2.0);

    ct4j                          [nphotonscounter]   =  TxC.getJct(photon,0.4, 0.04, 0.015,0);
    ct4j10                        [nphotonscounter]   =  TxC.getJct(photon,0.4, 0.04, 0.015,1.0);
    ct4j15                        [nphotonscounter]   =  TxC.getJct(photon,0.4, 0.04, 0.015,1.5);
    ct4j20                        [nphotonscounter]   =  TxC.getJct(photon,0.4, 0.04, 0.015,2.0);

    dr11                         [nphotonscounter]   =  dRxy.getDRxy(photon,1,1);
    dr12                         [nphotonscounter]   =  dRxy.getDRxy(photon,1,2);
    dr13                         [nphotonscounter]   =  dRxy.getDRxy(photon,1,3);
    dr14                         [nphotonscounter]   =  dRxy.getDRxy(photon,1,4);

    dr21                         [nphotonscounter]   =  dRxy.getDRxy(photon,2,1);
    dr22                         [nphotonscounter]   =  dRxy.getDRxy(photon,2,2);
    dr23                         [nphotonscounter]   =  dRxy.getDRxy(photon,2,3);
    dr24                         [nphotonscounter]   =  dRxy.getDRxy(photon,2,4);

    dr31                         [nphotonscounter]   =  dRxy.getDRxy(photon,3,1);
    dr32                         [nphotonscounter]   =  dRxy.getDRxy(photon,3,2);
    dr33                         [nphotonscounter]   =  dRxy.getDRxy(photon,3,3);
    dr34                         [nphotonscounter]   =  dRxy.getDRxy(photon,3,4);

    dr41                         [nphotonscounter]   =  dRxy.getDRxy(photon,4,1);
    dr42                         [nphotonscounter]   =  dRxy.getDRxy(photon,4,2);
    dr43                         [nphotonscounter]   =  dRxy.getDRxy(photon,4,3);
    dr44                         [nphotonscounter]   =  dRxy.getDRxy(photon,4,4);

    t11                         [nphotonscounter]   =  Txy.getTxy(photon,1,1);
    t12                         [nphotonscounter]   =  Txy.getTxy(photon,1,2);
    t13                         [nphotonscounter]   =  Txy.getTxy(photon,1,3);
    t14                         [nphotonscounter]   =  Txy.getTxy(photon,1,4);

    t21                         [nphotonscounter]   =  Txy.getTxy(photon,2,1);
    t22                         [nphotonscounter]   =  Txy.getTxy(photon,2,2);
    t23                         [nphotonscounter]   =  Txy.getTxy(photon,2,3);
    t24                         [nphotonscounter]   =  Txy.getTxy(photon,2,4);

    t31                         [nphotonscounter]   =  Txy.getTxy(photon,3,1);
    t32                         [nphotonscounter]   =  Txy.getTxy(photon,3,2);
    t33                         [nphotonscounter]   =  Txy.getTxy(photon,3,3);
    t34                         [nphotonscounter]   =  Txy.getTxy(photon,3,4);

    t41                         [nphotonscounter]   =  Txy.getTxy(photon,4,1);
    t42                         [nphotonscounter]   =  Txy.getTxy(photon,4,2);
    t43                         [nphotonscounter]   =  Txy.getTxy(photon,4,3);
    t44                         [nphotonscounter]   =  Txy.getTxy(photon,4,4);


    //    nAllTracks                  (nphotonscounter)   =  (float)Txy.getNumAllTracks(1);   // pt Cut of the track = 1GeV
    //  nLocalTracks                (nphotonscounter)   =  (float)Txy.getNumLocalTracks(photon,0.5,1); // dEta cut = 0.5     and    pt Cut = 1GeV
    hasPixelSeed       [nphotonscounter]   =  photon.hasPixelSeed();



    //////////////////////////////////    // MC matching /////////////////////////////////////////
    isGenMatched[nphotonscounter] = kFALSE;
    genMomId[nphotonscounter] = 0;
    genGrandMomId[nphotonscounter] = 0;
    genNSiblings[nphotonscounter] = 0;
    genMatchedPt[nphotonscounter] = -1000;
    genMatchedEta[nphotonscounter] = -1000;
    genMatchedPhi[nphotonscounter] = -1000;
    genMatchedCollId[nphotonscounter] = -1000;
    genCalIsoDR03[nphotonscounter] = 99999.;
    genTrkIsoDR03[nphotonscounter] = 99999.;
    genCalIsoDR04[nphotonscounter] = 99999.;
    genTrkIsoDR04[nphotonscounter] = 99999.;

    if (isMC_) {
      edm::Handle<reco::GenParticleCollection> genParticles;

      //  get generated particles and store generator ntuple
      try { e.getByLabel( genParticleProducer_,      genParticles );} catch (...) {;}

      float delta(0.15);

      const reco::Candidate *cndMc(0);

      bool gpTemp(false);
      reco::GenParticleCollection::const_iterator matchedPart;
      Float_t currentMaxPt(-1);

      for (reco::GenParticleCollection::const_iterator it_gen =
	     genParticles->begin(); it_gen!= genParticles->end(); it_gen++){

	const reco::Candidate &p = (*it_gen);
	if (p.status() != 1 || (p.pdgId()) != pdgId_ ) continue;
	if(ROOT::Math::VectorUtil::DeltaR(p.p4(),phoItr->p4())<delta && p.pt() > currentMaxPt ) {

	  gpTemp = true;
	  cndMc = &p;
	  currentMaxPt = p.pt();
	  matchedPart  = it_gen;
	}
      }


      // if no matching photon was found try with other particles
      if( !gpTemp ) {

	currentMaxPt = -1;
	for (reco::GenParticleCollection::const_iterator it_gen =
	       genParticles->begin(); it_gen!= genParticles->end(); it_gen++){
	  const reco::Candidate &p = (*it_gen);

	  if (p.status() != 1 || find(otherPdgIds_.begin(),otherPdgIds_.end(),fabs(p.pdgId())) == otherPdgIds_.end() ) continue;
	  if(ROOT::Math::VectorUtil::DeltaR(p.p4(),phoItr->p4())<delta && p.pt() > currentMaxPt ) {

	    cndMc = &p; // do not set the isGenMatched in this case
	    currentMaxPt = p.pt();
	    matchedPart  = it_gen;
	  }

	} // end of loop over gen particles
      } // if not matched to gen photon

      if(cndMc) {
	// genMatchedP4 [nphotonscounter]  = TLorentzVector(cndMc->px(),cndMc->py(),cndMc->pz(),cndMc->energy());
	genMatchedPt [nphotonscounter]  = cndMc->pt();
	genMatchedEta[nphotonscounter]  = cndMc->eta();
	genMatchedPhi[nphotonscounter]  = cndMc->phi();
	//cout <<"Matched!" << cndMc->pt()<<endl;

	genCalIsoDR03[nphotonscounter]= getGenCalIso(genParticles,matchedPart,0.3);
	genTrkIsoDR03[nphotonscounter]= getGenTrkIso(genParticles,matchedPart,0.3);
	genCalIsoDR04[nphotonscounter]= getGenCalIso(genParticles,matchedPart,0.4);
	genTrkIsoDR04[nphotonscounter]= getGenTrkIso(genParticles,matchedPart,0.4);

	isGenMatched[nphotonscounter] = gpTemp;
	genMatchedCollId[nphotonscounter] = matchedPart->collisionId();

	if( cndMc->numberOfMothers() > 0 ) {
	  genMomId[nphotonscounter] = cndMc->mother()->pdgId();
	  genNSiblings[nphotonscounter] = cndMc->mother()->numberOfDaughters();
	  if( cndMc->mother()->numberOfMothers() > 0 ) {
	    genGrandMomId[nphotonscounter] = cndMc->mother()->mother()->pdgId();
	  }
	}
      }

    } // if it's a MC
    if (nphotonscounter>kMaxPhotons-1) break;

    nphotonscounter++;
  }

  nPho  = nphotonscounter;

  return (int)nphotonscounter;

}



DEFINE_FWK_MODULE(MultiPhotonAnalyzerTree);
