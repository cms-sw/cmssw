// system include files
#include <memory>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"


class IsoTrackCalib : public edm::EDAnalyzer {

public:
  explicit IsoTrackCalib(const edm::ParameterSet&);
  ~IsoTrackCalib();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  double dR(double eta1, double eta2, double phi1, double phi2);

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  double dPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dP(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dinvPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dEta(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dPhi(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);

  // ----------member data ---------------------------
  bool                       changed;
  edm::Service<TFileService> fs;
  HLTConfigProvider          hltConfig_;
  std::vector<std::string>   trigNames, HLTNames;
  int                        verbosity;
  spr::trackSelectionParameters selectionParameters;
  std::string                theTrackQuality;
  double                     dr_L1, a_mipR, a_coneR, a_charIsoR, a_neutIsoR;
  double                     a_neutR1, a_neutR2, cutMip, cutCharge, cutNeutral;
  int                        minRunNo, maxRunNo, nRun;
  std::vector<double>        drCuts;

  edm::InputTag              triggerEvent_, theTriggerResultsLabel;
  edm::EDGetTokenT<LumiDetails>            tok_lumi;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes;
 
  edm::EDGetTokenT<reco::TrackCollection>  tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>         tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>   tok_hbhe_;

  bool                       firstEvent;
  std::map<std::pair<unsigned int, std::string>, unsigned int> TrigList;
  std::map<std::pair<unsigned int, std::string>, const std::pair<int, int> > TrigPreList;
  TH1I                       *h_nHLT, *h_HLTAccept;
  std::vector<TH1I*>         h_HLTAccepts;
  TH1I                       *g_Pre, *g_PreL1, *g_PreHLT, *g_Accepts;
};

IsoTrackCalib::IsoTrackCalib(const edm::ParameterSet& iConfig) : changed(false),
								 nRun(0) {
   //now do whatever initialization is needed
  verbosity                           = iConfig.getUntrackedParameter<int>("Verbosity",0);
  trigNames                           = iConfig.getUntrackedParameter<std::vector<std::string> >("Triggers");
  theTrackQuality                     = iConfig.getUntrackedParameter<std::string>("TrackQuality","highPurity");
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);
  selectionParameters.minPt           = iConfig.getUntrackedParameter<double>("MinTrackPt", 10.0);
  selectionParameters.minQuality      = trackQuality_;
  selectionParameters.maxDxyPV        = iConfig.getUntrackedParameter<double>("MaxDxyPV", 0.2);
  selectionParameters.maxDzPV         = iConfig.getUntrackedParameter<double>("MaxDzPV",  5.0);
  selectionParameters.maxChi2         = iConfig.getUntrackedParameter<double>("MaxChi2",  5.0);
  selectionParameters.maxDpOverP      = iConfig.getUntrackedParameter<double>("MaxDpOverP",  0.1);
  selectionParameters.minOuterHit     = iConfig.getUntrackedParameter<int>("MinOuterHit", 4);
  selectionParameters.minLayerCrossed = iConfig.getUntrackedParameter<int>("MinLayerCrossed", 8);
  selectionParameters.maxInMiss       = iConfig.getUntrackedParameter<int>("MaxInMiss", 0);
  selectionParameters.maxOutMiss      = iConfig.getUntrackedParameter<int>("MaxOutMiss", 0);
  dr_L1                               = iConfig.getUntrackedParameter<double>("IsolationL1",1.0);
  a_coneR                             = iConfig.getUntrackedParameter<double>("ConeRadius",34.98);
  a_charIsoR                          = a_coneR + 28.9;
  a_neutIsoR                          = a_charIsoR*0.726;
  a_mipR                              = iConfig.getUntrackedParameter<double>("ConeRadiusMIP",14.0);
  a_neutR1                            = iConfig.getUntrackedParameter<double>("ConeRadiusNeut1",21.0);
  a_neutR2                            = iConfig.getUntrackedParameter<double>("ConeRadiusNeut2",29.0);
  cutMip                              = iConfig.getUntrackedParameter<double>("MIPCut", 1.0);
  cutCharge                           = iConfig.getUntrackedParameter<double>("ChargeIsolation",  2.0);
  cutNeutral                          = iConfig.getUntrackedParameter<double>("NeutralIsolation",  2.0);
  minRunNo                            = iConfig.getUntrackedParameter<int>("minRun");
  maxRunNo                            = iConfig.getUntrackedParameter<int>("maxRun");
  drCuts                              = iConfig.getUntrackedParameter<std::vector<double> >("DRCuts");
  bool isItAOD                        = iConfig.getUntrackedParameter<bool>("IsItAOD", false);
  triggerEvent_                       = edm::InputTag("hltTriggerSummaryAOD","","HLT");
 theTriggerResultsLabel               = edm::InputTag("TriggerResults","","HLT");

  // define tokens for access
  tok_lumi      = consumes<LumiDetails>(edm::InputTag("lumiProducer"));
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_       = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  if (isItAOD) {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }

  if (verbosity>=0) {
    std::cout <<"Parameters read from config file \n" 
	      <<"\t minPt "           << selectionParameters.minPt   
              <<"\t theTrackQuality " << theTrackQuality
	      <<"\t minQuality "      << selectionParameters.minQuality
	      <<"\t maxDxyPV "        << selectionParameters.maxDxyPV          
	      <<"\t maxDzPV "         << selectionParameters.maxDzPV          
	      <<"\t maxChi2 "         << selectionParameters.maxChi2          
	      <<"\t maxDpOverP "      << selectionParameters.maxDpOverP
	      <<"\t minOuterHit "     << selectionParameters.minOuterHit
	      <<"\t minLayerCrossed " << selectionParameters.minLayerCrossed
	      <<"\t maxInMiss "       << selectionParameters.maxInMiss
	      <<"\t maxOutMiss "      << selectionParameters.maxOutMiss
	      <<"\t a_coneR "         << a_coneR          
	      <<"\t a_charIsoR "      << a_charIsoR          
	      <<"\t a_neutIsoR "      << a_neutIsoR          
	      <<"\t a_mipR "          << a_mipR 
	      <<"\t a_neutR "         << a_neutR1 << ":" << a_neutR2
              <<"\t cuts (MIP "       << cutMip << " : Charge " << cutCharge
	      <<" : Neutral "         << cutNeutral << ")"
	      << std::endl;
    std::cout << trigNames.size() << " triggers to be studied";
    for (unsigned int k=0; k<trigNames.size(); ++k)
      std::cout << ": " << trigNames[k];
    std::cout << std::endl;
    std::cout << drCuts.size() << " Delta R zones wrt trigger objects";
    for (unsigned int k=0; k<drCuts.size(); ++k)
      std::cout << ": " << drCuts[k];
    std::cout << std::endl;
  }
}

IsoTrackCalib::~IsoTrackCalib() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}

void IsoTrackCalib::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (verbosity%10 > 0) 
    std::cout << "Run " << iEvent.id().run() << " Event " << iEvent.id().event()
	      << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
	      << iEvent.bunchCrossing() << " starts ==========" << std::endl;
  int RunNo = iEvent.id().run();

  //Get magnetic field and ECAL channel status
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField *bField = bFieldH.product();
  /*
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
  const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();
  */
  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  /*
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
  const CaloTopology *caloTopology = theCaloTopology.product();

  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<IdealGeometryRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();
  */
  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  reco::TrackCollection::const_iterator trkItr;
  
  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);  
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0,0,0);
  if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint( (*recVtxs)[0].x(),(*recVtxs)[0].y(), (*recVtxs)[0].z() );
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
  if ((verbosity/100)%10>0) {
    std::cout << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid()) std::cout << " Beam Spot " 
				       << beamSpotH->position();
    std::cout << std::endl;
  }
  
  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);

  edm::Handle<LumiDetails> Lumid;
  iEvent.getLuminosityBlock().getByToken(tok_lumi, Lumid); 
  float mybxlumi=-1;
  if (Lumid.isValid()) 
    mybxlumi=Lumid->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
  if (verbosity%10 > 0) std::cout << "Luminosity " << mybxlumi << std::endl;

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    std::cout << "Error! Can't get the product "<< triggerEvent_.label() 
	      << std::endl;
  } else {
    triggerEvent = *(triggerEventHandle.product());
  
    const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes, triggerResults);
    if (triggerResults.isValid()) {
      std::vector<std::string> modules;
      h_nHLT->Fill(triggerResults->size());
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);

      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      bool ok(false);
      for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
	unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[iHLT]);
	const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
	int ipos = -1;
	for (unsigned int i=0; i<HLTNames.size(); ++i) {
	  if (triggerNames_[iHLT] == HLTNames[i]) {
	    ipos = i;
	    break;
	  }
	}
	if (ipos < 0) {
	  ipos = (int)(HLTNames.size()+1);
	  HLTNames.push_back(triggerNames_[iHLT]);
	  h_HLTAccept->GetXaxis()->SetBinLabel(ipos+1,triggerNames_[iHLT].c_str());
	}
	if (firstEvent)  h_HLTAccepts[nRun]->GetXaxis()->SetBinLabel(iHLT+1, triggerNames_[iHLT].c_str());
	int hlt    = triggerResults->accept(iHLT);
	if (hlt) {
	  h_HLTAccepts[nRun]->Fill(iHLT+1);
	  h_HLTAccept->Fill(ipos+1);
	}
	if (iHLT >= 499) std::cout << "Wrong trigger " << RunNo << " Event " 
				   << iEvent.id().event() << " Hlt " << iHLT 
				   << std::endl;
	for (unsigned int i=0; i<trigNames.size(); ++i) {
	  if (triggerNames_[iHLT].find(trigNames[i].c_str())!=std::string::npos) {
	    if(verbosity)  std::cout << triggerNames_[iHLT] << std::endl;
	    if (hlt > 0) ok = true;
	  }
	}

	if (ok) {
	  std::vector<math::XYZTLorentzVector> vec[3];
	  const std::pair<int,int> prescales(hltConfig_.prescaleValues(iEvent,iSetup,triggerNames_[iHLT]));
	  int preL1  = prescales.first;
	  int preHLT = prescales.second;
	  int prescale = preL1*preHLT;
	  if (verbosity%10 > 0)
	    std::cout << triggerNames_[iHLT] << " accept " << hlt << " preL1 " 
		      << preL1 << " preHLT " << preHLT << " preScale " 
		      << prescale << std::endl;
	  std::pair<unsigned int, std::string> iRunTrig =
	    std::pair<unsigned int, std::string>(RunNo,triggerNames_[iHLT]);
	  if (TrigList.find(iRunTrig) != TrigList.end() ) {
	    TrigList[iRunTrig] += 1;
	  } else {
	    TrigList.insert(std::pair<std::pair<unsigned int, std::string>, unsigned int>(iRunTrig,1));
	    TrigPreList.insert(std::pair<std::pair<unsigned int, std::string>, std::pair<int, int>>(iRunTrig,prescales));
	  }
	  //loop over all trigger filters in event (i.e. filters passed)
	  for (unsigned int ifilter=0; ifilter<triggerEvent.sizeFilters(); ++ifilter) {  
	    std::vector<int> Keys;
	    std::string label = triggerEvent.filterTag(ifilter).label();
	    //loop over keys to objects passing this filter
	    for (unsigned int imodule=0; imodule<moduleLabels.size(); imodule++) {
	      if (label.find(moduleLabels[imodule]) != std::string::npos) {
		if (verbosity%10 > 0) std::cout << "FilterName " << label << std::endl;
		for (unsigned int ifiltrKey=0; ifiltrKey<triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
		  Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
		  const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
		  math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
		  if(label.find("L2Filter") != std::string::npos) {
		    vec[1].push_back(v4);
		  } else if (label.find("Filter") != std::string::npos) {
		    vec[2].push_back(v4);
		  } else {
		    vec[0].push_back(v4);
		  }
		  if (verbosity%10 > 0)
		    std::cout << "key " << ifiltrKey << " : pt " << TO.pt() << " eta " << TO.eta() << " phi " << TO.phi() << " mass " << TO.mass() << " Id " << TO.id() << std::endl;
		}
	      }
	    }
	  }

	  //// Filling Pt, eta, phi of L1 and L2 objects
	  for (int j=0; j<2; j++) {
	    for (unsigned int k=0; k<vec[j].size(); k++) {
	      if (verbosity%10 > 0) std::cout << "vec[" << j << "][" << k << "] pt " << vec[j][k].pt() << " eta " << vec[j][k].eta() << " phi " << vec[j][k].phi() << std::endl;
	    }
	  }

	  double deta, dphi, dr;
	  //// deta, dphi and dR for leading L1 object with L2 objects
	  math::XYZTLorentzVector mindRvec1;
	  double mindR1(999);
	  for (int lvl=1; lvl<2; lvl++) {
	    for (unsigned int i=0; i<vec[lvl].size(); i++) {
	      deta = dEta(vec[0][0],vec[lvl][i]);
	      dphi = dPhi(vec[0][0],vec[lvl][i]);
	      dr   = dR(vec[0][0],vec[lvl][i]);
	      if (verbosity%10 > 0) std::cout << "lvl " <<lvl << " i " << i 
					      << " deta " << deta << " dphi " 
					      << dphi << " dR " << dr 
					      << std::endl;
	      if (dr<mindR1) {
		mindR1    = dr;
		mindRvec1 = vec[lvl][i];
	      }
	    }
	  }

	  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
	  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections, ((verbosity/100)%10>2));
	  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
	  unsigned int nTracks=0;
	  for (trkDetItr = trkCaloDirections.begin(),nTracks=0; 
	       trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++) {
	    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
	    math::XYZTLorentzVector v4(trkItr->px(), trkItr->py(), 
				       trkItr->pz(), trkItr->p());

	    math::XYZTLorentzVector mindRvec2;
	    double mindR2(999);
	    for (unsigned int k=0; k<vec[1].size(); ++k) {
	      //// Find min of deta/dphi/dR for each of track with L2 objects
	      if (verbosity%10 > 0) std::cout << "L2obj: pt " << vec[1][k].pt() << " eta " << vec[1][k].eta() << " phi " << vec[1][k].phi() << std::endl;
	      dr   = dR(vec[1][k],v4);
	      if (dr<mindR2) {
		mindR2    = dr;
		mindRvec2 = vec[1][k];
	      }
	    }
	    double mindR = dR(mindRvec1,v4);

	    unsigned int i1 = drCuts.size();
	    for (unsigned int ik=0; ik<drCuts.size(); ++ik) {
	      if (mindR < drCuts[ik]) {
		i1 = ik; break;
	      }
	    }
	    unsigned int i2 = drCuts.size();
	    for (unsigned int ik=0; ik<drCuts.size(); ++ik) {
	      if (mindR2 < drCuts[ik]) {
		i2 = ik; break;
	      }
	    }

	    //Selection of good track
	    bool selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>2));
	    int ieta(0);
	    if (trkDetItr->okHCAL) {
	      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
	      ieta = detId.ieta();
	    }
	    if (verbosity%10 > 0) std::cout << "iEta " << ieta << " Classify "
					    << i1 << ":" << i2 << std::endl;

	    if (selectTk && trkDetItr->okECAL && trkDetItr->okHCAL) {
	      int nRH_eMipDR=0, nNearTRKs=0;
	      double e1 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
					   trkDetItr->pointHCAL, trkDetItr->pointECAL,
					   a_neutR1, trkDetItr->directionECAL, nRH_eMipDR);
	      double e2 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
					   trkDetItr->pointHCAL, trkDetItr->pointECAL,
					   a_neutR2, trkDetItr->directionECAL, nRH_eMipDR);
	      double eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
					      trkDetItr->pointHCAL, trkDetItr->pointECAL,
					      a_mipR, trkDetItr->directionECAL, nRH_eMipDR);
	      int ietaHotCell(-99), iphiHotCell(-99), nRecHitsCone(-999);
	      double distFromHotCell(-99.0);
	      std::vector<DetId>    coneRecHitDetIds;
	      GlobalPoint           gposHotCell(0.,0.,0.);
	      double eHcal = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, trkDetItr->pointECAL,
					     a_coneR, trkDetItr->directionHCAL, nRecHitsCone, 
					     coneRecHitDetIds, distFromHotCell, 
					     ietaHotCell, iphiHotCell, gposHotCell);
					     
	      double conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR, nNearTRKs, ((verbosity/100)%10>2));
	      double e_inCone = e2 - e1;
	      if (verbosity%10 > 0) std::cout << "Neutral isolation " << e2
					      << ":" << e1 << ":" << e_inCone
					      << " Charge isolation " 
					      << conehmaxNearP << " eMIP "
					      << eMipDR << " eHCAL " << eHcal
					      << std::endl;
	    }
	  }
	}
      }
      

      // check if trigger names in (new) config                       
      if (changed) {
	changed = false;
	if ((verbosity/10)%10 > 1) {
	  std::cout<<"New trigger menu found !!!" << std::endl;
	  const unsigned int n(hltConfig_.size());
	  for (unsigned itrig=0; itrig<triggerNames_.size(); itrig++) {
	    unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[itrig]);
	    std::cout << triggerNames_[itrig] << " " << triggerindx << " ";
	    if (triggerindx >= n)
	      std::cout << "does not exist in the current menu" << std::endl;
	    else
	      std::cout << "exists" << std::endl;
	  }
	}
      }
    }
  }
}

void IsoTrackCalib::beginJob() {
  h_nHLT        = fs->make<TH1I>("h_nHLT" , "size of trigger Names", 1000, 0, 1000);
  h_HLTAccept   = fs->make<TH1I>("h_HLTAccept", "HLT Accepts for all runs", 1000, 0, 1000);
}
// ------------ method called once each job just after ending the event loop  ------------
void IsoTrackCalib::endJob() {
  unsigned int preL1, preHLT;
  std::map<std::pair<unsigned int, std::string>, unsigned int>::iterator itr;
  std::map<std::pair<unsigned int, std::string>, const std::pair<int, int>>::iterator itrPre;
  std::cout << "RunNo vs HLT accepts" << std::endl;
  unsigned int n = maxRunNo - minRunNo +1;
  g_Pre = fs->make<TH1I>("h_PrevsRN", "PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreL1 = fs->make<TH1I>("h_PreL1vsRN", "L1 PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreHLT = fs->make<TH1I>("h_PreHLTvsRN", "HLT PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_Accepts = fs->make<TH1I>("h_HLTAcceptsvsRN", "HLT Accepts Vs Run Number", n, minRunNo, maxRunNo); 

  for (itr=TrigList.begin(), itrPre=TrigPreList.begin(); itr!=TrigList.end(); itr++, itrPre++) {
    preL1 = (itrPre->second).first;
    preHLT = (itrPre->second).second;
    std::cout << (itr->first).first << " " << (itr->first).second << " " 
	      << itr->second << " " << (itrPre->first).first << " " 
	      << (itrPre->first).second << " " << preL1 << " " << preHLT 
	      << std::endl;
    g_Accepts->Fill((itr->first).first, itr->second);
    g_PreL1->Fill((itr->first).first, preL1);
    g_PreHLT->Fill((itr->first).first, preHLT);
    g_Pre->Fill((itr->first).first, preL1*preHLT);
  }
}

// ------------ method called when starting to processes a run  ------------
void IsoTrackCalib::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  std::cout  << "Run[" << nRun <<"] " << iRun.run() << " hltconfig.init " 
	     << hltConfig_.init(iRun,iSetup,"HLT",changed) << std::endl;;
  char  hname[100], htit[100];
  sprintf(hname, "h_HLTAccepts_%i", iRun.run());
  sprintf(htit, "HLT Accepts for Run No %i", iRun.run());
  TH1I *hnew = fs->make<TH1I>(hname, htit, 1000, 0, 1000);
  h_HLTAccepts.push_back(hnew);
  firstEvent = true;
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrackCalib::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  std::cout << "endRun[" << nRun << "] " << iRun.run() << std::endl;
}

// ------------ method called when starting to processes a luminosity block  ------------
void IsoTrackCalib::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void IsoTrackCalib::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void IsoTrackCalib::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

double IsoTrackCalib::dEta(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.eta()-vec2.eta());
}

double IsoTrackCalib::dPhi(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {

  double phi1 = vec1.phi();
  if (phi1 < 0) phi1 += 2.0*M_PI;
  double phi2 = vec2.phi();
  if (phi2 < 0) phi2 += 2.0*M_PI;
  double dphi = phi1-phi2;
  if (dphi > M_PI)       dphi -= 2.*M_PI;
  else if (dphi < -M_PI) dphi += 2.*M_PI;
  return dphi;
}

double IsoTrackCalib::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  double deta = dEta(vec1,vec2);
  double dphi = dPhi(vec1,vec2);
  return std::sqrt(deta*deta + dphi*dphi);
}

double IsoTrackCalib::dPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.pt()-vec2.pt());
}

double IsoTrackCalib::dP(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (std::abs(vec1.r()-vec2.r()));
}

double IsoTrackCalib::dinvPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return ((1/vec1.pt())-(1/vec2.pt()));
}
//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrackCalib);
