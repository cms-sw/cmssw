// -*- C++ -*-
//
// Package:    IsoTrig
// Class:      IsoTrig
// 
/**\class IsoTrig IsoTrig.cc IsoTrig/IsoTrig/src/IsoTrig.cc

v Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ruchi Gupta
//         Created:  Fri May 25 12:02:48 CDT 2012
// $Id: IsoTrig.cc,v 1.1 2012/10/10 15:04:59 sunanda Exp $
//
//
#include "IsoTrig.h"

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

#include "FWCore/Common/interface/TriggerNames.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"

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

IsoTrig::IsoTrig(const edm::ParameterSet& iConfig) : changed(false) {
   //now do whatever initialization is needed
  Det                                 = iConfig.getParameter<std::string>("Det");
  verbosity                           = iConfig.getUntrackedParameter<int>("Verbosity",0);
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
  }
}

IsoTrig::~IsoTrig() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}

void IsoTrig::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (verbosity%10 > 0) std::cout << "Event starts====================================" << std::endl;
  int RunNo = iEvent.id().run();
  int EvtNo = iEvent.id().event();
  int Lumi  = iEvent.luminosityBlock();
  int Bunch = iEvent.bunchCrossing();

  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField *bField = bFieldH.product();

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByLabel("generalTracks", trkCollection);
  reco::TrackCollection::const_iterator trkItr;

  edm::InputTag lumiProducer("LumiProducer", "", "RECO");
  edm::Handle<LumiDetails> Lumid;
  iEvent.getLuminosityBlock().getByLabel("lumiProducer",Lumid); 
  float mybxlumi=-1;
  if (Lumid.isValid()) 
    mybxlumi=Lumid->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
  if (verbosity%10 > 0)
    std::cout << "RunNo " << RunNo << " EvtNo " << EvtNo << " Lumi " << Lumi 
	      << " Bunch " << Bunch << " mybxlumi " << mybxlumi << std::endl;

  edm::InputTag triggerEvent_ ("hltTriggerSummaryAOD","","HLT");
  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByLabel(triggerEvent_,triggerEventHandle);
  if (!triggerEventHandle.isValid())
    std::cout << "Error! Can't get the product "<< triggerEvent_.label() << std::endl;
  else {
    triggerEvent = *(triggerEventHandle.product());
  
    const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
    /////////////////////////////TriggerResults
    edm::InputTag theTriggerResultsLabel ("TriggerResults","","HLT");
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByLabel( theTriggerResultsLabel, triggerResults);
    char TrigName[50];
    sprintf(TrigName, "HLT_IsoTrack%s", Det.c_str());
    if (triggerResults.isValid()) {
      std::vector<std::string> modules;
      h_nHLT->Fill(triggerResults->size());
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);

      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      int hlt(-1), preL1(-1), preHLT(-1), prescale(-1);
      for (unsigned int i=0; i<triggerResults->size(); i++) {
	unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[i]);
	const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
	if (triggerNames_[i].find(TrigName)!=std::string::npos) {
	  const std::pair<int,int> prescales(hltConfig_.prescaleValues(iEvent,iSetup,triggerNames_[i]));
	  hlt = triggerResults->accept(i);
	  preL1  = prescales.first;
	  preHLT = prescales.second;
	  prescale = preL1*preHLT;
	  if (verbosity%10 > 0)
	    std::cout << triggerNames_[i] << " accept " << hlt << " preL1 " 
		      << preL1 << " preHLT " << preHLT << std::endl;
	  if (hlt>0) {
	    std::vector<math::XYZTLorentzVector> vec[3];
	    if (TrigList.find(RunNo) != TrigList.end() ) {
	      TrigList[RunNo] += 1;
	    } else {
	      TrigList.insert(std::pair<unsigned int, unsigned int>(RunNo,1));
	      TrigPreList.insert(std::pair<unsigned int, std::pair<int, int>>(RunNo,prescales));
	    }
	    //loop over all trigger filters in event (i.e. filters passed)
	    for (unsigned int ifilter=0; ifilter<triggerEvent.sizeFilters(); ++ifilter) {  
	      std::vector<int> Keys;
	      std::string label = triggerEvent.filterTag(ifilter).label();
	      //loop over keys to objects passing this filter
	      for (unsigned int imodule=0; imodule<moduleLabels.size(); imodule++) {
		if (label.find(moduleLabels[imodule]) != std::string::npos) {
		  if (verbosity%10 > 0) std::cout << "FILTERNAME " << label << std::endl;
		  for (unsigned int ifiltrKey=0; ifiltrKey<triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
		    Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
		    const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
		    math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
		    if(label.find("L2Filter") != std::string::npos) {
		      vec[1].push_back(v4);
		    } else if (label.find("L3Filter") != std::string::npos) {
		      vec[2].push_back(v4);
		    } else {
		      vec[0].push_back(v4);
		      h_L1ObjEnergy->Fill(TO.energy());
		    }
		    if (verbosity%10 > 0)
		      std::cout << "key " << ifiltrKey << " : pt " << TO.pt() << " eta " << TO.eta() << " phi " << TO.phi() << " mass " << TO.mass() << " Id " << TO.id() << std::endl;
		  }
		}
	      }
	    }
	    h_nL3Objs  -> Fill(vec[2].size());

	    //// Filling Pt, eta, phi of L1, L2 and L3 objects
	    for (int j=0; j<3; j++) {
	      for (unsigned int k=0; k<vec[j].size(); k++) {
		if (verbosity%10 > 0) std::cout << "vec[" << j << "][" << k << "] pt " << vec[j][k].pt() << " eta " << vec[j][k].eta() << " phi " << vec[j][k].phi() << std::endl;
		fillHist(j, vec[j][k]);
	      }
	    }

	    double deta, dphi, dr;
	    //// deta, dphi and dR for leading L1 object with L2 and L3 objects
	    for (int lvl=1; lvl<3; lvl++) {
	      for (unsigned int i=0; i<vec[lvl].size(); i++) {
		deta = dEta(vec[0][0],vec[lvl][i]);
		dphi = dPhi(vec[0][0],vec[lvl][i]);
		dr   = dR(vec[0][0],vec[lvl][i]);
		if (verbosity%10 > 0) std::cout << "lvl " <<lvl << " i " << i << " deta " << deta << " dphi " << dphi << " dR " << dr << std::endl;
		h_dEtaL1[lvl-1] -> Fill(deta);
		h_dPhiL1[lvl-1] -> Fill(dphi);
		h_dRL1[lvl-1]   -> Fill(std::sqrt(dr));
	      }
	    }

	    math::XYZTLorentzVector mindRvec;
	    double mindR;
	    for (unsigned int k=0; k<vec[2].size(); ++k) {
	      //// Find min of deta/dphi/dR for each of L3 objects with L2 objects
	      mindR=999.9;
	      if (verbosity%10 > 0) std::cout << "L3obj: pt " << vec[2][i].pt() << " eta " << vec[2][i].eta() << " phi " << vec[2][i].phi() << std::endl;
	      for (unsigned int j=0; j<vec[1].size(); j++) {
		dr   = dR(vec[2][k],vec[1][j]);
		if(dr<mindR) {
		  mindR=dr;
		  mindRvec=vec[1][j];
		}
	      }
	      fillDifferences(0, vec[2][k], mindRvec, (verbosity%10 >0));
	      if(mindR < 0.03) {
		fillDifferences(1, vec[2][k], mindRvec, (verbosity%10 >0));
		fillHist(6, mindRvec);
		fillHist(8, vec[2][k]);
	      } else {
		fillDifferences(2, vec[2][k], mindRvec, (verbosity%10 >0));
		fillHist(7, mindRvec);
		fillHist(9, vec[2][k]);
	      }
	      	      
	      ////// Minimum deltaR for each of L3 objs with Reco::tracks
	      mindR=999.9;
	      if (verbosity%10 > 0) std::cout << "vec[2][k].eta() " << vec[2][k].eta() << " vec[k][0].phi " << vec[2][k].phi() << std::endl;
	      reco::TrackCollection::const_iterator goodTk = trkCollection->end();
	      if (trkCollection.isValid()) {
		double mindP = 9999.9;
		for (trkItr=trkCollection->begin(); 
		     trkItr!=trkCollection->end(); trkItr++) {
		  math::XYZTLorentzVector v4(trkItr->px(), trkItr->py(), 
					     trkItr->pz(), trkItr->p());
		  double deltaR = dR(v4, vec[2][k]);
		  double dp     = std::abs(v4.r()/vec[2][k].r()-1.0);
		  if (deltaR<mindR) {
		    mindR    = deltaR;
		    mindP    = dp;
		    mindRvec = v4;
		    goodTk   = trkItr;
		  }
		  if ((verbosity/10)%10>1 && deltaR<1.0) {
		    std::cout << "track: pt " << v4.pt() << " eta " << v4.eta()
			      << " phi " << v4.phi() << " DR " << deltaR 
			      << std::endl;
		  }
		}
		if (mindR < 0.03 && mindP > 0.1) {
		  for (trkItr=trkCollection->begin(); 
		       trkItr!=trkCollection->end(); trkItr++) {
		    math::XYZTLorentzVector v4(trkItr->px(), trkItr->py(), 
					       trkItr->pz(), trkItr->p());
		    double deltaR = dR(v4, vec[2][k]);
		    double dp     = std::abs(v4.r()/vec[2][k].r()-1.0);
		    if (dp<mindP && deltaR<0.03) {
		      mindR    = deltaR;
		      mindP    = dp;
		      mindRvec = v4;
		      goodTk   = trkItr;
		    }
		  }
		}
		fillDifferences(3, vec[2][k], mindRvec, (verbosity%10 >0));
		fillHist(3, mindRvec);
		if(mindR < 0.03) {
		  fillDifferences(4, vec[2][k], mindRvec, (verbosity%10 >0));
		  fillHist(4, mindRvec);
		} else {
		  fillDifferences(5, vec[2][k], mindRvec, (verbosity%10 >0));
		  fillHist(5, mindRvec);
		}

		//Define the best vertex
		edm::Handle<reco::VertexCollection> recVtxs;
		iEvent.getByLabel("offlinePrimaryVertices",recVtxs);  
		// Get the beamspot
		edm::Handle<reco::BeamSpot> beamSpotH;
		iEvent.getByLabel("offlineBeamSpot", beamSpotH);
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

		std::vector<spr::propagatedTrackDirection> trkCaloDirections;
		spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections, ((verbosity/100)%10>2));
		std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  
		edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
		edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
		iEvent.getByLabel("ecalRecHit","EcalRecHitsEB",barrelRecHitsHandle);
		iEvent.getByLabel("ecalRecHit","EcalRecHitsEE",endcapRecHitsHandle);

		unsigned int nTracks=0, ngoodTk=0, nselTk=0;
		int          ieta=999;
		for (trkDetItr = trkCaloDirections.begin(); trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++){
		  bool l3Track  = (trkDetItr->trkItr == goodTk);
		  const reco::Track* pTrack = &(*(trkDetItr->trkItr));
		  math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), 
					     pTrack->pz(), pTrack->p());
		  bool selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>2));
		  double eMipDR=9999., e_inCone=0, conehmaxNearP=0, mindR=999.9;
		  if (trkDetItr->okHCAL) {
		    HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
		    ieta = detId.ieta();
		  }
		  for (unsigned k=0; k<vec[0].size(); ++k) {
		    double deltaR = dR(v4, vec[0][k]);
		    if (deltaR<mindR) mindR = deltaR;
		  }
		  if (selectTk && trkDetItr->okECAL && trkDetItr->okHCAL) {
		    ngoodTk++;
		    int nRH_eMipDR=0, nNearTRKs=0;
		    double e1 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
						 trkDetItr->pointHCAL, trkDetItr->pointECAL,
						 a_neutR1, trkDetItr->directionECAL, nRH_eMipDR);
		    double e2 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
						 trkDetItr->pointHCAL, trkDetItr->pointECAL,
						 a_neutR2, trkDetItr->directionECAL, nRH_eMipDR);
		    eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
					     trkDetItr->pointHCAL, trkDetItr->pointECAL,
					     a_mipR, trkDetItr->directionECAL, nRH_eMipDR);
		    conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR, nNearTRKs, ((verbosity/100)%10>2));
		    e_inCone = e2 - e1;
		    if (eMipDR<1.0) nselTk++;
		  }
		  if (l3Track) {
		    fillHist(10,v4);
		    if (selectTk) {
		      fillHist(11,v4);
		      fillCuts(0, eMipDR, conehmaxNearP, e_inCone, v4, ieta, (mindR>dr_L1));
		      if (conehmaxNearP < cutCharge) {
			fillHist(12,v4);
			if (eMipDR < cutMip) {
			  fillHist(13,v4);
			  if (e_inCone < cutNeutral) fillHist(14, v4);
			}
		      }
		    }
		  } else {
		    fillHist(15,v4);
		    if (selectTk) {
		      fillHist(16,v4);
		      fillCuts(1, eMipDR, conehmaxNearP, e_inCone, v4, ieta, (mindR>dr_L1));
		      if (conehmaxNearP < cutCharge) {
			fillHist(17,v4);
			if (eMipDR < cutMip) {
			  fillHist(18,v4);
			  if (e_inCone < cutNeutral) fillHist(19, v4);
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	  break;
	}
      }
      
      h_HLT      -> Fill(hlt);
      h_PreL1    -> Fill(preL1);
      h_PreHLT   -> Fill(preHLT);
      h_Pre      -> Fill(prescale);
      h_PreL1wt  -> Fill(preL1, mybxlumi);
      h_PreHLTwt -> Fill(preHLT, mybxlumi);

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

void IsoTrig::beginJob() {
  char hname[100], htit[100];
  std::string levels[20] = {"L1", "L2", "L3", 
			    "Reco", "RecoMatch", "RecoNoMatch", 
			    "L2Match", "L2NoMatch", "L3Match", "L3NoMatch", 
			    "HLTTrk", "HLTGoodTrk", "HLTIsoTrk", "HLTMip", "HLTSelect",
			    "nonHLTTrk", "nonHLTGoodTrk", "nonHLTIsoTrk", "nonHLTMip", "nonHLTSelect"};
  std::string pairs[6] = {"L2L3", "L2L3Match", "L2L3NoMatch", "L3Reco", "L3RecoMatch", "L3RecoNoMatch"};

  std::string cuts[2] = {"HLTMatched", "HLTNotMatched"};
  std::string cuts2[2] = {"All", "Away from L1"};
  h_nHLT        = fs->make<TH1I>("h_nHLT" , "size of rigger Names", 1000, 1, 1000);
  h_HLT         = fs->make<TH1I>("h_HLT"  , "HLT accept", 3, -1, 2);
  h_PreL1       = fs->make<TH1I>("h_PreL1", "L1 Prescale", 500, 0, 500);
  h_PreHLT      = fs->make<TH1I>("h_PreHLT", "HLT Prescale", 50, 0, 50);
  h_Pre         = fs->make<TH1I>("h_Pre", "Prescale", 3000, 0, 3000);
  h_nL3Objs     = fs->make<TH1I>("h_nL3Objs", "Number of L3 objects", 10, 0, 10);

  h_PreL1wt     = fs->make<TH1D>("h_PreL1wt", "Weighted L1 Prescale", 500, 0, 500);
  h_PreHLTwt    = fs->make<TH1D>("h_PreHLTwt", "Weighted HLT Prescale", 500, 0, 100);
  h_L1ObjEnergy = fs->make<TH1D>("h_L1ObjEnergy", "Energy of L1Object", 500, 0.0, 500.0);

  for(int ipair=0; ipair<6; ipair++) {
    sprintf(hname, "h_dEta%s", pairs[ipair].c_str());
    sprintf(htit, "dEta for %s", pairs[ipair].c_str());
    h_dEta[ipair]        = fs->make<TH1D>(hname, htit, 200, -10.0, 10.0);
    h_dEta[ipair]->GetXaxis()->SetTitle("d#eta");

    sprintf(hname, "h_dPhi%s", pairs[ipair].c_str());
    sprintf(htit, "dPhi for %s", pairs[ipair].c_str());
    h_dPhi[ipair]        = fs->make<TH1D>(hname, htit, 140, -7.0, 7.0);
    h_dPhi[ipair]->GetXaxis()->SetTitle("d#phi");

    sprintf(hname, "h_dPt%s", pairs[ipair].c_str());
    sprintf(htit, "dPt for %s objects", pairs[ipair].c_str());
    h_dPt[ipair]         = fs->make<TH1D>(hname, htit, 400, -200.0, 200.0);
    h_dPt[ipair]->GetXaxis()->SetTitle("dp_{T} (GeV)");

    sprintf(hname, "h_dP%s", pairs[ipair].c_str());
    sprintf(htit, "dP for %s objects", pairs[ipair].c_str());
    h_dP[ipair]         = fs->make<TH1D>(hname, htit, 400, -200.0, 200.0);
    h_dP[ipair]->GetXaxis()->SetTitle("dP (GeV)");

    sprintf(hname, "h_dinvPt%s", pairs[ipair].c_str());
    sprintf(htit, "dinvPt for %s objects", pairs[ipair].c_str());
    h_dinvPt[ipair]      = fs->make<TH1D>(hname, htit, 500, -0.4, 0.1);
    h_dinvPt[ipair]->GetXaxis()->SetTitle("d(1/p_{T})");

    sprintf(hname, "h_mindR%s", pairs[ipair].c_str());
    sprintf(htit, "mindR for %s objects", pairs[ipair].c_str());
    h_mindR[ipair]       = fs->make<TH1D>(hname, htit, 500, 0.0, 1.0);
    h_mindR[ipair]->GetXaxis()->SetTitle("dR");
  }

  for(int ilevel=0; ilevel<20; ilevel++) {
    sprintf(hname, "h_p%s", levels[ilevel].c_str());
    sprintf(htit, "p for %s objects", levels[ilevel].c_str());
    h_p[ilevel] = fs->make<TH1D>(hname, htit, 100, 0.0, 500.0);
    h_p[ilevel]->GetXaxis()->SetTitle("p (GeV)");

    sprintf(hname, "h_pt%s", levels[ilevel].c_str());
    sprintf(htit, "pt for %s objects", levels[ilevel].c_str());
    h_pt[ilevel] = fs->make<TH1D>(hname, htit, 100, 0.0, 500.0);
    h_pt[ilevel]->GetXaxis()->SetTitle("p_{T} (GeV)");

    sprintf(hname, "h_eta%s", levels[ilevel].c_str());
    sprintf(htit, "eta for %s objects", levels[ilevel].c_str());
    h_eta[ilevel] = fs->make<TH1D>(hname, htit, 100, -5.0, 5.0);
    h_eta[ilevel]->GetXaxis()->SetTitle("#eta");

    sprintf(hname, "h_phi%s", levels[ilevel].c_str());
    sprintf(htit, "phi for %s objects", levels[ilevel].c_str());
    h_phi[ilevel] = fs->make<TH1D>(hname, htit, 70, -3.5, 3.50);
    h_phi[ilevel]->GetXaxis()->SetTitle("#phi");
  }
  for(int lvl=0; lvl<2; lvl++) {
    sprintf(hname, "h_dEtaL1%s", levels[lvl+1].c_str());
    sprintf(htit, "dEta for L1 and %s objects", levels[lvl+1].c_str());
    h_dEtaL1[lvl] = fs->make<TH1D>(hname, htit, 400, -10.0, 10.0);

    sprintf(hname, "h_dPhiL1%s", levels[lvl+1].c_str());
    sprintf(htit, "dPhi for L1 and %s objects", levels[lvl+1].c_str());
    h_dPhiL1[lvl] = fs->make<TH1D>(hname, htit, 280, -7.0, 7.0);

    sprintf(hname, "h_dRL1%s", levels[lvl+1].c_str());
    sprintf(htit, "dR for L1 and %s objects", levels[lvl+1].c_str());
    h_dRL1[lvl] = fs->make<TH1D>(hname, htit, 100, 0.0, 10.0);
  }

  for(int icut=0; icut<2; icut++) {
    sprintf(hname, "h_eMip%s", cuts[icut].c_str());
    sprintf(htit, "eMip for %s tracks", cuts[icut].c_str());
    h_eMip[icut]     =fs->make<TH1D>(hname, htit, 200, 0.0, 10.0);
    h_eMip[icut]->GetXaxis()->SetTitle("E_{Mip} (GeV)");

    sprintf(hname, "h_eMaxNearP%s", cuts[icut].c_str());
    sprintf(htit, "eMaxNearP for %s tracks", cuts[icut].c_str());
    h_eMaxNearP[icut]=fs->make<TH1D>(hname, htit, 240, -2.0, 10.0);
    h_eMaxNearP[icut]->GetXaxis()->SetTitle("E_{MaxNearP} (GeV)");

    sprintf(hname, "h_eNeutIso%s", cuts[icut].c_str());
    sprintf(htit, "eNeutIso for %s ", cuts[icut].c_str());
    h_eNeutIso[icut] =fs->make<TH1D>(hname, htit, 200, 0.0, 10.0);
    h_eNeutIso[icut]->GetXaxis()->SetTitle("E_{NeutIso} (GeV)");

    for (int kcut=0; kcut<2; ++kcut) {
      sprintf(hname, "h_etaCalibTracks%s%d", cuts[icut].c_str(), kcut);
      sprintf(htit, "etaCalibTracks for %s (%s)", cuts[icut].c_str(), cuts2[kcut].c_str());
      h_etaCalibTracks[icut][kcut]=fs->make<TH1D>(hname, htit, 60, -30.0, 30.0);
      h_etaCalibTracks[icut][kcut]->GetXaxis()->SetTitle("i#eta");

      sprintf(hname, "h_etaMipTracks%s%d", cuts[icut].c_str(), kcut);
      sprintf(htit, "etaMipTracks for %s (%s)", cuts[icut].c_str(), cuts2[kcut].c_str());
      h_etaMipTracks[icut][kcut]=fs->make<TH1D>(hname, htit, 60, -30.0, 30.0);
      h_etaMipTracks[icut][kcut]->GetXaxis()->SetTitle("i#eta");
    }
  }
}
// ------------ method called once each job just after ending the event loop  ------------
void IsoTrig::endJob() {
  unsigned int preL1, preHLT;
  std::map<unsigned int, unsigned int>::iterator itr;
  std::map<unsigned int, const std::pair<int, int>>::iterator itrPre;
  std::cout << "RunNo vs HLT accepts for " << Det << std::endl;
  unsigned int n = maxRunNo - minRunNo +1;
  g_Pre = fs->make<TH1I>("h_PrevsRN", "PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreL1 = fs->make<TH1I>("h_PreL1vsRN", "L1 PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreHLT = fs->make<TH1I>("h_PreHLTvsRN", "HLT PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_Accepts = fs->make<TH1I>("h_HLTAcceptsvsRN", "HLT Accepts Vs Run Number", n, minRunNo, maxRunNo); 

  for (itr=TrigList.begin(), itrPre=TrigPreList.begin(); itr!=TrigList.end(); itr++, itrPre++) {
    preL1 = (itrPre->second).first;
    preHLT = (itrPre->second).second;
    std::cout << itr->first << " " << itr->second << " " <<  itrPre->first << " " << preL1 << " " << preHLT << std::endl;
    g_Accepts->Fill(itr->first, itr->second);
    g_PreL1->Fill(itr->first, preL1);
    g_PreHLT->Fill(itr->first, preHLT);
    g_Pre->Fill(itr->first, preL1*preHLT);
  }
}

// ------------ method called when starting to processes a run  ------------
void IsoTrig::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  std::cout  << "Run " << iRun.run() << " hltconfig.init " 
	     << hltConfig_.init(iRun,iSetup,"HLT",changed) << std::endl;;
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrig::endRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when starting to processes a luminosity block  ------------
void IsoTrig::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void IsoTrig::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void IsoTrig::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void IsoTrig::fillHist(int indx, math::XYZTLorentzVector& vec) {
  h_p[indx]->Fill(vec.r());
  h_pt[indx]->Fill(vec.pt());
  h_eta[indx]->Fill(vec.eta());
  h_phi[indx]->Fill(vec.phi());
}

void IsoTrig::fillDifferences(int indx, math::XYZTLorentzVector& vec1, 
			      math::XYZTLorentzVector& vec2, bool debug) {
  double dr     = dR(vec1,vec2);
  double deta   = dEta(vec1, vec2);
  double dphi   = dPhi(vec1, vec2);
  double dpt    = dPt(vec1, vec2);
  double dp     = dP(vec1, vec2);
  double dinvpt = dinvPt(vec1, vec2);
  h_dEta[indx]  ->Fill(deta);
  h_dPhi[indx]  ->Fill(dphi);
  h_dPt[indx]   ->Fill(dpt);
  h_dP[indx]    ->Fill(dp);
  h_dinvPt[indx]->Fill(dinvpt);
  h_mindR[indx] ->Fill(dr);
  if (debug) std::cout << "mindR for index " << indx << " is " << dr << " deta " << deta 
		       << " dphi " << dphi << " dpt " << dpt <<  " dinvpt " << dinvpt <<std::endl;
}

void IsoTrig::fillCuts(int indx, double eMipDR, double conehmaxNearP, double e_inCone, math::XYZTLorentzVector& vec, int ieta, bool cut) {
  h_eMip[indx]     ->Fill(eMipDR);
  h_eMaxNearP[indx]->Fill(conehmaxNearP);
  h_eNeutIso[indx] ->Fill(e_inCone);
  if (conehmaxNearP < cutCharge && eMipDR < cutMip && vec.r()<60 && vec.r()>40) {
    h_etaMipTracks[indx][0]->Fill((double)(ieta));
    if (cut) h_etaMipTracks[indx][1]->Fill((double)(ieta));
    if (e_inCone < cutNeutral) {
      h_etaCalibTracks[indx][0]->Fill((double)(ieta));
      if (cut) h_etaCalibTracks[indx][1]->Fill((double)(ieta));
    }
  }
}

double IsoTrig::dEta(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.eta()-vec2.eta());
}

double IsoTrig::dPhi(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {

  double phi1 = vec1.phi();
  if (phi1 < 0) phi1 += 2.0*M_PI;
  double phi2 = vec2.phi();
  if (phi2 < 0) phi2 += 2.0*M_PI;
  double dphi = phi1-phi2;
  if (dphi > M_PI)       dphi -= 2.*M_PI;
  else if (dphi < -M_PI) dphi += 2.*M_PI;
  return dphi;
}

double IsoTrig::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  double deta = dEta(vec1,vec2);
  double dphi = dPhi(vec1,vec2);
  return std::sqrt(deta*deta + dphi*dphi);
}

double IsoTrig::dPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.pt()-vec2.pt());
}

double IsoTrig::dP(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (std::abs(vec1.r()-vec2.r()));
}

double IsoTrig::dinvPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return ((1/vec1.pt())-(1/vec2.pt()));
}
//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrig);
