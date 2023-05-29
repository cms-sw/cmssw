// -*- C++ -*-
//
// Package:    CSCTFEfficiency
// Class:      CSCTFEfficiency
//
/**\class CSCTFEfficiency CSCTFEfficiency.cc jhugon/CSCTFEfficiency/src/CSCTFEfficiency.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Justin Hugon,Ivan's graduate student,jhugon@phys.ufl.edu
//         Created:  Thu Jun 10 10:40:10 EDT 2010
// $Id: CSCTFEfficiency.cc,v 1.2 2012/02/10 10:54:57 jhugon Exp $
//
//
//

#include "L1Trigger/CSCTrackFinder/test/analysis/CSCTFEfficiency.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include <algorithm>
#include "TStyle.h"
using namespace std;
int counterTFT;     // Added By Daniel 07/02
int counterRLimit;  // Added By Daniel 07/02
CSCTFEfficiency::CSCTFEfficiency(const edm::ParameterSet& iConfig) {
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed
  inputTag = iConfig.getUntrackedParameter<edm::InputTag>("inputTag");
  dataType = iConfig.getUntrackedParameter<int>("type_of_data");
  minPtSim = iConfig.getUntrackedParameter<double>("MinPtSim");
  maxPtSim = iConfig.getUntrackedParameter<double>("MaxPtSim");
  minEtaSim = iConfig.getUntrackedParameter<double>("MinEtaSim");
  maxEtaSim = iConfig.getUntrackedParameter<double>("MaxEtaSim");
  minPtTF = iConfig.getUntrackedParameter<double>("MinPtTF");
  minQualityTF = iConfig.getUntrackedParameter<double>("MinQualityTF");
  ghostLoseParam = iConfig.getUntrackedParameter<std::string>("GhostLoseParam");
  inputData = iConfig.getUntrackedParameter<bool>("InputData");
  minMatchRParam = iConfig.getUntrackedParameter<double>("MinMatchR");
  statsFilename = iConfig.getUntrackedParameter<std::string>("StatsFilename");
  saveHistImages = iConfig.getUntrackedParameter<bool>("SaveHistImages");
  singleMuSample = iConfig.getUntrackedParameter<bool>("SingleMuSample");
  noRefTracks = iConfig.getUntrackedParameter<bool>("NoRefTracks");
  cutOnModes = iConfig.getUntrackedParameter<std::vector<unsigned> >("CutOnModes");
  nEvents = 0;
  configuration = &iConfig;
}
CSCTFEfficiency::~CSCTFEfficiency() {
  //
}
// ------------ method called to for each event  ------------
void CSCTFEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace csctf_analysis;
  ///////////////////////////////
  ////// Event Set-Up ///////////
  ///////////////////////////////
  std::vector<double>* Rlist = new std::vector<double>;
  std::vector<RefTrack>* referenceTrack = new std::vector<RefTrack>;
  std::vector<TFTrack>* trackFinderTrack = new std::vector<TFTrack>;
  std::vector<RefTrack>::iterator refTrack;
  std::vector<TFTrack>::iterator tfTrack;

  if (!noRefTracks) {
    //Reference tracks
    edm::Handle<edm::SimTrackContainer> BaseSimTracks;
    iEvent.getByLabel("g4SimHits", BaseSimTracks);
    edm::SimTrackContainer::const_iterator BaseSimTrk;
    for (BaseSimTrk = BaseSimTracks->begin(); BaseSimTrk != BaseSimTracks->end(); BaseSimTrk++) {
      RefTrack refTrackTemp = RefTrack(*BaseSimTrk);
      double fabsEta = fabs(refTrackTemp.getEta());
      refTrackTemp.setHistList(refHistList);
      bool limits = refTrackTemp.getPt() < maxPtSim && refTrackTemp.getPt() > minPtSim && (fabsEta < maxEtaSim) &&
                    (fabsEta > minEtaSim);
      bool isMuon = fabs(refTrackTemp.getType()) == 13;
      if (isMuon && limits) {
        //std::cout<<"\nMinPtSim= "<<minPtSim<<"\n";
        //std::cout<<"Ref_Pt = "<<refTrackTemp.getPt()<<"\n";
        referenceTrack->push_back(refTrackTemp);
      }
    }
  }

  //dataType key: 0 - L1CSCTrack
  //              1 - L1MuRegionalCand
  //              2 - L1MuGMTExtendedCand
  if (dataType ==
      1)  //If you want the original production of sim csctf tracks without modes and ghost modes information/plots
  {
    edm::Handle<std::vector<L1MuRegionalCand> > trackFinderTracks;
    iEvent.getByLabel(inputTag, trackFinderTracks);
    std::vector<L1MuRegionalCand>::const_iterator BaseTFTrk;
    for (BaseTFTrk = trackFinderTracks->begin(); BaseTFTrk != trackFinderTracks->end(); BaseTFTrk++) {
      TFTrack tfTrackTemp = TFTrack(*BaseTFTrk);
      if (tfTrackTemp.getQuality() >= minQualityTF && tfTrackTemp.getPt() >= minPtTF) {
        tfTrackTemp.setHistList(tfHistList);
        trackFinderTrack->push_back(tfTrackTemp);
      }
    }
  } else if (dataType == 2) {
    //GMT Tracks
    edm::Handle<L1MuGMTReadoutCollection> gmtreadoutCollectionH;
    iEvent.getByLabel(inputTag, gmtreadoutCollectionH);
    L1MuGMTReadoutCollection const* GMTrc = gmtreadoutCollectionH.product();
    vector<L1MuGMTReadoutRecord> gmt_records = GMTrc->getRecords();
    vector<L1MuGMTReadoutRecord>::const_iterator gmt_records_iter;
    for (gmt_records_iter = gmt_records.begin(); gmt_records_iter != gmt_records.end(); gmt_records_iter++) {
      vector<L1MuGMTExtendedCand>::const_iterator ExtCand_iter;
      vector<L1MuGMTExtendedCand> extendedCands = gmt_records_iter->getGMTCands();
      for (ExtCand_iter = extendedCands.begin(); ExtCand_iter != extendedCands.end(); ExtCand_iter++) {
        L1MuGMTExtendedCand gmtec = *ExtCand_iter;
        TFTrack tfTrackTemp = TFTrack(gmtec);

        if (tfTrackTemp.getPt() >= minPtTF && fabs(tfTrackTemp.getEta()) <= maxEtaSim &&
            fabs(tfTrackTemp.getEta()) >= minEtaSim) {
          tfTrackTemp.setHistList(tfHistList);
          trackFinderTrack->push_back(tfTrackTemp);
        }
      }
    }
  } else  //If you want the new production of sim csctf tracks with modes and ghost modes information/plots
  {
    edm::Handle<L1CSCTrackCollection> trackFinderTracks;
    iEvent.getByLabel(inputTag, trackFinderTracks);
    L1CSCTrackCollection::const_iterator BaseTFTrk;

    for (BaseTFTrk = trackFinderTracks->begin(); BaseTFTrk != trackFinderTracks->end(); BaseTFTrk++) {
      TFTrack tfTrackTemp = TFTrack(*BaseTFTrk, iSetup);

      //Skips track if mode is not found in cutOnModes;
      if (cutOnModes.size() > 0) {
        std::vector<unsigned>::iterator found;
        found = std::find(cutOnModes.begin(), cutOnModes.end(), tfTrackTemp.getMode());
        if (found == cutOnModes.end())
          continue;
      }

      if (tfTrackTemp.getQuality() >= minQualityTF && tfTrackTemp.getPt() >= minPtTF) {
        tfTrackTemp.setHistList(tfHistList);
        trackFinderTrack->push_back(tfTrackTemp);
      }
    }
  }
  multHistList->FillMultiplicityHist(trackFinderTrack);
  if (trackFinderTrack->size() >= 2) {
    rHistogram->fillR(trackFinderTrack->at(0), trackFinderTrack->at(1));
  }

  //////////////////////////////////////
  //////// Track Matching //////////////
  //////////////////////////////////////
  // Loop over all Reference tracks for an Event
  unsigned int iRefTrack = 0;
  for (refTrack = referenceTrack->begin(); refTrack != referenceTrack->end(); refTrack++) {
    bool tftracksExist = trackFinderTrack->size() > 0;
    if (tftracksExist) {
      for (tfTrack = trackFinderTrack->begin(); tfTrack != trackFinderTrack->end(); tfTrack++) {
        Rlist->push_back(refTrack->distanceTo(&(*tfTrack)));
      }
      unsigned int iSmallR = minIndex(Rlist);
      double smallR = Rlist->at(iSmallR);
      bool bestMatch;
      bool oldMatch;
      if (trackFinderTrack->at(iSmallR).getMatched()) {
        bestMatch = smallR < trackFinderTrack->at(iSmallR).getR();
        oldMatch = true;
      } else if (smallR < minMatchRParam) {
        bestMatch = true;
        oldMatch = false;
      } else {
        bestMatch = false;
        oldMatch = false;
      }

      if (bestMatch) {
        if (oldMatch) {
          int oldRefMatchi = trackFinderTrack->at(iSmallR).getMatchedIndex();
          referenceTrack->at(oldRefMatchi).unMatch();
        }
        int tfQ = trackFinderTrack->at(iSmallR).getQuality();
        double tfPt = trackFinderTrack->at(iSmallR).getPt();
        refTrack->matchedTo(iSmallR, smallR, tfQ, tfPt);
        refTrack->setQuality(tfQ);
        refTrack->setTFPt(tfPt);
        trackFinderTrack->at(iSmallR).matchedTo(iRefTrack, smallR);
      }
    }
    Rlist->clear();
  }

  //Fill Histograms
  int iHighest = 0;
  int i = 0;
  double highestPt = -100.0;
  for (refTrack = referenceTrack->begin(); refTrack != referenceTrack->end(); refTrack++) {
    refTrack->fillHist();
    if (refTrack->getPt() > highestPt) {
      iHighest = i;
      highestPt = refTrack->getPt();
    }
    if (refTrack->getMatched()) {
      refTrack->setMatch(trackFinderTrack->at(refTrack->getMatchedIndex()));  //this line links the two tracks
      refTrack->fillSimvTFHist(*refTrack, refTrack->getMatchedTrack());
      refTrack->fillMatchHist();
      // resolution
      int TFTIndex = refTrack->getMatchedIndex();
      TFTrack tfTrk = trackFinderTrack->at(TFTIndex);
      resHistList->FillResolutionHist(*refTrack, tfTrk);
    }
    i++;
  }
  if (!referenceTrack->empty()) {
    referenceTrack->at(iHighest).fillRateHist();
  }

  iHighest = 0;
  i = 0;
  highestPt = -100.0;
  for (tfTrack = trackFinderTrack->begin(); tfTrack != trackFinderTrack->end(); tfTrack++) {
    tfTrack->fillHist();
    if (tfTrack->getPt() > highestPt) {
      iHighest = i;
      highestPt = tfTrack->getPt();
    }
    if (tfTrack->getMatched()) {
      tfTrack->fillMatchHist();
    }
  }
  if (!trackFinderTrack->empty()) {
    trackFinderTrack->at(iHighest).fillRateHist();
  }

  ////////////////////////////////
  ////// Ghost Finding ///////////
  ////////////////////////////////
  if (singleMuSample) {
    if (trackFinderTrack->size() > 1) {
      std::vector<double>* Qlist = new std::vector<double>;
      for (tfTrack = trackFinderTrack->begin(); tfTrack != trackFinderTrack->end(); tfTrack++) {
        if (referenceTrack->size() > 0) {
          Rlist->push_back(referenceTrack->begin()->distanceTo(&(*tfTrack)));
        }
        Qlist->push_back(tfTrack->getQuality());
      }
      unsigned int iSmallR = minIndex(Rlist);
      unsigned int iSmallQ = minIndex(Qlist);
      if (referenceTrack->size() > 0) {
        if (ghostLoseParam == "Q") {
          trackFinderTrack->at(iSmallQ).fillGhostHist();
        } else if (ghostLoseParam == "R") {
          trackFinderTrack->at(iSmallR).fillGhostHist();
        } else {
          std::cout << "Warning: ghostLoseParam Not set to R or Q!" << std::endl;
        }
        referenceTrack->begin()->fillGhostHist();
      } else {
        std::cout << "Warning: Multiple TFTracks found, RefTrack NOT found!" << std::endl;
        trackFinderTrack->at(iSmallQ).fillGhostHist();
      }
      delete Qlist;
    }
  } else {
    unsigned int iTFTrack = 0;
    for (tfTrack = trackFinderTrack->begin(); tfTrack != trackFinderTrack->end(); tfTrack++) {
      bool reftracksExist = referenceTrack->size() > 0;
      if (reftracksExist) {
        for (refTrack = referenceTrack->begin(); refTrack != referenceTrack->end(); refTrack++) {
          Rlist->push_back(refTrack->distanceTo(&(*tfTrack)));
        }
        unsigned int iSmallR = minIndex(Rlist);
        double smallR = Rlist->at(iSmallR);
        RefTrack* matchedRefTrack = &referenceTrack->at(iSmallR);
        matchedRefTrack->ghostMatchedTo(*tfTrack, iTFTrack, smallR);
        Rlist->clear();
        iTFTrack++;
      }
    }
    for (refTrack = referenceTrack->begin(); refTrack != referenceTrack->end(); refTrack++) {
      if (refTrack->getGhost()) {
        refTrack->loseBestGhostCand(ghostLoseParam);
        std::vector<unsigned int>* ghosts;
        ghosts = refTrack->ghostMatchedIndecies();
        std::vector<unsigned int>::const_iterator iGhost;
        for (iGhost = ghosts->begin(); iGhost != ghosts->end(); iGhost++) {
          TFTrack* tempTFTrack = &trackFinderTrack->at(*iGhost);
          int tfQ = tempTFTrack->getQuality();
          refTrack->setQuality(tfQ);
          refTrack->fillGhostHist();
          tempTFTrack->fillGhostHist();
        }
      }
    }
  }
  delete Rlist;
  delete referenceTrack;
  delete trackFinderTrack;
  nEvents++;
}
// ------------ method called once each job just before starting event loop  ------------
void CSCTFEfficiency::beginJob() {
  using namespace csctf_analysis;

  gStyle->SetOptStat(0);
  tfHistList = new TrackHistogramList("TrackFinder", configuration);
  refHistList = new TrackHistogramList("Reference", configuration);
  effhistlist = new EffHistogramList("Efficiency", configuration);
  resHistList = new ResolutionHistogramList("Resolution", configuration);
  multHistList = new MultiplicityHistogramList();
  statFile = new StatisticsFile(statsFilename);
  rHistogram = new RHistogram("RHistograms");
}
// ------------ method called once each job just after ending the event loop  ------------
void CSCTFEfficiency::endJob() {
  effhistlist->ComputeEff(refHistList);
  statFile->WriteStatistics(*tfHistList, *refHistList);
  if (saveHistImages) {
    effhistlist->Print();
    resHistList->Print();
  }
  delete tfHistList;
  delete refHistList;
  delete effhistlist;
  delete resHistList;
  delete statFile;
}
namespace csctf_analysis {
  unsigned int minIndex(const std::vector<int>* list) {
    unsigned int minI = 0;
    for (unsigned int i = 0; i < list->size(); i++) {
      if (list[i] < list[minI]) {
        minI = i;
      }
    }
    return minI;
  }
  unsigned int minIndex(const std::vector<double>* list) {
    unsigned int minI = 0;
    for (unsigned int i = 0; i < list->size(); i++) {
      if (list->at(i) < list->at(minI)) {
        minI = i;
      }
    }
    return minI;
  }

}  // namespace csctf_analysis
//define this as a plug-in
DEFINE_FWK_MODULE(CSCTFEfficiency);
