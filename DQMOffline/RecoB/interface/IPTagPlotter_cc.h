#include <cstddef>
#include <string>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQMOffline/RecoB/interface/IPTagPlotter.h"

template <class Container, class Base>
IPTagPlotter<Container, Base>::IPTagPlotter(const std::string & tagName,
                              const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, 
                              unsigned int mc, bool wf, DQMStore::IBooker & ibook_):
  BaseTagInfoPlotter(tagName, etaPtBin),
  nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
  startEffPur_(pSet.getParameter<double>("startEffPur")),
  endEffPur_(pSet.getParameter<double>("endEffPur")),
  mcPlots_(mc), willFinalize_(wf),
  makeQualityPlots_(pSet.getParameter<bool>("QualityPlots")),
  lowerIPSBound(pSet.getParameter<double>("LowerIPSBound")),
  upperIPSBound(pSet.getParameter<double>("UpperIPSBound")),
  lowerIPBound(pSet.getParameter<double>("LowerIPBound")),
  upperIPBound(pSet.getParameter<double>("UpperIPBound")),
  lowerIPEBound(pSet.getParameter<double>("LowerIPEBound")),
  upperIPEBound(pSet.getParameter<double>("UpperIPEBound")),
  nBinsIPS(pSet.getParameter<int>("NBinsIPS")),
  nBinsIP(pSet.getParameter<int>("NBinsIP")),
  nBinsIPE(pSet.getParameter<int>("NBinsIPE")),
  minDecayLength(pSet.getParameter<double>("MinDecayLength")),
  maxDecayLength(pSet.getParameter<double>("MaxDecayLength")),
  minJetDistance(pSet.getParameter<double>("MinJetDistance")),
  maxJetDistance(pSet.getParameter<double>("MaxJetDistance"))
{
  const std::string trackIPDir(theExtensionString.substr(1));

  if (willFinalize_) return;

  // Number of tracks
  // 3D
  trkNbr3D = std::make_unique<TrackIPHistograms<int>>
    ("selTrksNbr_3D" + theExtensionString, "Number of selected tracks for 3D IPS", 31, -0.5, 30.5,
     false, true, true, "b", trackIPDir ,mc, makeQualityPlots_, ibook_);

  // 2D
  trkNbr2D = std::make_unique<TrackIPHistograms<int>>
    ("selTrksNbr_2D" + theExtensionString, "Number of selected tracks for 2D IPS", 31, -0.5, 30.5,
     false, true, true, "b", trackIPDir ,mc, makeQualityPlots_, ibook_);

  // IP significance
  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosSig3D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("ips" + std::to_string(i) + "_3D" + theExtensionString, "3D IP significance " + std::to_string(i) + ".trk",
            nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }
  tkcntHistosSig3D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("ips_3D" + theExtensionString, "3D IP significance",
        nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  //2D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosSig2D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("ips" + std::to_string(i) + "_2D" + theExtensionString, "2D IP significance " + std::to_string(i) + ".trk",
            nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosSig2D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("ips_2D" + theExtensionString, "2D IP significance",
        nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // IP value
  //3D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosVal3D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("ip" + std::to_string(i) + "_3D" + theExtensionString, "3D IP value " + std::to_string(i) + ".trk",
            nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosVal3D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("ip_3D" + theExtensionString, "3D IP value",
        nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  //2D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosVal2D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("ip" + std::to_string(i) + "_2D" + theExtensionString, "2D IP value " + std::to_string(i) + ".trk",
            nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosVal2D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("ip_2D" + theExtensionString, "2D IP value",
        nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));


  // IP error
  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosErr3D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("ipe" + std::to_string(i) + "_3D" + theExtensionString, "3D IP error " + std::to_string(i) + ".trk",
            nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosErr3D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("ipe_3D" + theExtensionString, "3D IP error",
        nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  //2D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosErr2D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("ipe" + std::to_string(i) + "_2D" + theExtensionString, "2D IP error " + std::to_string(i) + ".trk",
            nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosErr2D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("ipe_2D" + theExtensionString, "2D IP error",
        nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // decay length
  //2D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosDecayLengthVal2D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("decLen" + std::to_string(i) + "_2D" + theExtensionString, "2D Decay Length " + std::to_string(i) + ".trk",
            50, 0.0, 5.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosDecayLengthVal2D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("decLen_2D" + theExtensionString, "Decay Length 2D",
        50, 0.0, 5.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosDecayLengthVal3D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("decLen" + std::to_string(i) + "_3D" + theExtensionString, "3D Decay Length " + std::to_string(i) + ".trk",
            50, 0.0, 5.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosDecayLengthVal3D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("decLen_3D" + theExtensionString, "3D Decay Length",
        50, 0.0, 5.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // jet distance
  //2D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosJetDistVal2D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("jetDist" + std::to_string(i) + "_2D" + theExtensionString, "JetDistance 2D " + std::to_string(i) + ".trk",
            50, -0.1, 0.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosJetDistVal2D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("jetDist_2D" + theExtensionString, "JetDistance 2D",
        50, -0.1, 0.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosJetDistVal3D.push_back(std::make_unique<TrackIPHistograms<double>>
           ("jetDist" + std::to_string(i) + "_3D" + theExtensionString, "JetDistance 3D " + std::to_string(i) + ".trk",
            50, -0.1, 0.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosJetDistVal3D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("jetDist_3D" + theExtensionString, "JetDistance 3D",
        50, -0.1, 0.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // track chi-squared
  // 2D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkNChiSqr2D.push_back(std::make_unique<TrackIPHistograms<double>>
         ("tkNChiSqr" + std::to_string(i) + "_2D" + theExtensionString, "Normalized Chi Squared 2D " + std::to_string(i) + ".trk",
          50, -0.1, 10.0, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkNChiSqr2D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("tkNChiSqr_2D" + theExtensionString, "Normalized Chi Squared 2D",
        50, -0.1, 10.0, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkNChiSqr3D.push_back(std::make_unique<TrackIPHistograms<double>>
         ("tkNChiSqr" + std::to_string(i) + "_3D" + theExtensionString, "Normalized Chi Squared 3D " + std::to_string(i) + ".trk",
          50, -0.1, 10.0, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkNChiSqr3D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("tkNChiSqr_3D" + theExtensionString, "Normalized Chi Squared 3D",
        50, -0.1, 10.0, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  // track pT
  // 2D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkPt2D.push_back(std::make_unique<TrackIPHistograms<double>>
         ("tkPt" + std::to_string(i) + "_2D" + theExtensionString, "Track Pt 2D " + std::to_string(i) + ".trk",
          50, -0.1, 50.1, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkPt2D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("tkPt_2D" + theExtensionString, "Track Pt 2D",
        50, -0.1, 50.1, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkPt3D.push_back(std::make_unique<TrackIPHistograms<double>>
         ("tkPt" + std::to_string(i) + "_3D" + theExtensionString, "Track Pt 3D " + std::to_string(i) + ".trk",
          50, -0.1, 50.1, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkPt3D.push_back(std::make_unique<TrackIPHistograms<double>>
       ("tkPt_3D" + theExtensionString, "Track Pt 3D",
        50, -0.1, 50.1, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  // track nHits
  // 2D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkNHits2D.push_back(std::make_unique<TrackIPHistograms<int>>
         ("tkNHits" + std::to_string(i) + "_2D" + theExtensionString, "Track NHits 2D " + std::to_string(i) + ".trk",
          31, -0.5, 30.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkNHits2D.push_back(std::make_unique<TrackIPHistograms<int>>
       ("tkNHits_2D" + theExtensionString, "Track NHits 2D",
        31, -0.5, 30.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkNHits3D.push_back(std::make_unique<TrackIPHistograms<int>>
         ("tkNHits" + std::to_string(i) + "_3D" + theExtensionString, "Track NHits 3D " + std::to_string(i) + ".trk",
          31, -0.5, 30.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkNHits3D.push_back(std::make_unique<TrackIPHistograms<int>>
       ("tkNHits_3D" + theExtensionString, "Track NHits 3D",
        31, -0.5, 30.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  //Pixel hits
  // 2D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkNPixelHits2D.push_back(std::make_unique<TrackIPHistograms<int>>
         ("tkNPixelHits" + std::to_string(i) + "_2D" + theExtensionString, "Track NPixelHits 2D " + std::to_string(i) + ".trk",
          11, -0.5, 10.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkNPixelHits2D.push_back(std::make_unique<TrackIPHistograms<int>>
       ("tkNPixelHits_2D" + theExtensionString, "Track NPixelHits 2D",
        11, -0.5, 10.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
    tkcntHistosTkNPixelHits3D.push_back(std::make_unique<TrackIPHistograms<int>>
         ("tkNPixelHits" + std::to_string(i) + "_3D" + theExtensionString, "Track NPixelHits 3D " + std::to_string(i) + ".trk",
          11, -0.5, 10.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosTkNPixelHits3D.push_back(std::make_unique<TrackIPHistograms<int>>
       ("tkNPixelHits_3D" + theExtensionString, "Track NPixelHits 3D",
        11, -0.5, 10.5, false, true, true, "b",  trackIPDir, mc, makeQualityPlots_, ibook_));

  // probability
  // 3D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosProb3D.push_back(std::make_unique<TrackIPHistograms<float>>
           ("prob" + std::to_string(i) + "_3D" + theExtensionString, "3D IP probability " + std::to_string(i) + ".trk",
            52, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosProb3D.push_back(std::make_unique<TrackIPHistograms<float>>
       ("prob_3D" + theExtensionString, "3D IP probability",
    50, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // 2D
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosProb2D.push_back(std::make_unique<TrackIPHistograms<float>>
           ("prob" + std::to_string(i) + "_2D" + theExtensionString, "2D IP probability " + std::to_string(i) + ".trk",
            52, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));
  }

  tkcntHistosProb2D.push_back(std::make_unique<TrackIPHistograms<float>>
       ("prob_2D" + theExtensionString, "2D IP probability",
        52, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_));

  // probability for tracks with IP value < 0 or IP value >> 0
  tkcntHistosTkProbIPneg2D = std::make_unique<TrackIPHistograms<float>>
       ("probIPneg_2D" + theExtensionString, "2D negative IP probability",
        52, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_);
  tkcntHistosTkProbIPpos2D = std::make_unique<TrackIPHistograms<float>>
       ("probIPpos_2D" + theExtensionString, "2D positive IP probability",
        52, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_);
  tkcntHistosTkProbIPneg3D = std::make_unique<TrackIPHistograms<float>>
       ("probIPneg_3D" + theExtensionString, "3D negative IP probability",
        52, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_);
  tkcntHistosTkProbIPpos3D = std::make_unique<TrackIPHistograms<float>>
       ("probIPpos_3D" + theExtensionString, "3D positive IP probability",
        52, -1.04, 1.04, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_);

  // ghost Tracks and others
  ghostTrackDistanceValuHisto = std::make_unique<TrackIPHistograms<double>>
       ("ghostTrackDist" + theExtensionString, "GhostTrackDistance",
        50, 0.0, 0.1, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_);
  ghostTrackDistanceSignHisto = std::make_unique<TrackIPHistograms<double>>
       ("ghostTrackDistSign" + theExtensionString, "GhostTrackDistance significance",
        50, -5.0, 15.0, false, true, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_);
  ghostTrackWeightHisto = std::make_unique<TrackIPHistograms<double>>
       ("ghostTrackWeight" + theExtensionString, "GhostTrack fit participation weight",
        50, 0.0, 1.0, false, false, true, "b", trackIPDir, mc, makeQualityPlots_, ibook_);

  trackQualHisto = std::make_unique<FlavourHistograms<int>>
       ("trackQual" + theExtensionString, "Track Quality of Tracks Associated to Jets",
        4, -1.5, 2.5, false, true, true, "b",  trackIPDir, mc, ibook_);

  selectedTrackQualHisto = std::make_unique<FlavourHistograms<int>>
       ("selectedTrackQual" + theExtensionString, "Track Quality of Selected Tracks Associated to Jets",
        4, -1.5, 2.5, false, true, true, "b",  trackIPDir, mc, ibook_);

  trackMultVsJetPtHisto = std::make_unique<FlavourHistograms2D<double, int>>
       ("trackMultVsJetPt" + theExtensionString, "Track Multiplicity vs Jet Pt for Tracks Associated to Jets",
        50, 0.0, 250.0, 21, -0.5, 30.5, false,  trackIPDir, mc, true, ibook_);

  selectedTrackMultVsJetPtHisto = std::make_unique<FlavourHistograms2D<double, int>>
       ("selectedTrackMultVsJetPt" + theExtensionString, "Track Multiplicity vs Jet Pt for Selected Tracks Associated to Jets",
        50, 0.0, 250.0, 21, -0.5, 20.5, false, trackIPDir, mc, true, ibook_);

}

template <class Container, class Base> 
IPTagPlotter<Container, Base>::~IPTagPlotter() { }

template <class Container, class Base> 
void IPTagPlotter<Container, Base>::analyzeTag (const reco::BaseTagInfo * baseTagInfo, double jec, int jetFlavour, float w/*=1*/)
{
  //  const reco::TrackIPTagInfo * tagInfo = 
  //    dynamic_cast<const reco::TrackIPTagInfo *>(baseTagInfo);
  const reco::IPTagInfo<Container, Base> * tagInfo = 
    dynamic_cast<const reco::IPTagInfo<Container, Base> *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type TrackIPTagInfo. " << std::endl;
  }

  const GlobalPoint pv(tagInfo->primaryVertex()->position().x(),
                       tagInfo->primaryVertex()->position().y(),
                       tagInfo->primaryVertex()->position().z());

  const std::vector<reco::btag::TrackIPData>& ip = tagInfo->impactParameterData();

  std::vector<float> prob2d, prob3d;
  if (tagInfo->hasProbabilities()) {
    prob2d = tagInfo->probabilities(0);    
    prob3d = tagInfo->probabilities(1);    
  }

  std::vector<std::size_t> sortedIndices = tagInfo->sortedIndexes(reco::btag::IP2DSig);
  std::vector<std::size_t> selectedIndices;
  Container sortedTracks = tagInfo->sortedTracks(sortedIndices);
  Container selectedTracks;
  for (unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if (decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance ) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }

  trkNbr2D->fill(jetFlavour, selectedIndices.size(),w);

  for (unsigned int n = 0; n != selectedIndices.size(); ++n) {
    const reco::Track * track =  reco::btag::toTrack(selectedTracks[n]);
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosSig2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.significance(), true, w);
    tkcntHistosVal2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.value(), true, w);
    tkcntHistosErr2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.error(), true, w);
    const double& decayLen = (ip[selectedIndices[n]].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal2D[4]->fill(jetFlavour, trackQual, decayLen, true, w);
    tkcntHistosJetDistVal2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true, w);
    tkcntHistosTkNChiSqr2D[4]->fill(jetFlavour, trackQual, track->normalizedChi2(), true, w);
    tkcntHistosTkPt2D[4]->fill(jetFlavour, trackQual, track->pt(), true, w);
    tkcntHistosTkNHits2D[4]->fill(jetFlavour, trackQual, track->found(), true, w);
    tkcntHistosTkNPixelHits2D[4]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true, w);
    if (n >= 4) continue;
    tkcntHistosSig2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.significance(), true, w);
    tkcntHistosVal2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.value(), true, w);
    tkcntHistosErr2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.error(), true, w);
    tkcntHistosDecayLengthVal2D[n]->fill(jetFlavour, trackQual, decayLen, true, w);
    tkcntHistosJetDistVal2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true, w);
    tkcntHistosTkNChiSqr2D[n]->fill(jetFlavour, trackQual, track->normalizedChi2(), true, w);
    tkcntHistosTkPt2D[n]->fill(jetFlavour, trackQual, track->pt(), true, w);
    tkcntHistosTkNHits2D[n]->fill(jetFlavour, trackQual, track->found(), true, w);
    tkcntHistosTkNPixelHits2D[n]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true, w);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::btag::Prob2D);
  selectedIndices.clear();
  sortedTracks = tagInfo->sortedTracks(sortedIndices);
  selectedTracks.clear();
  for (unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if (decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance ) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }
  for (unsigned int n = 0; n != selectedIndices.size(); ++n) {
    const reco::Track * track = reco::btag::toTrack(selectedTracks[n]);
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosProb2D[4]->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true, w);
    if (ip[selectedIndices[n]].ip2d.value() < 0) tkcntHistosTkProbIPneg2D->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true, w);
    else tkcntHistosTkProbIPpos2D->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true, w);
    if (n >= 4) continue;
    tkcntHistosProb2D[n]->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true, w);
  }
  for (unsigned int n = selectedIndices.size(); n < 4; ++n){
    const reco::TrackBase::TrackQuality trackQual = reco::TrackBase::undefQuality;
    tkcntHistosSig2D[n]->fill(jetFlavour, trackQual, lowerIPSBound-1.0, false, w);
    tkcntHistosVal2D[n]->fill(jetFlavour, trackQual, lowerIPBound-1.0, false, w);
    tkcntHistosErr2D[n]->fill(jetFlavour, trackQual, lowerIPEBound-1.0, false, w);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::btag::IP3DSig);
  selectedIndices.clear();
  sortedTracks = tagInfo->sortedTracks(sortedIndices);
  selectedTracks.clear();
  for (unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if (decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance ) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }

  trkNbr3D->fill(jetFlavour, selectedIndices.size(), w);
  int nSelectedTracks = selectedIndices.size();

  for (unsigned int n = 0; n != selectedIndices.size(); ++n) {
    const reco::Track * track = reco::btag::toTrack(selectedTracks[n]);
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosSig3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.significance(), true, w);
    tkcntHistosVal3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.value(), true, w);
    tkcntHistosErr3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.error(), true, w);
    const double& decayLen = (ip[selectedIndices[n]].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal3D[4]->fill(jetFlavour, trackQual, decayLen, true, w);
    tkcntHistosJetDistVal3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true, w);
    tkcntHistosTkNChiSqr3D[4]->fill(jetFlavour, trackQual, track->normalizedChi2(), true, w);
    tkcntHistosTkPt3D[4]->fill(jetFlavour, trackQual, track->pt(), true,w);
    tkcntHistosTkNHits3D[4]->fill(jetFlavour, trackQual, track->found(), true,w);
    tkcntHistosTkNPixelHits3D[4]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true,w);
    //ghostTrack infos  
    ghostTrackDistanceValuHisto->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToGhostTrack.value(), true, w);
    ghostTrackDistanceSignHisto->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToGhostTrack.significance(), true, w);
    ghostTrackWeightHisto->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ghostTrackWeight, true, w);
    selectedTrackQualHisto->fill(jetFlavour, trackQual, w);
    if (n >= 4) continue;
    tkcntHistosSig3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.significance(), true, w);
    tkcntHistosVal3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.value(), true, w);
    tkcntHistosErr3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.error(), true, w);
    tkcntHistosDecayLengthVal3D[n]->fill(jetFlavour, trackQual, decayLen, true, w);
    tkcntHistosJetDistVal3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true, w);
    tkcntHistosTkNChiSqr3D[n]->fill(jetFlavour, trackQual, track->normalizedChi2(), true, w);
    tkcntHistosTkPt3D[n]->fill(jetFlavour, trackQual, track->pt(), true, w);
    tkcntHistosTkNHits3D[n]->fill(jetFlavour, trackQual, track->found(), true, w);
    tkcntHistosTkNPixelHits3D[n]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true, w);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::btag::Prob3D);
  selectedIndices.clear();
  sortedTracks = tagInfo->sortedTracks(sortedIndices);
  selectedTracks.clear();
  for (unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if (decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }
  for (unsigned int n = 0; n != selectedIndices.size(); ++n) {
    const reco::Track * track = reco::btag::toTrack(selectedTracks[n]);
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosProb3D[4]->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true, w);
    if (ip[selectedIndices[n]].ip3d.value() < 0) tkcntHistosTkProbIPneg3D->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true, w);
    else tkcntHistosTkProbIPpos3D->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true, w);
    if (n >= 4) continue;
    tkcntHistosProb3D[n]->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true, w);
  }
  for (unsigned int n = selectedIndices.size(); n < 4; ++n){
    const reco::TrackBase::TrackQuality trackQual = reco::TrackBase::undefQuality;
    tkcntHistosSig3D[n]->fill(jetFlavour, trackQual, lowerIPSBound-1.0, false, w);
    tkcntHistosVal3D[n]->fill(jetFlavour, trackQual, lowerIPBound-1.0, false, w);
    tkcntHistosErr3D[n]->fill(jetFlavour, trackQual, lowerIPEBound-1.0, false, w);
  }
  for (unsigned int n = 0; n != sortedTracks.size(); ++n) {
    trackQualHisto->fill(jetFlavour, highestTrackQual(reco::btag::toTrack(sortedTracks[n])), w);
  }

  //still need to implement weights in FlavourHistograms2D
  trackMultVsJetPtHisto->fill(jetFlavour, tagInfo->jet()->pt() * jec, sortedTracks.size());
  selectedTrackMultVsJetPtHisto->fill(jetFlavour, tagInfo->jet()->pt() * jec, nSelectedTracks); //tagInfo->selectedTracks().size());
}

template <class Container, class Base> 
void IPTagPlotter<Container, Base>::finalize(DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_)
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  const std::string trackIPDir(theExtensionString.substr(1));

  tkcntHistosSig3D.clear();
  tkcntHistosSig2D.clear();
  effPurFromHistos.clear();
  
  for (unsigned int i = 2; i <= 3; i++) {
    tkcntHistosSig3D.push_back(
            std::make_unique<TrackIPHistograms<double>>
                        ("ips" + std::to_string(i) + "_3D" + theExtensionString, "3D IP significance " + std::to_string(i) + ".trk",
                        nBinsIPS, lowerIPSBound, upperIPSBound, "b", trackIPDir, mcPlots_, makeQualityPlots_, igetter_));
    effPurFromHistos.push_back(
            std::make_unique<EffPurFromHistos>(*tkcntHistosSig3D.back(), trackIPDir, mcPlots_, ibook_,
                            nBinEffPur_, startEffPur_, endEffPur_));
  }

  for (unsigned int i = 2; i <= 3; i++) {
    tkcntHistosSig2D.push_back(
            std::make_unique<TrackIPHistograms<double>>
                    ("ips" + std::to_string(i) + "_2D" + theExtensionString, "2D IP significance " + std::to_string(i) + ".trk",
                    nBinsIPS, lowerIPSBound, upperIPSBound, "b", trackIPDir, mcPlots_, makeQualityPlots_, igetter_));
    effPurFromHistos.push_back(
            std::make_unique<EffPurFromHistos>(*tkcntHistosSig2D.back(), trackIPDir, mcPlots_, ibook_,
                            nBinEffPur_, startEffPur_, endEffPur_));
  }

  for (int n = 0; n != 4; ++n) effPurFromHistos[n]->compute(ibook_);
}


template <class Container, class Base> 
void IPTagPlotter<Container, Base>::psPlot(const std::string & name)
{
  const std::string cName("TrackIPPlots"+ theExtensionString);
  RecoBTag::setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();
  if (willFinalize_) {
    for (int n = 0; n != 2; ++n) {
      canvas.Print((name + cName + ".ps").c_str());
      canvas.Clear();
      canvas.Divide(2,3);
      canvas.cd(1);
      effPurFromHistos[0+n]->discriminatorNoCutEffic().plot();
      canvas.cd(2);
      effPurFromHistos[0+n]->discriminatorCutEfficScan().plot();
      canvas.cd(3);
      effPurFromHistos[0+n]->plot();
      canvas.cd(4);
      effPurFromHistos[1+n]->discriminatorNoCutEffic().plot();
      canvas.cd(5);
      effPurFromHistos[1+n]->discriminatorCutEfficScan().plot();
      canvas.cd(6);
      effPurFromHistos[1+n]->plot();
    }
    return;
  }

  canvas.Print((name + cName + ".ps[").c_str());
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  trkNbr3D->plot();
  canvas.cd(2);
  tkcntHistosSig3D[4]->plot();
  for (int n = 0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosSig3D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  trkNbr3D->plot();
  canvas.cd(2);
  tkcntHistosProb3D[4]->plot();
  for (int n = 0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosProb3D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);
  canvas.cd(1);
  trkNbr2D->plot();
  canvas.cd(2);
  tkcntHistosSig2D[4]->plot();
  for (int n = 0; n != 4; ++n) {
    canvas.cd(3+n);
    tkcntHistosSig2D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);
  canvas.cd(1);
  trkNbr2D->plot();
  canvas.cd(2);
  tkcntHistosProb2D[4]->plot();
  for (int n = 0; n != 4; ++n) {
    canvas.cd(3+n);
    tkcntHistosProb2D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(1,3);
  canvas.cd(1);
  ghostTrackDistanceValuHisto->plot();
  canvas.cd(2);
  ghostTrackDistanceSignHisto->plot();
  canvas.cd(3);
  ghostTrackWeightHisto->plot();

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Print((name + cName + ".ps]").c_str());
}


template <class Container, class Base>
void IPTagPlotter<Container, Base>::epsPlot(const std::string & name)
{
  if (willFinalize_) {
    for (int n = 0; n != 4; ++n) effPurFromHistos[n]->epsPlot(name);
    return;
  }
  trkNbr2D->epsPlot(name);
  trkNbr3D->epsPlot(name);
  tkcntHistosTkProbIPneg2D->epsPlot(name);
  tkcntHistosTkProbIPpos2D->epsPlot(name);
  tkcntHistosTkProbIPneg3D->epsPlot(name);
  tkcntHistosTkProbIPpos3D->epsPlot(name);
  ghostTrackDistanceValuHisto->epsPlot(name);
  ghostTrackDistanceSignHisto->epsPlot(name);
  ghostTrackWeightHisto->epsPlot(name);
  for (int n = 0; n != 5; ++n) {
    tkcntHistosSig2D[n]->epsPlot(name);
    tkcntHistosSig3D[n]->epsPlot(name);
    tkcntHistosVal2D[n]->epsPlot(name);
    tkcntHistosVal3D[n]->epsPlot(name);
    tkcntHistosErr2D[n]->epsPlot(name);
    tkcntHistosErr3D[n]->epsPlot(name);
    tkcntHistosProb2D[n]->epsPlot(name);
    tkcntHistosProb3D[n]->epsPlot(name);
    tkcntHistosDecayLengthVal2D[n]->epsPlot(name);
    tkcntHistosDecayLengthVal3D[n]->epsPlot(name);
    tkcntHistosJetDistVal2D[n]->epsPlot(name);
    tkcntHistosJetDistVal3D[n]->epsPlot(name);
    tkcntHistosTkNChiSqr2D[n]->epsPlot(name);
    tkcntHistosTkNChiSqr3D[n]->epsPlot(name);
    tkcntHistosTkPt2D[n]->epsPlot(name);
    tkcntHistosTkPt3D[n]->epsPlot(name);
    tkcntHistosTkNHits2D[n]->epsPlot(name);
    tkcntHistosTkNHits3D[n]->epsPlot(name);    
    tkcntHistosTkNPixelHits2D[n]->epsPlot(name);
    tkcntHistosTkNPixelHits3D[n]->epsPlot(name);

  }
}

template <class Container, class Base> 
reco::TrackBase::TrackQuality IPTagPlotter<Container, Base>::highestTrackQual(const reco::Track * track) const {
  for (reco::TrackBase::TrackQuality i = reco::TrackBase::highPurity; i != reco::TrackBase::undefQuality; i = reco::TrackBase::TrackQuality(i - 1))
  {
    if (track->quality(i))
      return i;
  }

  return reco::TrackBase::undefQuality;
}
