#ifndef Alignment_APEEstimation_TrackerSectorStruct_h
#define Alignment_APEEstimation_TrackerSectorStruct_h

#include <vector>
#include <map>
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TString.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "Alignment/APEEstimation/interface/EventVariables.h"

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
//class TH1;
//class TH2;
//class TH1F;
//class TH2F;
//class TProfile;
//class TString;
//class TFileDirectory;

class TrackerSectorStruct {
public:
  inline TrackerSectorStruct();

  inline ~TrackerSectorStruct();

  struct CorrelationHists {
    CorrelationHists()
        : Variable(nullptr),
          NorResXVsVar(nullptr),
          ProbXVsVar(nullptr),
          SigmaXHitVsVar(nullptr),
          SigmaXTrkVsVar(nullptr),
          SigmaXVsVar(nullptr),
          PNorResXVsVar(nullptr),
          PProbXVsVar(nullptr),
          PSigmaXHitVsVar(nullptr),
          PSigmaXTrkVsVar(nullptr),
          PSigmaXVsVar(nullptr){};

    inline void fillCorrHists(const TString,
                              const TrackStruct::HitParameterStruct &hitParameterStruct,
                              double variable);
    inline void fillCorrHistsX(const TrackStruct::HitParameterStruct &hitParameterStruct, double variable);
    inline void fillCorrHistsY(const TrackStruct::HitParameterStruct &hitParameterStruct, double variable);

    TH1F *Variable;
    TH2F *NorResXVsVar, *ProbXVsVar, *SigmaXHitVsVar, *SigmaXTrkVsVar, *SigmaXVsVar;
    TProfile *PNorResXVsVar, *PProbXVsVar, *PSigmaXHitVsVar, *PSigmaXTrkVsVar, *PSigmaXVsVar;
  };

  inline void setCorrHistParams(TFileDirectory *, double, double, double);
  inline CorrelationHists bookCorrHists(
      TString, TString, TString, TString, TString, int, int, double, double, std::string = "nphtr");
  inline CorrelationHists bookCorrHistsX(
      TString, TString, TString, TString, int, int, double, double, std::string = "nphtr");
  inline CorrelationHists bookCorrHistsY(
      TString, TString, TString, TString, int, int, double, double, std::string = "nphtr");
  /// same, but without booking 1D histo
  inline CorrelationHists bookCorrHists(TString, TString, TString, TString, int, double, double, std::string = "nphtr");
  inline CorrelationHists bookCorrHistsX(TString, TString, TString, int, double, double, std::string = "nphtr");
  inline CorrelationHists bookCorrHistsY(TString, TString, TString, int, double, double, std::string = "nphtr");

  TFileDirectory *directory_;
  double norResXMax_, sigmaXHitMax_, sigmaXMax_;  // Used for x and y
  std::map<std::string, CorrelationHists> m_correlationHistsX;
  std::map<std::string, CorrelationHists> m_correlationHistsY;

  // Name of sector as string and as title of a histogram
  std::string name;
  TH1 *Name;

  // Module IDs of modules in sector
  std::vector<unsigned int> v_rawId;

  TH1 *ResX, *NorResX, *XHit, *XTrk, *SigmaX2, *ProbX;
  TH2 *WidthVsPhiSensX, *WidthVsWidthProjected, *WidthDiffVsMaxStrip, *WidthDiffVsSigmaXHit, *PhiSensXVsBarycentreX;
  TProfile *PWidthVsPhiSensX, *PWidthVsWidthProjected, *PWidthDiffVsMaxStrip, *PWidthDiffVsSigmaXHit,
      *PPhiSensXVsBarycentreX;
  std::map<std::string, std::vector<TH1 *> > m_sigmaX;

  TH1 *ResY, *NorResY, *YHit, *YTrk, *SigmaY2, *ProbY;
  TH2 *PhiSensYVsBarycentreY;
  TProfile *PPhiSensYVsBarycentreY;
  std::map<std::string, std::vector<TH1 *> > m_sigmaY;

  //for every bin in sigmaX or sigmaY the needful histos to calculate the APE
  std::map<unsigned int, std::map<std::string, TH1 *> > m_binnedHists;

  //for presenting results
  TTree *RawId;
  TH1 *EntriesX;
  TH1 *WeightX, *MeanX, *RmsX, *FitMeanX1, *ResidualWidthX1, *CorrectionX1, *FitMeanX2, *ResidualWidthX2, *CorrectionX2;
  TH1 *EntriesY;
  TH1 *WeightY, *MeanY, *RmsY, *FitMeanY1, *ResidualWidthY1, *CorrectionY1, *FitMeanY2, *ResidualWidthY2, *CorrectionY2;

  // To book pixel-specific or strip-specific histos only
  bool isPixel;
};

TrackerSectorStruct::TrackerSectorStruct()
    : directory_(nullptr),
      norResXMax_(999.),
      sigmaXHitMax_(999.),
      sigmaXMax_(999.),
      name("default"),
      Name(nullptr),
      ResX(nullptr),
      NorResX(nullptr),
      XHit(nullptr),
      XTrk(nullptr),
      SigmaX2(nullptr),
      ProbX(nullptr),
      WidthVsPhiSensX(nullptr),
      WidthVsWidthProjected(nullptr),
      WidthDiffVsMaxStrip(nullptr),
      WidthDiffVsSigmaXHit(nullptr),
      PhiSensXVsBarycentreX(nullptr),
      PWidthVsPhiSensX(nullptr),
      PWidthVsWidthProjected(nullptr),
      PWidthDiffVsMaxStrip(nullptr),
      PWidthDiffVsSigmaXHit(nullptr),
      PPhiSensXVsBarycentreX(nullptr),
      ResY(nullptr),
      NorResY(nullptr),
      YHit(nullptr),
      YTrk(nullptr),
      SigmaY2(nullptr),
      ProbY(nullptr),
      PhiSensYVsBarycentreY(nullptr),
      PPhiSensYVsBarycentreY(nullptr),
      RawId(nullptr),
      EntriesX(nullptr),
      MeanX(nullptr),
      RmsX(nullptr),
      FitMeanX1(nullptr),
      ResidualWidthX1(nullptr),
      CorrectionX1(nullptr),
      FitMeanX2(nullptr),
      ResidualWidthX2(nullptr),
      CorrectionX2(nullptr),
      EntriesY(nullptr),
      MeanY(nullptr),
      RmsY(nullptr),
      FitMeanY1(nullptr),
      ResidualWidthY1(nullptr),
      CorrectionY1(nullptr),
      FitMeanY2(nullptr),
      ResidualWidthY2(nullptr),
      CorrectionY2(nullptr),
      isPixel(false) {}

TrackerSectorStruct::~TrackerSectorStruct() {}

void TrackerSectorStruct::setCorrHistParams(TFileDirectory *directory,
                                            double norResXMax,
                                            double sigmaXHitMax,
                                            double sigmaXMax) {
  directory_ = directory;
  norResXMax_ = norResXMax;
  sigmaXHitMax_ = sigmaXHitMax;
  sigmaXMax_ = sigmaXMax;
}

TrackerSectorStruct::CorrelationHists TrackerSectorStruct::bookCorrHistsX(TString varName,
                                                                          TString varTitle,
                                                                          TString labelX,
                                                                          TString unitX,
                                                                          int nBinX1D,
                                                                          int nBinX2D,
                                                                          double minBinX,
                                                                          double maxBinX,
                                                                          std::string options) {
  return bookCorrHists("X", varName, varTitle, labelX, unitX, nBinX1D, nBinX2D, minBinX, maxBinX, options);
}
TrackerSectorStruct::CorrelationHists TrackerSectorStruct::bookCorrHistsY(TString varName,
                                                                          TString varTitle,
                                                                          TString labelX,
                                                                          TString unitX,
                                                                          int nBinX1D,
                                                                          int nBinX2D,
                                                                          double minBinX,
                                                                          double maxBinX,
                                                                          std::string options) {
  return bookCorrHists("Y", varName, varTitle, labelX, unitX, nBinX1D, nBinX2D, minBinX, maxBinX, options);
}
TrackerSectorStruct::CorrelationHists TrackerSectorStruct::bookCorrHists(TString xY,
                                                                         TString varName,
                                                                         TString varTitle,
                                                                         TString labelX,
                                                                         TString unitX,
                                                                         int nBinX1D,
                                                                         int nBinX2D,
                                                                         double minBinX,
                                                                         double maxBinX,
                                                                         std::string options) {
  TString xy;
  TString suffix;
  if (xY == "X") {
    xy = "x";
    suffix = "";
  }
  if (xY == "Y") {
    xy = "y";
    suffix = "_y";
  }

  const std::string &o(options);
  CorrelationHists correlationHists;

  if (!(o.find('n') != std::string::npos || o.find('p') != std::string::npos || o.find('h') != std::string::npos ||
        o.find('t') != std::string::npos || o.find('r') != std::string::npos))
    return correlationHists;

  TFileDirectory *directory(directory_);
  double norResXMax(norResXMax_), sigmaXHitMax(sigmaXHitMax_), sigmaXMax(sigmaXMax_);

  if (!directory)
    return correlationHists;

  correlationHists.Variable = directory->make<TH1F>("h_" + varName + suffix,
                                                    varTitle + " " + labelX + ";" + labelX + "  " + unitX + ";# hits",
                                                    nBinX1D,
                                                    minBinX,
                                                    maxBinX);

  if (options.find('n') != std::string::npos)
    correlationHists.NorResXVsVar =
        directory->make<TH2F>("h2_norRes" + xY + "Vs" + varName,
                              "r_{" + xy + "}/#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX +
                                  ";(" + xy + "_{trk}-" + xy + "_{hit})/#sigma_{r," + xy + "}",
                              nBinX2D,
                              minBinX,
                              maxBinX,
                              25,
                              -norResXMax,
                              norResXMax);
  if (options.find('p') != std::string::npos)
    correlationHists.ProbXVsVar =
        directory->make<TH2F>("h2_prob" + xY + "Vs" + varName,
                              "prob_{" + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";prob_{" + xy + "}",
                              nBinX2D,
                              minBinX,
                              maxBinX,
                              60,
                              -0.1,
                              1.1);
  if (options.find('h') != std::string::npos)
    correlationHists.SigmaXHitVsVar = directory->make<TH2F>(
        "h2_sigma" + xY + "HitVs" + varName,
        "#sigma_{hit," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{hit," + xy + "}  [#mum]",
        nBinX2D,
        minBinX,
        maxBinX,
        50,
        0 * 10000.,
        sigmaXHitMax * 10000.);
  if (options.find('t') != std::string::npos)
    correlationHists.SigmaXTrkVsVar = directory->make<TH2F>(
        "h2_sigma" + xY + "TrkVs" + varName,
        "#sigma_{trk," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{trk," + xy + "}  [#mum]",
        nBinX2D,
        minBinX,
        maxBinX,
        50,
        0 * 10000.,
        sigmaXMax * 10000.);
  if (options.find('r') != std::string::npos)
    correlationHists.SigmaXVsVar = directory->make<TH2F>(
        "h2_sigma" + xY + "Vs" + varName,
        "#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{r," + xy + "}  [#mum]",
        nBinX2D,
        minBinX,
        maxBinX,
        50,
        0 * 10000.,
        sigmaXMax * 10000.);

  if (options.find('n') != std::string::npos)
    correlationHists.PNorResXVsVar =
        directory->make<TProfile>("p_norRes" + xY + "Vs" + varName,
                                  "r_{" + xy + "}/#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX +
                                      ";(" + xy + "_{trk}-" + xy + "_{hit})/#sigma_{r," + xy + "}",
                                  nBinX2D,
                                  minBinX,
                                  maxBinX,
                                  "s");
  if (options.find('p') != std::string::npos)
    correlationHists.PProbXVsVar = directory->make<TProfile>(
        "p_prob" + xY + "Vs" + varName,
        "prob_{" + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";prob_{" + xy + "}",
        nBinX2D,
        minBinX,
        maxBinX,
        "s");
  if (options.find('h') != std::string::npos)
    correlationHists.PSigmaXHitVsVar = directory->make<TProfile>(
        "p_sigma" + xY + "HitVs" + varName,
        "#sigma_{hit," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{hit," + xy + "}  [#mum]",
        nBinX2D,
        minBinX,
        maxBinX);
  if (options.find('t') != std::string::npos)
    correlationHists.PSigmaXTrkVsVar = directory->make<TProfile>(
        "p_sigma" + xY + "TrkVs" + varName,
        "#sigma_{trk," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{trk," + xy + "}  [#mum]",
        nBinX2D,
        minBinX,
        maxBinX);
  if (options.find('r') != std::string::npos)
    correlationHists.PSigmaXVsVar = directory->make<TProfile>(
        "p_sigma" + xY + "Vs" + varName,
        "#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{r," + xy + "}  [#mum]",
        nBinX2D,
        minBinX,
        maxBinX);

  return correlationHists;
}

TrackerSectorStruct::CorrelationHists TrackerSectorStruct::bookCorrHistsX(
    TString varName, TString labelX, TString unitX, int nBinX, double minBinX, double maxBinX, std::string options) {
  return bookCorrHists("X", varName, labelX, unitX, nBinX, minBinX, maxBinX, options);
}
TrackerSectorStruct::CorrelationHists TrackerSectorStruct::bookCorrHistsY(
    TString varName, TString labelX, TString unitX, int nBinX, double minBinX, double maxBinX, std::string options) {
  return bookCorrHists("Y", varName, labelX, unitX, nBinX, minBinX, maxBinX, options);
}
TrackerSectorStruct::CorrelationHists TrackerSectorStruct::bookCorrHists(TString xY,
                                                                         TString varName,
                                                                         TString labelX,
                                                                         TString unitX,
                                                                         int nBinX,
                                                                         double minBinX,
                                                                         double maxBinX,
                                                                         std::string options) {
  TString xy;
  if (xY == "X") {
    xy = "x";
  }
  if (xY == "Y") {
    xy = "y";
  }

  const std::string &o(options);
  CorrelationHists correlationHists;

  if (!(o.find('n') != std::string::npos || o.find('p') != std::string::npos || o.find('h') != std::string::npos ||
        o.find('t') != std::string::npos || o.find('r') != std::string::npos))
    return correlationHists;

  TFileDirectory *directory(directory_);
  double norResXMax(norResXMax_), sigmaXHitMax(sigmaXHitMax_), sigmaXMax(sigmaXMax_);

  if (!directory)
    return correlationHists;

  if (options.find('n') != std::string::npos)
    correlationHists.NorResXVsVar =
        directory->make<TH2F>("h2_norRes" + xY + "Vs" + varName,
                              "r_{" + xy + "}/#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX +
                                  ";(" + xy + "_{trk}-" + xy + "_{hit})/#sigma_{r," + xy + "}",
                              nBinX,
                              minBinX,
                              maxBinX,
                              25,
                              -norResXMax,
                              norResXMax);
  if (options.find('p') != std::string::npos)
    correlationHists.ProbXVsVar =
        directory->make<TH2F>("h2_prob" + xY + "Vs" + varName,
                              "prob_{" + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";prob_{" + xy + "}",
                              nBinX,
                              minBinX,
                              maxBinX,
                              60,
                              -0.1,
                              1.1);
  if (options.find('h') != std::string::npos)
    correlationHists.SigmaXHitVsVar = directory->make<TH2F>(
        "h2_sigma" + xY + "HitVs" + varName,
        "#sigma_{hit," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{hit," + xy + "}  [#mum]",
        nBinX,
        minBinX,
        maxBinX,
        50,
        0 * 10000.,
        sigmaXHitMax * 10000.);
  if (options.find('t') != std::string::npos)
    correlationHists.SigmaXTrkVsVar = directory->make<TH2F>(
        "h2_sigma" + xY + "TrkVs" + varName,
        "#sigma_{trk," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{trk," + xy + "}  [#mum]",
        nBinX,
        minBinX,
        maxBinX,
        50,
        0 * 10000.,
        sigmaXMax * 10000.);
  if (options.find('r') != std::string::npos)
    correlationHists.SigmaXVsVar = directory->make<TH2F>(
        "h2_sigma" + xY + "Vs" + varName,
        "#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{r," + xy + "}  [#mum]",
        nBinX,
        minBinX,
        maxBinX,
        50,
        0 * 10000.,
        sigmaXMax * 10000.);

  if (options.find('n') != std::string::npos)
    correlationHists.PNorResXVsVar =
        directory->make<TProfile>("p_norRes" + xY + "Vs" + varName,
                                  "r_{" + xy + "}/#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX +
                                      ";(" + xy + "_{trk}-" + xy + "_{hit})/#sigma_{r," + xy + "}",
                                  nBinX,
                                  minBinX,
                                  maxBinX,
                                  "s");
  if (options.find('p') != std::string::npos)
    correlationHists.PProbXVsVar = directory->make<TProfile>(
        "p_prob" + xY + "Vs" + varName,
        "prob_{" + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";prob_{" + xy + "}",
        nBinX,
        minBinX,
        maxBinX,
        "s");
  if (options.find('h') != std::string::npos)
    correlationHists.PSigmaXHitVsVar = directory->make<TProfile>(
        "p_sigma" + xY + "HitVs" + varName,
        "#sigma_{hit," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{hit," + xy + "}  [#mum]",
        nBinX,
        minBinX,
        maxBinX);
  if (options.find('t') != std::string::npos)
    correlationHists.PSigmaXTrkVsVar = directory->make<TProfile>(
        "p_sigma" + xY + "TrkVs" + varName,
        "#sigma_{trk," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{trk," + xy + "}  [#mum]",
        nBinX,
        minBinX,
        maxBinX);
  if (options.find('r') != std::string::npos)
    correlationHists.PSigmaXVsVar = directory->make<TProfile>(
        "p_sigma" + xY + "Vs" + varName,
        "#sigma_{r," + xy + "} vs. " + labelX + ";" + labelX + "  " + unitX + ";#sigma_{r," + xy + "}  [#mum]",
        nBinX,
        minBinX,
        maxBinX);

  return correlationHists;
}

void TrackerSectorStruct::CorrelationHists::fillCorrHistsX(const TrackStruct::HitParameterStruct &hitParameterStruct,
                                                           double variable) {
  return fillCorrHists("X", hitParameterStruct, variable);
}
void TrackerSectorStruct::CorrelationHists::fillCorrHistsY(const TrackStruct::HitParameterStruct &hitParameterStruct,
                                                           double variable) {
  return fillCorrHists("Y", hitParameterStruct, variable);
}
void TrackerSectorStruct::CorrelationHists::fillCorrHists(const TString xY,
                                                          const TrackStruct::HitParameterStruct &hitParameterStruct,
                                                          double variable) {
  float norRes(999.);
  float prob(999.);
  float errHit(999.);
  float errTrk(999.);
  float err(999.);
  if (xY == "X") {
    norRes = hitParameterStruct.norResX;
    prob = hitParameterStruct.probX;
    errHit = hitParameterStruct.errXHit;
    errTrk = hitParameterStruct.errXTrk;
    err = hitParameterStruct.errX;
  }
  if (xY == "Y") {
    norRes = hitParameterStruct.norResY;
    prob = hitParameterStruct.probY;
    errHit = hitParameterStruct.errYHit;
    errTrk = hitParameterStruct.errYTrk;
    err = hitParameterStruct.errY;
  }

  if (Variable) {
    Variable->Fill(variable);
  }

  if (NorResXVsVar) {
    NorResXVsVar->Fill(variable, norRes);
    PNorResXVsVar->Fill(variable, norRes);
  }
  if (ProbXVsVar) {
    ProbXVsVar->Fill(variable, prob);
    PProbXVsVar->Fill(variable, prob);
  }
  if (SigmaXHitVsVar) {
    SigmaXHitVsVar->Fill(variable, errHit * 10000.);
    PSigmaXHitVsVar->Fill(variable, errHit * 10000.);
  }
  if (SigmaXTrkVsVar) {
    SigmaXTrkVsVar->Fill(variable, errTrk * 10000.);
    PSigmaXTrkVsVar->Fill(variable, errTrk * 10000.);
  }
  if (SigmaXVsVar) {
    SigmaXVsVar->Fill(variable, err * 10000.);
    PSigmaXVsVar->Fill(variable, err * 10000.);
  }
}

#endif
