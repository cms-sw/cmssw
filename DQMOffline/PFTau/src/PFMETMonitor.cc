#include "DQMOffline/PFTau/interface/Matchers.h"
#include "DataFormats/METReco/interface/MET.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMOffline/PFTau/interface/PFMETMonitor.h"

#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TROOT.h>

#include <TProfile.h>

//
// -- Constructor
//
PFMETMonitor::PFMETMonitor(Benchmark::Mode mode) : Benchmark(mode), candBench_(mode), matchCandBench_(mode) {
  setRange(0.0, 10e10, -10.0, 10.0, -3.14, 3.14);

  px_ = nullptr;
  sumEt_ = nullptr;
  delta_ex_ = nullptr;
  delta_ex_VS_set_ = nullptr;
  delta_set_VS_set_ = nullptr;
  delta_set_Over_set_VS_set_ = nullptr;

  profile_delta_ex_VS_set_ = nullptr;
  profile_delta_set_VS_set_ = nullptr;
  profile_delta_set_Over_set_VS_set_ = nullptr;
  profileRMS_delta_ex_VS_set_ = nullptr;
  profileRMS_delta_set_VS_set_ = nullptr;
  profileRMS_delta_set_Over_set_VS_set_ = nullptr;

  createMETSpecificHistos_ = false;
  histogramBooked_ = false;
}

//
// -- Destructor
//
PFMETMonitor::~PFMETMonitor() {}

//
// -- Set Parameters accessing them from ParameterSet
//
void PFMETMonitor::setParameters(const edm::ParameterSet &parameterSet) {
  mode_ = (Benchmark::Mode)parameterSet.getParameter<int>("mode");
  createMETSpecificHistos_ = parameterSet.getParameter<bool>("CreateMETSpecificHistos");
  setRange(parameterSet.getParameter<double>("ptMin"),
           parameterSet.getParameter<double>("ptMax"),
           parameterSet.getParameter<double>("etaMin"),
           parameterSet.getParameter<double>("etaMax"),
           parameterSet.getParameter<double>("phiMin"),
           parameterSet.getParameter<double>("phiMax"));

  candBench_.setParameters(mode_);
  matchCandBench_.setParameters(mode_);
}

//
// -- Set Parameters
//
void PFMETMonitor::setParameters(Benchmark::Mode mode,
                                 float ptmin,
                                 float ptmax,
                                 float etamin,
                                 float etamax,
                                 float phimin,
                                 float phimax,
                                 bool metSpHistos) {
  mode_ = mode;
  createMETSpecificHistos_ = metSpHistos;

  setRange(ptmin, ptmax, etamin, etamax, phimin, phimax);

  candBench_.setParameters(mode_);
  matchCandBench_.setParameters(mode_);
}

//
// -- Create histograms accessing parameters from ParameterSet
//
void PFMETMonitor::setup(DQMStore::IBooker &b, const edm::ParameterSet &parameterSet) {
  candBench_.setup(b, parameterSet);
  matchCandBench_.setup(b, parameterSet);

  if (createMETSpecificHistos_ && !histogramBooked_) {
    edm::ParameterSet pxPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPxHistoParameter");
    edm::ParameterSet dpxPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPxHistoParameter");
    edm::ParameterSet dptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtHistoParameter");
    edm::ParameterSet setPS = parameterSet.getParameter<edm::ParameterSet>("SumEtHistoParameter");
    edm::ParameterSet dsetPS = parameterSet.getParameter<edm::ParameterSet>("DeltaSumEtHistoParameter");
    edm::ParameterSet setOvsetPS = parameterSet.getParameter<edm::ParameterSet>("DeltaSumEtOvSumEtHistoParameter");

    if (pxPS.getParameter<bool>("switchOn")) {
      px_ = book1D(b,
                   "px_",
                   "px_;p_{X} (GeV)",
                   pxPS.getParameter<int32_t>("nBin"),
                   pxPS.getParameter<double>("xMin"),
                   pxPS.getParameter<double>("xMax"));
    }
    if (setPS.getParameter<bool>("switchOn")) {
      sumEt_ = book1D(b,
                      "sumEt_",
                      "sumEt_;#sumE_{T}",
                      setPS.getParameter<int32_t>("nBin"),
                      setPS.getParameter<double>("xMin"),
                      setPS.getParameter<double>("xMax"));
    }
    if (dpxPS.getParameter<bool>("switchOn")) {
      delta_ex_ = book1D(b,
                         "delta_ex_",
                         "#DeltaME_{X}",
                         dpxPS.getParameter<int32_t>("nBin"),
                         dpxPS.getParameter<double>("xMin"),
                         dpxPS.getParameter<double>("xMax"));
    }

    if (dpxPS.getParameter<bool>("switchOn")) {
      delta_ex_VS_set_ = book2D(b,
                                "delta_ex_VS_set_",
                                ";SE_{T, true} (GeV);#DeltaE_{X}",
                                setPS.getParameter<int32_t>("nBin"),
                                setPS.getParameter<double>("xMin"),
                                setPS.getParameter<double>("xMax"),
                                dptPS.getParameter<int32_t>("nBin"),
                                dptPS.getParameter<double>("xMin"),
                                dptPS.getParameter<double>("xMax"));
    }
    if (dsetPS.getParameter<bool>("switchOn")) {
      delta_set_VS_set_ = book2D(b,
                                 "delta_set_VS_set_",
                                 ";SE_{T, true} (GeV);#DeltaSE_{T}",
                                 setPS.getParameter<int32_t>("nBin"),
                                 setPS.getParameter<double>("xMin"),
                                 setPS.getParameter<double>("xMax"),
                                 dsetPS.getParameter<int32_t>("nBin"),
                                 dsetPS.getParameter<double>("xMin"),
                                 dsetPS.getParameter<double>("xMax"));
    }
    if (setOvsetPS.getParameter<bool>("switchOn")) {
      delta_set_Over_set_VS_set_ = book2D(b,
                                          "delta_set_Over_set_VS_set_",
                                          ";SE_{T, true} (GeV);#DeltaSE_{T}/SE_{T}",
                                          setPS.getParameter<int32_t>("nBin"),
                                          setPS.getParameter<double>("xMin"),
                                          setPS.getParameter<double>("xMax"),
                                          setOvsetPS.getParameter<int32_t>("nBin"),
                                          setOvsetPS.getParameter<double>("xMin"),
                                          setOvsetPS.getParameter<double>("xMax"));
    }

    // TProfile
    if (dpxPS.getParameter<bool>("switchOn")) {
      profile_delta_ex_VS_set_ = bookProfile(b,
                                             "profile_delta_ex_VS_set_",
                                             ";SE_{T, true} (GeV);#DeltaE_{X}",
                                             setPS.getParameter<int32_t>("nBin"),
                                             setPS.getParameter<double>("xMin"),
                                             setPS.getParameter<double>("xMax"),
                                             dptPS.getParameter<double>("xMin"),
                                             dptPS.getParameter<double>("xMax"),
                                             "");
      profileRMS_delta_ex_VS_set_ = bookProfile(b,
                                                "profileRMS_delta_ex_VS_set_",
                                                ";SE_{T, true} (GeV);#DeltaE_{X}",
                                                setPS.getParameter<int32_t>("nBin"),
                                                setPS.getParameter<double>("xMin"),
                                                setPS.getParameter<double>("xMax"),
                                                dptPS.getParameter<double>("xMin"),
                                                dptPS.getParameter<double>("xMax"),
                                                "s");
    }
    if (dsetPS.getParameter<bool>("switchOn")) {
      profile_delta_set_VS_set_ = bookProfile(b,
                                              "profile_delta_set_VS_set_",
                                              ";SE_{T, true} (GeV);#DeltaSE_{T}",
                                              setPS.getParameter<int32_t>("nBin"),
                                              setPS.getParameter<double>("xMin"),
                                              setPS.getParameter<double>("xMax"),
                                              dsetPS.getParameter<double>("xMin"),
                                              dsetPS.getParameter<double>("xMax"),
                                              "");
      profileRMS_delta_set_VS_set_ = bookProfile(b,
                                                 "profileRMS_delta_set_VS_set_",
                                                 ";SE_{T, true} (GeV);#DeltaSE_{T}",
                                                 setPS.getParameter<int32_t>("nBin"),
                                                 setPS.getParameter<double>("xMin"),
                                                 setPS.getParameter<double>("xMax"),
                                                 dsetPS.getParameter<double>("xMin"),
                                                 dsetPS.getParameter<double>("xMax"),
                                                 "s");
    }
    if (setOvsetPS.getParameter<bool>("switchOn")) {
      profile_delta_set_Over_set_VS_set_ = bookProfile(b,
                                                       "profile_delta_set_Over_set_VS_set_",
                                                       ";SE_{T, true} (GeV);#DeltaSE_{T}/SE_{T}",
                                                       setPS.getParameter<int32_t>("nBin"),
                                                       setPS.getParameter<double>("xMin"),
                                                       setPS.getParameter<double>("xMax"),
                                                       setOvsetPS.getParameter<double>("xMin"),
                                                       setOvsetPS.getParameter<double>("xMax"),
                                                       "");
      profileRMS_delta_set_Over_set_VS_set_ = bookProfile(b,
                                                          "profileRMS_delta_set_Over_set_VS_set_",
                                                          ";SE_{T, true} (GeV);#DeltaSE_{T}/SE_{T}",
                                                          setPS.getParameter<int32_t>("nBin"),
                                                          setPS.getParameter<double>("xMin"),
                                                          setPS.getParameter<double>("xMax"),
                                                          setOvsetPS.getParameter<double>("xMin"),
                                                          setOvsetPS.getParameter<double>("xMax"),
                                                          "s");
    }
    histogramBooked_ = true;
  }
}

//
// -- Create histograms using local parameters
//
void PFMETMonitor::setup(DQMStore::IBooker &b) {
  candBench_.setup(b);
  matchCandBench_.setup(b);

  if (createMETSpecificHistos_ && !histogramBooked_) {
    PhaseSpace pxPS = PhaseSpace(50, 0, 200);
    PhaseSpace dpxPS = PhaseSpace(50, -500, 500);
    PhaseSpace setPS = PhaseSpace(50, 0.0, 3000);
    PhaseSpace dsetPS = PhaseSpace(50, -1000.0, 1000);
    PhaseSpace setOvsetPS = PhaseSpace(100, 0., 2.);

    px_ = book1D(b, "px_", "px_;p_{X} (GeV)", pxPS.n, pxPS.m, pxPS.M);
    sumEt_ = book1D(b, "sumEt_", "sumEt_;#sumE_{T}", setPS.n, setPS.m, setPS.M);
    delta_ex_ = book1D(b, "delta_ex_", "#DeltaME_{X}", dpxPS.n, dpxPS.m, dpxPS.M);
    delta_ex_VS_set_ = book2D(
        b, "delta_ex_VS_set_", ";SE_{T, true} (GeV);#DeltaE_{X}", setPS.n, setPS.m, setPS.M, dpxPS.n, dpxPS.m, dpxPS.M);
    delta_set_VS_set_ = book2D(b,
                               "delta_set_VS_set_",
                               ";SE_{T, true} (GeV);#DeltaSE_{T}",
                               setPS.n,
                               setPS.m,
                               setPS.M,
                               dsetPS.n,
                               dsetPS.m,
                               dsetPS.M);

    delta_set_Over_set_VS_set_ = book2D(b,
                                        "delta_set_Over_set_VS_set_",
                                        ";SE_{T, true} (GeV);#DeltaSE_{T}/SE_{T}",
                                        setPS.n,
                                        setPS.m,
                                        setPS.M,
                                        setOvsetPS.n,
                                        setOvsetPS.m,
                                        setOvsetPS.M);

    // TProfile
    profile_delta_ex_VS_set_ = bookProfile(b,
                                           "profile_delta_ex_VS_set_",
                                           ";SE_{T, true} (GeV);#DeltaE_{X}",
                                           setPS.n,
                                           setPS.m,
                                           setPS.M,
                                           setOvsetPS.m,
                                           setOvsetPS.M,
                                           "");

    profile_delta_set_VS_set_ = bookProfile(b,
                                            "profile_delta_set_VS_set_",
                                            ";SE_{T, true} (GeV);#DeltaSE_{T}",
                                            setPS.n,
                                            setPS.m,
                                            setPS.M,
                                            setOvsetPS.m,
                                            setOvsetPS.M,
                                            "");

    profile_delta_set_Over_set_VS_set_ = bookProfile(b,
                                                     "profile_delta_set_Over_set_VS_set_",
                                                     ";SE_{T, true} (GeV);#DeltaSE_{T}/SE_{T}",
                                                     setPS.n,
                                                     setPS.m,
                                                     setPS.M,
                                                     setOvsetPS.m,
                                                     setOvsetPS.M,
                                                     "");

    // TProfile RMS
    profileRMS_delta_ex_VS_set_ = bookProfile(b,
                                              "profileRMS_delta_ex_VS_set_",
                                              ";SE_{T, true} (GeV);#DeltaE_{X}",
                                              setPS.n,
                                              setPS.m,
                                              setPS.M,
                                              setOvsetPS.m,
                                              setOvsetPS.M,
                                              "s");

    profileRMS_delta_set_VS_set_ = bookProfile(b,
                                               "profileRMS_delta_set_VS_set_",
                                               ";SE_{T, true} (GeV);#DeltaSE_{T}",
                                               setPS.n,
                                               setPS.m,
                                               setPS.M,
                                               setOvsetPS.m,
                                               setOvsetPS.M,
                                               "s");

    profileRMS_delta_set_Over_set_VS_set_ = bookProfile(b,
                                                        "profileRMS_delta_set_Over_set_VS_set_",
                                                        ";SE_{T, true} (GeV);#DeltaSE_{T}/SE_{T}",
                                                        setPS.n,
                                                        setPS.m,
                                                        setPS.M,
                                                        setOvsetPS.m,
                                                        setOvsetPS.M,
                                                        "s");
    histogramBooked_ = true;
  }
}

void PFMETMonitor::setDirectory(TDirectory *dir) {
  Benchmark::setDirectory(dir);

  candBench_.setDirectory(dir);
  matchCandBench_.setDirectory(dir);
}

void PFMETMonitor::fillOne(const reco::MET &met, const reco::MET &matchedMet, float &minVal, float &maxVal) {
  candBench_.fillOne(met);                   // std::cout <<"\nfillone MET candBench" <<std::endl;
  matchCandBench_.fillOne(met, matchedMet);  // std::cout <<"\nfillone MET
                                             // MatchCandBench done" <<std::endl;

  if (createMETSpecificHistos_ && histogramBooked_) {
    if (!isInRange(met.pt(), met.eta(), met.phi()))
      return;

    if (px_)
      px_->Fill(met.px());
    if (delta_ex_) {
      delta_ex_->Fill(met.px() - matchedMet.px());
      delta_ex_->Fill(met.py() - matchedMet.py());
    }
    if (sumEt_)
      sumEt_->Fill(met.sumEt());

    if (delta_ex_VS_set_) {
      delta_ex_VS_set_->Fill(matchedMet.sumEt(), met.px() - matchedMet.px());
      delta_ex_VS_set_->Fill(matchedMet.sumEt(), met.py() - matchedMet.py());
      profile_delta_ex_VS_set_->Fill(matchedMet.sumEt(), met.px() - matchedMet.px());
      profile_delta_ex_VS_set_->Fill(matchedMet.sumEt(), met.py() - matchedMet.py());
      profileRMS_delta_ex_VS_set_->Fill(matchedMet.sumEt(), met.px() - matchedMet.px());
      profileRMS_delta_ex_VS_set_->Fill(matchedMet.sumEt(), met.py() - matchedMet.py());
    }
    if (delta_set_VS_set_) {
      delta_set_VS_set_->Fill(matchedMet.sumEt(), met.sumEt() - matchedMet.sumEt());
      profile_delta_set_VS_set_->Fill(matchedMet.sumEt(), met.sumEt() - matchedMet.sumEt());
      profileRMS_delta_set_VS_set_->Fill(matchedMet.sumEt(), met.sumEt() - matchedMet.sumEt());
    }
    if (delta_set_Over_set_VS_set_ && matchedMet.sumEt() > 0.001) {
      float setRes = (met.sumEt() - matchedMet.sumEt()) / matchedMet.sumEt();
      if (setRes > maxVal)
        maxVal = setRes;
      if (setRes < minVal)
        minVal = setRes;
      delta_set_Over_set_VS_set_->Fill(matchedMet.sumEt(), setRes);
      profile_delta_set_Over_set_VS_set_->Fill(matchedMet.sumEt(), setRes);
      profileRMS_delta_set_Over_set_VS_set_->Fill(matchedMet.sumEt(), setRes);
    }
  }
}
