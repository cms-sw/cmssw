#ifndef PFProducer_PFCandConnector_H_
#define PFProducer_PFCandConnector_H_

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// \author : M. Gouzevitch
// \date : May 2010

/// Based on a class from : V. Roberfroid, February 2008

class PFCandConnector {
public:
  PFCandConnector() {
    bCorrect_ = false;
    bCalibPrimary_ = false;

    fConst_.push_back(1), fConst_.push_back(0);
    fNorm_.push_back(0), fNorm_.push_back(0);
    fExp_.push_back(0);

    dptRel_PrimaryTrack_ = 0.;
    dptRel_MergedTrack_ = 0.;
    ptErrorSecondary_ = 0.;
  }

  void setParameters(const edm::ParameterSet& iCfgCandConnector) {
    bool bCorrect, bCalibPrimary;
    double dptRel_PrimaryTrack, dptRel_MergedTrack, ptErrorSecondary;
    std::vector<double> nuclCalibFactors;

    /// Flag to apply the correction procedure for nuclear interactions
    bCorrect = iCfgCandConnector.getParameter<bool>("bCorrect");
    /// Flag to calibrate the reconstructed nuclear interactions with primary or merged tracks
    bCalibPrimary = iCfgCandConnector.getParameter<bool>("bCalibPrimary");

    if (iCfgCandConnector.exists("dptRel_PrimaryTrack"))
      dptRel_PrimaryTrack = iCfgCandConnector.getParameter<double>("dptRel_PrimaryTrack");
    else {
      edm::LogWarning("PFCandConnector") << "dptRel_PrimaryTrack doesn't exist. Setting a default safe value 0"
                                         << std::endl;
      dptRel_PrimaryTrack = 0;
    }

    if (iCfgCandConnector.exists("dptRel_MergedTrack"))
      dptRel_MergedTrack = iCfgCandConnector.getParameter<double>("dptRel_MergedTrack");
    else {
      edm::LogWarning("PFCandConnector") << "dptRel_MergedTrack doesn't exist. Setting a default safe value 0"
                                         << std::endl;
      dptRel_MergedTrack = 0;
    }

    if (iCfgCandConnector.exists("ptErrorSecondary"))
      ptErrorSecondary = iCfgCandConnector.getParameter<double>("ptErrorSecondary");
    else {
      edm::LogWarning("PFCandConnector") << "ptErrorSecondary doesn't exist. Setting a default safe value 0"
                                         << std::endl;
      ptErrorSecondary = 0;
    }

    if (iCfgCandConnector.exists("nuclCalibFactors"))
      nuclCalibFactors = iCfgCandConnector.getParameter<std::vector<double> >("nuclCalibFactors");
    else {
      edm::LogWarning("PFCandConnector") << "nuclear calib factors doesn't exist the factor would not be applyed"
                                         << std::endl;
    }

    setParameters(bCorrect, bCalibPrimary, dptRel_PrimaryTrack, dptRel_MergedTrack, ptErrorSecondary, nuclCalibFactors);
  }

  void setParameters(bool bCorrect,
                     bool bCalibPrimary,
                     double dptRel_PrimaryTrack,
                     double dptRel_MergedTrack,
                     double ptErrorSecondary,
                     const std::vector<double>& nuclCalibFactors);

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  reco::PFCandidateCollection connect(reco::PFCandidateCollection& pfCand) const;

private:
  /// Analyse nuclear interactions where a primary or merged track is present
  void analyseNuclearWPrim(reco::PFCandidateCollection&, std::vector<bool>&, unsigned int) const;

  /// Analyse nuclear interactions where a secondary track is present
  void analyseNuclearWSec(reco::PFCandidateCollection&, std::vector<bool>&, unsigned int) const;

  bool isPrimaryNucl(const reco::PFCandidate& pf) const;

  bool isSecondaryNucl(const reco::PFCandidate& pf) const;

  /// Return a calibration factor for a reconstructed nuclear interaction
  double rescaleFactor(const double pt, const double cFrac) const;

  /// Parameters
  bool bCorrect_;

  /// Calibration parameters for the reconstructed nuclear interactions
  bool bCalibPrimary_;
  std::vector<double> fConst_;
  std::vector<double> fNorm_;
  std::vector<double> fExp_;

  // Maximal accepatble uncertainty on primary tracks to usem them as MC truth for calibration
  double dptRel_PrimaryTrack_;
  double dptRel_MergedTrack_;
  double ptErrorSecondary_;

  /// Useful constants
  static const double pion_mass2;
  static const reco::PFCandidate::Flags fT_TO_DISP_;
  static const reco::PFCandidate::Flags fT_FROM_DISP_;
};

#endif
