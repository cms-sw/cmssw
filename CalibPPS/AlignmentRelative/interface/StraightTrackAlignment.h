/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Cristian Baldenegro (crisx.baldenegro@gmail.com)
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_StraightTrackAlignment_h
#define CalibPPS_AlignmentRelative_StraightTrackAlignment_h

#include <set>
#include <vector>
#include <string>

#include <TMatrixD.h>
#include <TVectorD.h>
#include <TFile.h>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "CalibPPS/AlignmentRelative/interface/AlignmentGeometry.h"
#include "CalibPPS/AlignmentRelative/interface/HitCollection.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentAlgorithm.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentConstraint.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentTask.h"
#include "CalibPPS/AlignmentRelative/interface/LocalTrackFit.h"
#include "CalibPPS/AlignmentRelative/interface/LocalTrackFitter.h"
#include "FWCore/Framework/interface/ESHandle.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class TH1D;
class TGraph;

/**
 *\brief Track-based alignment using straight tracks.
 **/
class StraightTrackAlignment {
public:
  StraightTrackAlignment(const edm::ParameterSet &);
  virtual ~StraightTrackAlignment();

  virtual void begin(edm::ESHandle<CTPPSRPAlignmentCorrectionsData> hRealAlignment,
                     edm::ESHandle<CTPPSGeometry> hRealGeometry,
                     edm::ESHandle<CTPPSGeometry> hMisalignedGeometry);

  virtual void processEvent(const edm::EventID &eventId,
                            const edm::DetSetVector<TotemRPUVPattern> &uvPatternsStrip,
                            const edm::DetSetVector<CTPPSDiamondRecHit> &hitsDiamond,
                            const edm::DetSetVector<CTPPSPixelRecHit> &hitsPixel,
                            const edm::DetSetVector<CTPPSPixelLocalTrack> &tracksPixel);

  /// performs analyses and fill results variable
  virtual void finish();

protected:
  // ---------- input parameters -----------

  /// verbosity level
  unsigned int verbosity;

  /// list of RPs for which the alignment parameters shall be optimized
  std::vector<unsigned int> rpIds;

  /// list of planes to be excluded from processing
  std::vector<unsigned int> excludePlanes;

  /// a characteristic z in mm
  /// to keep values of z small - this helps the numerical solution
  double z0;

  /// the collection of the alignment algorithms
  std::vector<AlignmentAlgorithm *> algorithms;

  /// constraint types
  enum ConstraintsType { ctFixedDetectors, ctStandard };

  /// the chosen type of constraints
  ConstraintsType constraintsType;

  /// stops after this event number has been reached
  signed int maxEvents;

  // ---------- hit/track selection parameters ----------

  /// remove events with impossible signatures (i.e. simultaneously top and bottom)
  bool removeImpossible;

  /// select only tracks with activity in minimal number of units
  unsigned int requireNumberOfUnits;

  /// if a track goes through overlap, select it only if it leaves signal in at least 3 pots
  bool requireAtLeast3PotsInOverlap;

  /// if true, only track through vertical-horizontal overlap are seleceted
  bool requireOverlap;

  /// whether to cut on chi^2/ndf
  bool cutOnChiSqPerNdf;

  /// the value of chi^2/ndf cut threshold
  double chiSqPerNdfCut;

  /// cuts on absolute values of the track angle
  double maxTrackAx;
  double maxTrackAy;

  /// list of RP sets accepted irrespective of the other "require" settings
  std::vector<std::set<unsigned int> > additionalAcceptedRPSets;

  // ---------- output parameters ----------

  /// file name prefix for result files
  std::string fileNamePrefix;

  /// file name prefix for cumulative result files
  std::string cumulativeFileNamePrefix;

  /// file name prefix for cumulative expanded result files
  std::string expandedFileNamePrefix;

  /// file name prefix for cumulative factored result files
  std::string factoredFileNamePrefix;

  /// whether to use long format (many decimal digits) when saving XML files
  bool preciseXMLFormat;

  /// whether to save uncertainties in the result XML files
  bool saveXMLUncertainties;

  /// whether itermediate results (S, CS matrices) of alignments shall be saved
  bool saveIntermediateResults;

  /// the name task data file
  std::string taskDataFileName;

  /// the file with task data
  TFile *taskDataFile;

  // ---------- internal data members ----------

  /// the alignment task to be solved
  AlignmentTask task;

  /// track fitter
  LocalTrackFitter fitter;

  /// (real geometry) alignments before this alignment iteration
  CTPPSRPAlignmentCorrectionsData initialAlignments;

  // ---------- diagnostics parameters and plots ----------

  /// whether to build and save diagnostic plots
  bool buildDiagnosticPlots;

  /// file name for some event selection statistics
  std::string diagnosticsFile;

  signed int eventsTotal;     ///< counter of events
  signed int eventsFitted;    ///< counter of processed tracks
  signed int eventsSelected;  ///< counter of processed tracks

  std::map<std::set<unsigned int>, unsigned long> fittedTracksPerRPSet;  ///< counter of fitted tracks in a certain RP set
  std::map<std::set<unsigned int>, unsigned long>
      selectedTracksPerRPSet;  ///< counter of selected tracks in a certain RP set

  std::map<unsigned int, unsigned int> selectedHitsPerPlane;  ///< counter of selected hits per plane

  TH1D *fitNdfHist_fitted, *fitNdfHist_selected;  ///< fit num. of degrees of freedom histograms for all/selected tracks
  TH1D *fitPHist_fitted, *fitPHist_selected;      ///< fit p-value histograms for all/selected tracks
  TH1D *fitAxHist_fitted, *fitAxHist_selected;    ///< fit ax histograms for all/selected tracks
  TH1D *fitAyHist_fitted, *fitAyHist_selected;    ///< fit ay histograms for all/selected tracks
  TH1D *fitBxHist_fitted, *fitBxHist_selected;    ///< fit bx histograms for all/selected tracks
  TH1D *fitByHist_fitted, *fitByHist_selected;    ///< fit by histograms for all/selected tracks

  struct RPSetPlots {
    std::string name;

    /// normalised chi^2 histograms for all/selected tracks, in linear/logarithmic scale
    TH1D *chisqn_lin_fitted = nullptr, *chisqn_lin_selected = nullptr, *chisqn_log_fitted = nullptr,
         *chisqn_log_selected = nullptr;

    /// plots ax vs. ay
    TGraph *fitAxVsAyGraph_fitted = nullptr, *fitAxVsAyGraph_selected = nullptr;

    /// plots bx vs. by
    TGraph *fitBxVsByGraph_fitted = nullptr, *fitBxVsByGraph_selected = nullptr;

    RPSetPlots() {}

    RPSetPlots(const std::string &_name);

    void free();

    void write() const;
  };

  /// global (all RP sets) chi^2 histograms
  RPSetPlots globalPlots;

  /// chi^2 histograms per RP set
  std::map<std::set<unsigned int>, RPSetPlots> rpSetPlots;

  /// map: detector id --> residua histogram
  struct ResiduaHistogramSet {
    TH1D *total_fitted, *total_selected;
    TGraph *selected_vs_chiSq;
    std::map<std::set<unsigned int>, TH1D *> perRPSet_fitted, perRPSet_selected;
  };

  /// residua histograms
  std::map<unsigned int, ResiduaHistogramSet> residuaHistograms;

  // ----------- methods ------------

  /// creates a new residua histogram
  TH1D *newResiduaHist(const char *name);

  /// fits the collection of hits and removes hits with too high residual/sigma ratio
  /// \param failed whether the fit has failed
  /// \param selectionChanged whether some hits have been removed
  void fitLocalTrack(HitCollection &, LocalTrackFit &, bool &failed, bool &selectionChanged);

  /// removes the hits of pots with too few planes active
  void removeInsufficientPots(HitCollection &, bool &selectionChanged);

  /// builds a selected set of constraints
  void buildConstraints(std::vector<AlignmentConstraint> &);

  /// fills diagnostic (chi^2, residua, ...) histograms
  void updateDiagnosticHistograms(const HitCollection &selection,
                                  const std::set<unsigned int> &selectedRPs,
                                  const LocalTrackFit &trackFit,
                                  bool trackSelected);

  /// converts a set to string
  static std::string setToString(const std::set<unsigned int> &);

  /// result pretty printing routines
  void printN(const char *str, unsigned int N);
  void printLineSeparator(const std::vector<std::map<unsigned int, AlignmentResult> > &);
  void printQuantitiesLine(const std::vector<std::map<unsigned int, AlignmentResult> > &);
  void printAlgorithmsLine(const std::vector<std::map<unsigned int, AlignmentResult> > &);

  /// saves a ROOT file with diagnostic plots
  void saveDiagnostics() const;
};

#endif
