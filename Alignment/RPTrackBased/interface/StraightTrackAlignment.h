/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Alignment_RPTrackBased_StraightTrackAlignment
#define Alignment_RPTrackBased_StraightTrackAlignment

#include <set>
#include <vector>
#include <string>

#include <TMatrixD.h>
#include <TVectorD.h>
#include <TFile.h>

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"
#include "Alignment/RPDataFormats/interface/LocalTrackFit.h"
#include "Alignment/RPTrackBased/interface/AlignmentGeometry.h"
#include "Alignment/RPTrackBased/interface/HitCollection.h"
#include "Alignment/RPTrackBased/interface/AlignmentAlgorithm.h"
#include "Alignment/RPTrackBased/interface/AlignmentConstraint.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"
#include "Alignment/RPTrackBased/interface/LocalTrackFitter.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TH1D;
class TGraph;
  

/**
 *\brief Track-based alignment using straight tracks.
 **/
class StraightTrackAlignment
{
  public:
    StraightTrackAlignment(const edm::ParameterSet&);
    virtual ~StraightTrackAlignment();

    virtual void Begin(const edm::EventSetup&);
    virtual void ProcessEvent(const edm::Event&, const edm::EventSetup&);
    
    /// performs analyses and fill results variable
    virtual void Finish();

  protected:
    friend class RPStraightTrackAligner;
    friend class StraightTrackAlignmentIdealResult;

    // ---------- input parameters -----------

    /// verbosity level
    unsigned int verbosity;
    
    /// verbosity level while factorization
    unsigned int factorizationVerbosity;

    /// selection of input (pattern-recognition result)
    edm::InputTag tagRecognizedPatterns;

    /// list of RPs for which the alignment parameters shall be optimized
    std::vector<unsigned int> RPIds;

    /// list of planes to be excluded from processing
    std::vector<unsigned int> excludePlanes;

    /// a characteristic z in mm
    /// to keep values of z small - this helps the numerical solution
    double z0;

    /// the collection of the alignment algorithms
    std::vector<AlignmentAlgorithm *> algorithms;
    
    /// whether track fit shall be retrieved from an external source
    bool useExternalFitter;

    /// selection of track fit input
    edm::InputTag tagExternalFit;

    /// ctDynamic not yet implemented
    enum ConstraintsType { ctHomogeneous, ctFixedDetectors, ctDynamic, ctFinal };
    
    /// the chosen type of constraints
    ConstraintsType constraintsType;

    /// stops after this event number has been reached
    unsigned int maxEvents;
    
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

    /// the list of run numbers for which the horizontal RP data shall be discarded
    std::vector<unsigned int> runsWithoutHorizontalRPs;

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
    RPAlignmentCorrections initialAlignments;                  
    
    // ---------- diagnostics parameters and plots ----------                                        
    
    /// whether to build and save diagnostic plots
    bool buildDiagnosticPlots;
    
    /// file name for some event selection statistics
    std::string diagnosticsFile;
    
    unsigned long eventsTotal;                                                ///< counter of events
    unsigned long eventsFitted;                                               ///< counter of processed tracks
    unsigned long eventsSelected;                                             ///< counter of processed tracks

    std::map< std::set<unsigned int>, unsigned long> fittedTracksPerRPSet;    ///< counter of fitted tracks in a certain detector set
    std::map< std::set<unsigned int>, unsigned long> selectedTracksPerRPSet;  ///< counter of selected tracks in a certain detector set

    TH1D *fitNdfHist_fitted, *fitNdfHist_selected;                            ///< fit num. of degrees of freedom histograms for all/selected tracks 
    TH1D *fitPHist_fitted, *fitPHist_selected;                                ///< fit p-value histograms for all/selected tracks 
    TH1D *fitAxHist_fitted, *fitAxHist_selected;                              ///< fit ax histograms for all/selected tracks 
    TH1D *fitAyHist_fitted, *fitAyHist_selected;                              ///< fit ay histograms for all/selected tracks 
    TH1D *fitBxHist_fitted, *fitBxHist_selected;                              ///< fit bx histograms for all/selected tracks 
    TH1D *fitByHist_fitted, *fitByHist_selected;                              ///< fit by histograms for all/selected tracks 
    
    /// plots ax vs. ay
    TGraph *fitAxVsAyGraph_fitted, *fitAxVsAyGraph_selected;
    
    /// plots bx vs. by
    TGraph *fitBxVsByGraph_fitted, *fitBxVsByGraph_selected;

    struct ChiSqHistograms {
      /// chi^2 histograms for all/selected tracks, in linear/logarithmic scale
      TH1D *lin_fitted, *lin_selected, *log_fitted, *log_selected;
      ChiSqHistograms() : lin_fitted(NULL), lin_selected(NULL), 
        log_fitted(NULL), log_selected(NULL) {}
      ChiSqHistograms(const std::string &name);
    };

    /// global (all RP sets) chi^2 histograms
    ChiSqHistograms chiSqHists;

    /// chi^2 histograms per RP set
    std::map< std::set<unsigned int>, ChiSqHistograms > chiSqHists_perRP;

    /// map: detector id --> residua histogram
    struct ResiduaHistogramSet {
      TH1D *total_fitted, *total_selected;
      TGraph *selected_vs_chiSq;
      std::map< std::set<unsigned int>, TH1D* > perRPSet_fitted, perRPSet_selected;
    };

    /// residua histograms
    std::map<unsigned int, ResiduaHistogramSet> residuaHistograms;

    // ----------- methods ------------

    /// creates a new residua histogram
    TH1D* NewResiduaHist(const char *name);
    
    /// fits the collection of hits and removes hits with too high residual/sigma ratio
    /// \param failed whether the fit has failed
    /// \param selectionChanged whether some hits have been removed
    void FitLocalTrack(HitCollection&, LocalTrackFit&, bool &failed, bool &selectionChanged);
    
    /// removes the hits of pots with too few planes active
    void RemoveInsufficientPots(HitCollection&, bool &selectionChanged);
    
    /// builds a standard (homogeneous or fixed detectors) set of constraints
    void BuildStandardConstraints(std::vector<AlignmentConstraint>&);

    /// fills diagnostic (chi^2, residua, ...) histograms
    void UpdateDiagnosticHistograms(const HitCollection &selection,
      const std::set<unsigned int> &selectedRPs, const LocalTrackFit &trackFit, bool trackSelected);

    /// converts a set to string
    static std::string SetToString(const std::set<unsigned int> &);

    /// result pretty printing routines
    void PrintN(const char *str, unsigned int N);
    void PrintLineSeparator(const std::vector<RPAlignmentCorrections> &);
    void PrintQuantitiesLine(const std::vector<RPAlignmentCorrections> &);
    void PrintAlgorithmsLine(const std::vector<RPAlignmentCorrections> &);

    /// saves a ROOT file with diagnostic plots
    void SaveDiagnostics() const;
};

#endif

