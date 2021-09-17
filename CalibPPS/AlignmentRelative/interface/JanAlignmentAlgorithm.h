/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_JanAlignmentAlgorithm_h
#define CalibPPS_AlignmentRelative_JanAlignmentAlgorithm_h

#include "CalibPPS/AlignmentRelative/interface/AlignmentAlgorithm.h"
#include "CalibPPS/AlignmentRelative/interface/SingularMode.h"

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"

#include <vector>
#include <map>
#include <string>

/**
 *\brief Jan's alignment algorithm.
 **/
class JanAlignmentAlgorithm : public AlignmentAlgorithm {
public:
  /// a scatter plot, with graph and histogram representations
  struct ScatterPlot {
    TGraph *g;
    TH2D *h;
  };

  /// structure holding statistical information for one detector
  struct DetStat {
    TH1D *m_dist;
    TH1D *R_dist;
    std::vector<TH1D *> coefHist;
    std::vector<TGraph *> resVsCoef;
    std::map<std::set<unsigned int>, ScatterPlot> resVsCoefRot_perRPSet;
  };

private:
  /// S matrix components
  /// indeces correspond to the qClToOpt list
  TMatrixD **Sc;

  /// M vector components
  /// indeces correspond to the qClToOpt list
  TVectorD *Mc;

  /// final S matrix
  TMatrixD S;

  /// final M vector
  TVectorD M;

  /// eigen values of the S matrix
  TVectorD S_eigVal;

  /// matrix of S eigenvectors
  TMatrixD S_eigVec;

  /// a list of the singular modes of the S matrix
  std::vector<SingularMode> singularModes;

  /// whether to stop when singular modes are identified
  bool stopOnSingularModes;

  /// normalized eigen value below which the (CS) eigen vectors are considered as weak
  double weakLimit;

  /// event count
  unsigned int events;

  /// statistical data collection
  std::map<unsigned int, DetStat> statistics;

  /// flag whether to build statistical plots
  bool buildDiagnosticPlots;

public:
  /// dummy constructor (not to be used)
  JanAlignmentAlgorithm() {}

  /// normal constructor
  JanAlignmentAlgorithm(const edm::ParameterSet &ps, AlignmentTask *_t);

  ~JanAlignmentAlgorithm() override;

  std::string getName() override { return "Jan"; }

  bool hasErrorEstimate() override { return true; }

  void begin(const CTPPSGeometry *geometryReal, const CTPPSGeometry *geometryMisaligned) override;

  void feed(const HitCollection &, const LocalTrackFit &) override;

  void saveDiagnostics(TDirectory *) override;

  void analyze() override;

  unsigned int solve(const std::vector<AlignmentConstraint> &,
                     std::map<unsigned int, AlignmentResult> &results,
                     TDirectory *dir) override;

  void end() override;
};

#endif
