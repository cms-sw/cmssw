/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Alignment_RPTrackBased_JanAlignmentAlgorithm
#define Alignment_RPTrackBased_JanAlignmentAlgorithm


#include "Alignment/RPTrackBased/interface/AlignmentAlgorithm.h"

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
class JanAlignmentAlgorithm : public AlignmentAlgorithm
{
  public:
    /// a scatter plot, with graph and histogram representations
    struct ScatterPlot {
      TGraph *g;
      TH2D* h;
    };

    /// structure holding statistical information for one detector
    struct DetStat {
      TH1D *m_dist;
      std::vector<TH1D*> coefHist;  
      std::vector<TGraph*> resVsCoef;  
      std::map< std::set<unsigned int>, ScatterPlot> resVsCoefRot_perRPSet;
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
    JanAlignmentAlgorithm(const edm::ParameterSet& ps, AlignmentTask *_t);

    ~JanAlignmentAlgorithm();

    virtual std::string GetName()
      { return "Jan"; }

    virtual bool HasErrorEstimate()
      { return true; }

    virtual void Begin(const edm::EventSetup&);
    virtual void Feed(const HitCollection&, const LocalTrackFit&, const LocalTrackFit&);
    virtual void SaveDiagnostics(TDirectory *);
    virtual std::vector<SingularMode> Analyze();
    virtual unsigned int Solve(const std::vector<AlignmentConstraint>&,
      RPAlignmentCorrections &result, TDirectory *dir);
    virtual void End();
};

#endif

