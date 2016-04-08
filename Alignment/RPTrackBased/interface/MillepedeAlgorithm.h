/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Alignment_RPTrackBased_MillepedeAlgorithm
#define Alignment_RPTrackBased_MillepedeAlgorithm


#include "Alignment/RPTrackBased/interface/AlignmentAlgorithm.h"
#include "Alignment/RPTrackBased/interface/Mille.h"

#include "TMatrixD.h"
#include "TVectorD.h"

#include <vector>

/**
 *\brief Alignment algorithm using Millepede.
 **/
class MillepedeAlgorithm : public AlignmentAlgorithm
{
  private:
    std::string workingDir;
    Mille *mille;

  public:
    /// dummy constructor (not to be used)
    MillepedeAlgorithm() {}
    
    /// normal constructor
    MillepedeAlgorithm(const edm::ParameterSet& ps, AlignmentTask *_t);
    
    ~MillepedeAlgorithm();

    virtual std::string GetName()
      { return "Millepede"; }

    virtual bool HasErrorEstimate()
      { return false; }

    virtual void Begin(const edm::EventSetup&);
    virtual void Feed(const HitCollection&, const LocalTrackFit&, const LocalTrackFit&);
    virtual void SaveDiagnostics(TDirectory *) {}
    virtual std::vector<SingularMode> Analyze();
    virtual unsigned int Solve(const std::vector<AlignmentConstraint>&,
      RPAlignmentCorrections &result, TDirectory *dir);
    virtual void End();
};

#endif

