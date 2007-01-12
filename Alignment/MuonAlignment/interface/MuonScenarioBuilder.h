#ifndef Alignment_MuonAlignment_MuonScenarioBuilder_h
#define Alignment_MuonAlignment_MuonScenarioBuilder_h

/** \class MuonScenarioBuilder
 *  The misalignment scenario builder.
 *
 *  $Date: 2006/08/04 20:18:50 $
 *  $Revision: 1.4 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableModifier.h"
#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

/// Builds a scenario from configuration and applies it to the alignable Muon.

class MuonScenarioBuilder : public MisalignmentScenarioBuilder
{

public:

  /// Constructor
  explicit MuonScenarioBuilder( Alignable* alignable );

  /// Destructor
  ~MuonScenarioBuilder() {};

  /// Apply misalignment scenario to the Muon
  void applyScenario( const edm::ParameterSet& scenario );

private: // Members

  AlignableMuon* theAlignableMuon;   ///< Pointer to alignable Muon object

};



#endif
