#ifndef Alignment_TrackerAlignment_TrackerScenarioBuilder_h
#define Alignment_TrackerAlignment_TrackerScenarioBuilder_h

/// \class TrackerScenarioBuilder
///
/// $Date: 2007/01/12 09:47:43 $
/// $Revision: 1.1 $
///
/// $Author: fronga $
/// \author Frederic Ronga - CERN-PH-CMG

#include <vector>

#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

/// Builds a scenario from configuration and applies it to the alignable tracker.

class TrackerScenarioBuilder : public MisalignmentScenarioBuilder
{

public:
 
  /// Constructor
  explicit TrackerScenarioBuilder( Alignable* alignable );

  /// Destructor
  ~TrackerScenarioBuilder() {};

  /// Apply misalignment scenario to the tracker
  void applyScenario( const edm::ParameterSet& scenario );

private: // Members

  AlignableTracker* theAlignableTracker;   ///< Pointer to mother alignable object

};



#endif
