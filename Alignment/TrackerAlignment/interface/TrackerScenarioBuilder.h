#ifndef Alignment_TrackerAlignment_TrackerScenarioBuilder_h
#define Alignment_TrackerAlignment_TrackerScenarioBuilder_h

/// \class TrackerScenarioBuilder
///
/// $Date: 2009/03/15 17:39:58 $
/// $Revision: 1.3 $
///
/// $Author: flucke $
/// \author Frederic Ronga - CERN-PH-CMG
///
/// Builds a scenario from configuration and applies it to the alignable tracker.

#include <vector>
#include <string>

#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"

class AlignableTracker;

/// Builds a scenario from configuration and applies it to the alignable tracker.

class TrackerScenarioBuilder : public MisalignmentScenarioBuilder
{

public:
 
  /// Constructor
  explicit TrackerScenarioBuilder( AlignableTracker* alignable );

  /// Destructor
  ~TrackerScenarioBuilder() {};

  /// Apply misalignment scenario to the tracker
  void applyScenario( const edm::ParameterSet& scenario );
  /// does this still make sense?
  virtual bool isTopLevel_(const std::string& parameterSetName) const;
  /// True if hierarchy level 'sub' could be part of hierarchy level 'large'.
  virtual bool possiblyPartOf(const std::string &sub, const std::string &large) const;

private: // Members

  AlignableTracker* theAlignableTracker;   ///< Pointer to mother alignable object
  /// following things are needed in possiblyPartOf:
  std::vector<std::string> theSubdets; ///< sub-detector acronyms appearing in StructureType.h (TPE)
  unsigned int theFirstStripIndex;     ///< index of first strip subdet in 'theSubdets' (pixel<strip)  

};


#endif
