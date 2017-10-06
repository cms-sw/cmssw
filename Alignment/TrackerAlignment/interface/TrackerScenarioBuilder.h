#ifndef Alignment_TrackerAlignment_TrackerScenarioBuilder_h
#define Alignment_TrackerAlignment_TrackerScenarioBuilder_h

/// \class TrackerScenarioBuilder
///
/// $Date: 2007/10/18 09:57:11 $
/// $Revision: 1.2 $
///
/// $Author: fronga $
/// \author Frederic Ronga - CERN-PH-CMG
///
/// Builds a scenario from configuration and applies it to the alignable tracker.

#include <vector>
#include <string>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"

class AlignableTracker;

/// Builds a scenario from configuration and applies it to the alignable tracker.

class TrackerScenarioBuilder : public MisalignmentScenarioBuilder
{

public:
 
  /// Constructor
  explicit TrackerScenarioBuilder( AlignableTracker* alignable );

  /// Destructor
  ~TrackerScenarioBuilder() override {};

  /// Apply misalignment scenario to the tracker
  void applyScenario( const edm::ParameterSet& scenario ) override;
  /// does this still make sense?
  bool isTopLevel_(const std::string& parameterSetName) const override;
  /// True if hierarchy level 'sub' could be part of hierarchy level 'large'.
  bool possiblyPartOf(const std::string &sub, const std::string &large) const override;

private:
  std::string stripOffModule(const align::StructureType& type) const;

 // Members

  AlignableTracker* theAlignableTracker;   ///< Pointer to mother alignable object
  /// following things are needed in possiblyPartOf:
  std::vector<std::string> theSubdets; ///< sub-detector acronyms appearing in StructureType.h (TPE)
  unsigned int theFirstStripIndex;     ///< index of first strip subdet in 'theSubdets' (pixel<strip)  

};


#endif
