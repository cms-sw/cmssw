#ifndef Alignment_TrackerAlignment_MisalignmentScenarioBuilder_h
#define Alignment_TrackerAlignment_MisalignmentScenarioBuilder_h


#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerModifier.h"

/// Builds a scenario from configuration and applies it to the alignable tracker.

class MisalignmentScenarioBuilder
{

public:

  /// Constructor
  MisalignmentScenarioBuilder( const edm::ParameterSet& scenario,
							   AlignableTracker* tracker );

  /// Destructor
  ~MisalignmentScenarioBuilder() {};

  /// Apply the scenario to the tracker
  void applyScenario( void );

private: // Methods

  /// Decode movements defined in given parameter set for given set of alignables
  void decodeMovements_( const edm::ParameterSet& pSet, std::vector<Alignable*> alignables );
  
  /// Decode movements defined in given parameter set for given set of alignables tagged by given name
  void decodeMovements_( const edm::ParameterSet& pSet, std::vector<Alignable*> alignables,
						 std::string levelName );

  /// Apply movements given by parameter set to given alignable
  void applyMovements_( Alignable* alignable, const edm::ParameterSet& pSet );
  
  /// Merge two sets of parameters into one (the first argument)
  void mergeParameters_( edm::ParameterSet& localSet, const edm::ParameterSet& globalSet ) const;

  /// Propagate global parameters to sub-parameters
  void propagateParameters_( const edm::ParameterSet& pSet, const std::string& globalName,
							 edm::ParameterSet& subSet ) const;

  /// Get parameter set corresponding to given name (returns empty parameter set if does not exist)
  edm::ParameterSet getParameterSet_( const edm::ParameterSet& pSet, const std::string& name ) const;

  /// Print all parameters and values for given set
  void printParameters_( const edm::ParameterSet& pSet, const bool showPsets = false ) const;

  /// Check if given parameter is for a top-level structure
  const bool isTopLevel_( const std::string& parameterSetName ) const; 
  

private: // Members

  AlignableTracker* theTracker;                 ///< Pointer to alignable tracker object
  edm::ParameterSet theScenario;                ///< Misalignment scenario to apply (from config file)
  AlignableTrackerModifier theTrackerModifier;  ///< Helper class for random movements

  std::string indent;                           ///< Depth in hierarchy
  

};



#endif
