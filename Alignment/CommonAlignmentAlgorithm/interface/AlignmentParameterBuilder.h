#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h

#include <string>
#include <vector>

/** \class AlignmentParameterBuilder
 *
 *  Build Alignment Parameter Structure 
 *
 *  $Date: 2006/11/03 16:28:54 $
 *  $Revision: 1.4 $
 *  (last update by $Author: flucke $)
 */

namespace edm {
  class ParameterSet;
}
class Alignabletracker;

class AlignmentParameterBuilder 
{
public:

  /// Constructor
  explicit AlignmentParameterBuilder(AlignableTracker *alignableTracker);
  /// Constructor adding selections by passing the ParameterSet named 'AlignmentParameterSelector'
  /// (expected in pSet) to addSelections(..)
  AlignmentParameterBuilder(AlignableTracker *alignableTracker, const edm::ParameterSet &pSet);

  /// destructor 
  virtual ~AlignmentParameterBuilder() {};

  /// Add selections of Alignables, using AlignmenParameterSelector::addSelections.
  /// For each Alignable, (Composite)RigidBodyAlignmentParameters will be attached
  /// using the selection of active parameters done in AlignmenParameterSelector,
  /// e.g. a selection string '11100' selects the degrees of freedom in (x,y,z), 
  /// but not (alpha,beta,gamma).
  /// Returns number of added selections 
  unsigned int addSelections(const edm::ParameterSet &pset);

  /// Add arbitrary selection of Alignables, return number of higher level Alignables
  unsigned int add(const std::vector<Alignable*> &alignables, const std::vector<bool> &sel);
  /// Add a single Alignable, true if it is higher level, false if it is an AlignableDet 
  bool add(Alignable *alignable, const std::vector<bool> &sel);

  /// Get list of alignables for which AlignmentParameters are built 
  std::vector<Alignable*> alignables() { return theAlignables; };

  /// Remove n Alignables from list 
  void fixAlignables( int n );

private:

  // data members

  /// Vector of alignables 
  std::vector<Alignable*> theAlignables;

  /// Alignable tracker   
  AlignableTracker* theAlignableTracker;

};

#endif
