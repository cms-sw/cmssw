#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h

#include <string>
#include <vector>

/** \class AlignmentParameterBuilder
 *
 *  Build Alignment Parameter Structure 
 *
 *  $Date: 2006/11/30 10:08:26 $
 *  $Revision: 1.6 $
 *  (last update by $Author: flucke $)
 */

namespace edm {
  class ParameterSet;
}
class AlignableTracker;
class AlignableMuon;
class Alignable;
class AlignmentParameters;

class AlignmentParameterBuilder 
{
public:

  /// Constructor from tracker only
  explicit AlignmentParameterBuilder( AlignableTracker *alignableTracker );

  /// Constructor from tracker and muon
  AlignmentParameterBuilder( AlignableTracker *alignableTracker, AlignableMuon *alignableMuon );

  /// Constructor adding selections by passing the ParameterSet named 'AlignmentParameterSelector'
  /// (expected in pSet) to addSelections(..)
  AlignmentParameterBuilder( AlignableTracker *alignableTracker, const edm::ParameterSet &pSet );

  /// Constructor from tracker and muon, plus selection
  AlignmentParameterBuilder( AlignableTracker *alignableTracker, AlignableMuon *alignableMuon, 
                             const edm::ParameterSet &pSet);


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

  /// convert char selection (from ParameterSelector) to bool (for AlignmentParameters)
  /// true if anything else than 0 and 1 is found in vector<char>
  bool decodeParamSel(const std::vector<char> &paramSelChar, std::vector<bool> &result) const;
  /// add SelectionUserVariables corresponding to fullSel 
  bool addFullParamSel(AlignmentParameters *aliPar, const std::vector<char> &fullSel) const;

  // data members

  /// Vector of alignables 
  std::vector<Alignable*> theAlignables;

  /// Alignable tracker   
  AlignableTracker* theAlignableTracker;

  /// Alignable muon
  AlignableMuon* theAlignableMuon;

};

#endif
