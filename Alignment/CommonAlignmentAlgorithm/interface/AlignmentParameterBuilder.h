#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterBuilder_h

/** \class AlignmentParameterBuilder
 *
 *  Build Alignment Parameter Structure 
 *
 *  $Date: 2007/10/08 14:38:15 $
 *  $Revision: 1.8 $
 *  (last update by $Author: cklae $)
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"

namespace edm {
  class ParameterSet;
}
class AlignableTracker;
class AlignableMuon;
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
  /// master initialisation method, PSet must have form as constructor wants it 
  void addAllSelections(const edm::ParameterSet &pSet);

  /// Add selections of Alignables, using AlignmenParameterSelector::addSelections.
  /// For each Alignable, AlignmentParameters of type parType will be attached
  /// using the selection of active parameters done in AlignmenParameterSelector,
  /// e.g. for RigidBody a selection string '11100' selects the degrees of freedom in
  /// (x,y,z), but not (alpha,beta,gamma).
  /// Returns number of added selections 
  unsigned int addSelections(const edm::ParameterSet &pset,
			     AlignmentParametersFactory::ParametersType parType);

  /// Add arbitrary selection of Alignables, return number of higher level Alignables
  unsigned int add(const align::Alignables &alignables,
		   AlignmentParametersFactory::ParametersType parType,
		   const std::vector<bool> &sel);
  /// Add a single Alignable, true if it is higher level, false if it is an AlignableDet 
  bool add(Alignable *alignable, AlignmentParametersFactory::ParametersType parType,
	   const std::vector<bool> &sel);

  /// Get list of alignables for which AlignmentParameters are built 
  const align::Alignables& alignables() const { return theAlignables; };

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
  align::Alignables theAlignables;

  /// Alignable tracker   
  AlignableTracker* theAlignableTracker;

  /// Alignable muon
  AlignableMuon* theAlignableMuon;

};

#endif
