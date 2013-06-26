#ifndef PEDELABELER_H
#define PEDELABELER_H

/**
 * \class PedeLabeler
 *
 * provides labels for AlignmentParameters for pede
 *
 * \author    : Gero Flucke
 * date       : September 2007
 * $Date: 2012/08/10 09:01:11 $
 * $Revision: 1.6 $
 * (last update by $Author: flucke $)
 */

#include <vector>
#include <map> 

#include <Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

/***************************************
****************************************/
class PedeLabeler : public PedeLabelerBase
{
 public:

  /// constructor from three Alignables (null pointers allowed )
  PedeLabeler(const PedeLabelerBase::TopLevelAlignables &alignables,
	      const edm::ParameterSet &config);
  /// destructor
  ~PedeLabeler();
  
  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  unsigned int alignableLabel(Alignable *alignable) const;
  unsigned int alignableLabelFromParamAndInstance(Alignable *alignable,
						  unsigned int param,
						  unsigned int instance) const;
  unsigned int lasBeamLabel(unsigned int lasBeamId) const;
  unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const;
  unsigned int parameterLabel(Alignable *alignable, unsigned int parNum,
			      const AlignmentAlgorithmBase::EventInfo &eventInfo,
			      const TrajectoryStateOnSurface &tsos) const {
    return parameterLabel(alignableLabel(alignable), parNum);
  }
  bool hasSplitParameters(Alignable *alignable) const { return false; }
  unsigned int numberOfParameterInstances(Alignable *alignable,
					  int param=-1) const { return 1; }
  unsigned int maxNumberOfParameterInstances() const { return 1; }

  /// parameter number, 0 <= .. < theMaxNumParam, belonging to unique parameter label
  unsigned int paramNumFromLabel(unsigned int paramLabel) const;
  /// alignable label from parameter label (works also for alignable label...)
  unsigned int alignableLabelFromLabel(unsigned int label) const;
  /// Alignable from alignable or parameter label,
  /// null if no alignable (but error only if not las beam, either!)
  Alignable* alignableFromLabel(unsigned int label) const;
  /// las beam id from las beam or parameter label
  /// zero and error if not a valid las beam label
  unsigned int lasBeamIdFromLabel(unsigned int label) const;
  
 private:
  typedef std::map <Alignable*, unsigned int> AlignableToIdMap;
  typedef AlignableToIdMap::value_type AlignableToIdPair;
  typedef std::map <unsigned int, Alignable*> IdToAlignableMap;
  typedef std::map <unsigned int, unsigned int> UintUintMap;

  /// returns size of map
  unsigned int buildMap(const std::vector<Alignable*> &alis);
  /// returns size of map
  unsigned int buildReverseMap();

  // data members
  AlignableToIdMap  theAlignableToIdMap; /// providing unique ID for alignable, space for param IDs
  IdToAlignableMap  theIdToAlignableMap; /// reverse map
  UintUintMap       theLasBeamToLabelMap;  /// labels for las beams
  UintUintMap       theLabelToLasBeamMap; /// reverse of the above
};

#endif
