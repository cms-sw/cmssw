#ifndef PEDELABELER_H
#define PEDELABELER_H

/**
 * \class PedeLabeler
 *
 * provides labels for AlignmentParameters for pede
 *
 * \author    : Gero Flucke
 * date       : September 2007
 * $Date: 2011/02/16 12:52:46 $
 * $Revision: 1.5 $
 * (last update by $Author: mussgill $)
 */

#include <vector>
#include <map>

#include <Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h>

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/***************************************
****************************************/
class PedeLabeler : public PedeLabelerBase {
public:
  /// constructor from three Alignables (null pointers allowed )
  PedeLabeler(const PedeLabelerBase::TopLevelAlignables &alignables, const edm::ParameterSet &config);
  /// destructor
  ~PedeLabeler() override;

  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  unsigned int alignableLabel(const Alignable *alignable) const override;
  unsigned int alignableLabelFromParamAndInstance(const Alignable *alignable,
                                                  unsigned int param,
                                                  unsigned int instance) const override;
  unsigned int lasBeamLabel(unsigned int lasBeamId) const override;
  unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const override;
  unsigned int parameterLabel(Alignable *alignable,
                              unsigned int parNum,
                              const AlignmentAlgorithmBase::EventInfo &eventInfo,
                              const TrajectoryStateOnSurface &tsos) const override {
    return parameterLabel(alignableLabel(alignable), parNum);
  }
  bool hasSplitParameters(Alignable *alignable) const override { return false; }
  unsigned int numberOfParameterInstances(Alignable *alignable, int param = -1) const override { return 1; }
  unsigned int maxNumberOfParameterInstances() const override { return 1; }

  /// parameter number, 0 <= .. < theMaxNumParam, belonging to unique parameter label
  unsigned int paramNumFromLabel(unsigned int paramLabel) const override;
  /// alignable label from parameter label (works also for alignable label...)
  unsigned int alignableLabelFromLabel(unsigned int label) const override;
  /// Alignable from alignable or parameter label,
  /// null if no alignable (but error only if not las beam, either!)
  Alignable *alignableFromLabel(unsigned int label) const override;
  /// las beam id from las beam or parameter label
  /// zero and error if not a valid las beam label
  unsigned int lasBeamIdFromLabel(unsigned int label) const override;

private:
  typedef std::map<Alignable *, unsigned int, AlignableComparator> AlignableToIdMap;
  typedef AlignableToIdMap::value_type AlignableToIdPair;
  typedef std::map<unsigned int, Alignable *> IdToAlignableMap;
  typedef std::map<unsigned int, unsigned int> UintUintMap;

  /// returns size of map
  unsigned int buildMap(const align::Alignables &);
  /// returns size of map
  unsigned int buildReverseMap();

  // data members
  AlignableToIdMap theAlignableToIdMap;  /// providing unique ID for alignable, space for param IDs
  IdToAlignableMap theIdToAlignableMap;  /// reverse map
  UintUintMap theLasBeamToLabelMap;      /// labels for las beams
  UintUintMap theLabelToLasBeamMap;      /// reverse of the above
};

#endif
