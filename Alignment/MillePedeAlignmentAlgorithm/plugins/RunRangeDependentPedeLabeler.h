#ifndef RUNRANGEDEPENDENTPEDELABELER_H
#define RUNRANGEDEPENDENTPEDELABELER_H

/**
 * \class RunRangeDependentPedeLabeler
 *
 * provides labels for AlignmentParameters for pede
 *
 * \author    : Gero Flucke
 * date       : September 2007
 * $Date: 2011/02/16 13:12:41 $
 * $Revision: 1.1 $
 * (last update by $Author: mussgill $)
 */

#include <vector>
#include <map>

#include <Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h>

#include "CondFormats/Common/interface/Time.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class EventID;
}

/***************************************
****************************************/
class RunRangeDependentPedeLabeler : public PedeLabelerBase {
public:
  /// constructor from three Alignables (null pointers allowed )
  RunRangeDependentPedeLabeler(const PedeLabelerBase::TopLevelAlignables &alignables, const edm::ParameterSet &config);
  /** non-virtual destructor: do not inherit from this class **/
  ~RunRangeDependentPedeLabeler() override;

  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  unsigned int alignableLabel(const Alignable *alignable) const override;
  /// uniqueId of Alignable for a given parameter index and instance,
  /// 0 if alignable not known between this ID and the next there is enough
  /// 'space' to add parameter numbers 0...nPar-1 to make unique IDs for the
  /// labels of active parameters
  unsigned int alignableLabelFromParamAndInstance(const Alignable *alignable,
                                                  unsigned int param,
                                                  unsigned int instance) const override;
  unsigned int lasBeamLabel(unsigned int lasBeamId) const override;
  /// returns the label for a given alignable parameter number combination
  unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const override;
  /// returns the label for a given alignable parameter number combination
  /// in case the parameters are split into v
  unsigned int parameterLabel(Alignable *alignable,
                              unsigned int parNum,
                              const AlignmentAlgorithmBase::EventInfo &eventInfo,
                              const TrajectoryStateOnSurface &tsos) const override;
  /// returns true if the alignable has parameters that are split into various bins
  bool hasSplitParameters(Alignable *alignable) const override;
  /// returns the number of instances for a given parameter
  unsigned int numberOfParameterInstances(Alignable *alignable, int param = -1) const override;
  unsigned int maxNumberOfParameterInstances() const override { return theMaxNumberOfParameterInstances; }

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
  const RunRange &runRangeFromLabel(unsigned int label) const override;

private:
  typedef std::map<Alignable *, unsigned int, AlignableComparator> AlignableToIdMap;
  typedef AlignableToIdMap::value_type AlignableToIdPair;
  typedef std::vector<RunRange> RunRangeVector;
  typedef std::map<unsigned int, RunRangeVector> RunRangeParamMap;
  typedef std::map<Alignable *, RunRangeParamMap, AlignableComparator> AlignableToRunRangeRangeMap;
  typedef AlignableToRunRangeRangeMap::value_type AlignableToRunRangeRangePair;
  typedef std::map<unsigned int, Alignable *> IdToAlignableMap;
  typedef std::map<unsigned int, unsigned int> UintUintMap;

  unsigned int runRangeIndexFromLabel(unsigned int label) const;

  std::vector<std::string> decompose(const std::string &s, std::string::value_type delimiter) const;
  std::vector<unsigned int> convertParamSel(const std::string &selString) const;
  unsigned int buildRunRangeDependencyMap(AlignableTracker *aliTracker,
                                          AlignableMuon *aliMuon,
                                          AlignableExtras *extras,
                                          const edm::ParameterSet &config);

  /// returns size of map
  unsigned int buildMap(const align::Alignables &);
  /// returns size of map
  unsigned int buildReverseMap();

  // data members
  AlignableToIdMap theAlignableToIdMap;  /// providing unique ID for alignable, space for param IDs
  AlignableToRunRangeRangeMap theAlignableToRunRangeRangeMap;  /// providing unique ID for alignable, space for param IDs
  IdToAlignableMap theIdToAlignableMap;                        /// reverse map
  UintUintMap theLasBeamToLabelMap;                            /// labels for las beams
  UintUintMap theLabelToLasBeamMap;                            /// reverse of the above
  unsigned int theMaxNumberOfParameterInstances;
};

#endif
