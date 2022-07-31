#ifndef Alignment_MillePedeAlignmentAlgorithm_PedeLabelerBase_h
#define Alignment_MillePedeAlignmentAlgorithm_PedeLabelerBase_h

/** \class PedeLabelerBase
 *
 * Baseclass for pede labelers
 *
 *  Original author: Andreas Mussgiller, January 2011
 *
 *  $Date: 2011/02/18 17:08:13 $
 *  $Revision: 1.2 $
 *  (last update by $Author: mussgill $)
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CondFormats/Common/interface/Time.h"

#include <vector>

class Alignable;
class AlignableTracker;
class AlignableMuon;
class AlignableExtras;
class IntegratedCalibrationBase;

/***************************************
****************************************/
class PedeLabelerBase {
public:
  using RunNumber = align::RunNumber;
  using RunRange = align::RunRange;
  using RunRanges = align::RunRanges;

  class TopLevelAlignables {
  public:
    TopLevelAlignables(AlignableTracker *aliTracker, AlignableMuon *aliMuon, AlignableExtras *extras)
        : aliTracker_(aliTracker), aliMuon_(aliMuon), aliExtras_(extras) {}
    AlignableTracker *aliTracker_;
    AlignableMuon *aliMuon_;
    AlignableExtras *aliExtras_;
  };

  /// constructor from three Alignables (null pointers allowed )
  PedeLabelerBase(const TopLevelAlignables &alignables, const edm::ParameterSet &config);
  virtual ~PedeLabelerBase() {}
  /// tell labeler to treat also integrated calibrations
  virtual void addCalibrations(const std::vector<IntegratedCalibrationBase *> &iCals);

  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  virtual unsigned int alignableLabel(const Alignable *alignable) const = 0;
  /// uniqueId of Alignable for a given parameter index and instance,
  /// 0 if alignable not known between this ID and the next there is enough
  /// 'space' to add parameter numbers 0...nPar-1 to make unique IDs for the
  /// labels of active parameters
  virtual unsigned int alignableLabelFromParamAndInstance(const Alignable *alignable,
                                                          unsigned int param,
                                                          unsigned int instance) const = 0;
  virtual unsigned int lasBeamLabel(unsigned int lasBeamId) const = 0;
  /// returns the label for a given alignable parameter number combination
  virtual unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const = 0;
  /// returns the label for a given alignable parameter number combination
  /// in case the parameters are split into various instances
  virtual unsigned int parameterLabel(Alignable *alignable,
                                      unsigned int parNum,
                                      const AlignmentAlgorithmBase::EventInfo &eventInfo,
                                      const TrajectoryStateOnSurface &tsos) const = 0;
  /// returns true if the alignable has parameters that are split into various bins
  virtual bool hasSplitParameters(Alignable *alignable) const = 0;
  /// returns the number of instances for a given parameter
  virtual unsigned int numberOfParameterInstances(Alignable *alignable, int param = -1) const = 0;
  /// returns the maximum number of instances for any parameter of an Alignable*
  virtual unsigned int maxNumberOfParameterInstances() const = 0;
  /// offset in labels between consecutive parameter instances of Alignable*s
  unsigned int parameterInstanceOffset() const { return theParamInstanceOffset; }

  /// parameter number, 0 <= .. < theMaxNumParam, belonging to unique parameter label
  virtual unsigned int paramNumFromLabel(unsigned int paramLabel) const = 0;
  /// alignable label from parameter label (works also for alignable label...)
  virtual unsigned int alignableLabelFromLabel(unsigned int label) const = 0;
  /// Alignable from alignable or parameter label,
  /// null if no alignable (but error only if not las beam, either!)
  virtual Alignable *alignableFromLabel(unsigned int label) const = 0;
  /// las beam id from las beam or parameter label
  /// zero and error if not a valid las beam label
  virtual unsigned int lasBeamIdFromLabel(unsigned int label) const = 0;
  /// calibration and its parameter number from label,
  /// if label does not belong to any calibration return nullptr as pair.first
  virtual std::pair<IntegratedCalibrationBase *, unsigned int> calibrationParamFromLabel(unsigned int label) const;

  virtual const RunRange &runRangeFromLabel(unsigned int label) const { return theOpenRunRange; }
  /// first free label not yet used (for hacks within millepede...)
  /// valid only after last call to addCalibrations(..)
  virtual unsigned int firstFreeLabel() const;

  /// label for parameter 'paramNum' (counted from 0) of an integrated calibration
  virtual unsigned int calibrationLabel(const IntegratedCalibrationBase *calib, unsigned int paramNum) const;
  const AlignableTracker *alignableTracker() const { return topLevelAlignables_.aliTracker_; }
  const AlignableMuon *alignableMuon() const { return topLevelAlignables_.aliMuon_; }
  const AlignableExtras *alignableExtras() const { return topLevelAlignables_.aliExtras_; }

  static const unsigned int theMaxNumParam;
  static const unsigned int theParamInstanceOffset;
  static const unsigned int theMinLabel;

protected:
  /// first free label after everything about Alignables and LAS beams
  /// (to be used for calibrations)
  virtual unsigned int firstNonAlignableLabel() const;

  /// Return tracker alignable object ID provider derived from the tracker's geometry
  const AlignableObjectId &objectIdProvider() const { return alignableObjectId_; }

  const RunRange theOpenRunRange;

private:
  const TopLevelAlignables topLevelAlignables_;
  const AlignableObjectId alignableObjectId_;

  /// pairs of calibrations and their first label
  std::vector<std::pair<IntegratedCalibrationBase *, unsigned int> > theCalibrationLabels;
};

struct AlignableComparator {
  using is_transparent = void;  // needs to be defined, actual type is not relevant
  bool operator()(Alignable *a, Alignable *b) const { return a < b; }
  bool operator()(Alignable *a, const Alignable *b) const { return a < b; }
  bool operator()(const Alignable *a, Alignable *b) const { return a < b; }
};

#endif
