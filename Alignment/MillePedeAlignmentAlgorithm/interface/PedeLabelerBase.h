#ifndef Alignment_MillePedeAlignmentAlgorithm_PedeLabelerBase_h
#define Alignment_MillePedeAlignmentAlgorithm_PedeLabelerBase_h

/** \class PedeLabelerBase
 *
 * Baseclass for pede labelers
 *
 *  Original author: Andreas Mussgiller, January 2011
 *
 *  $Date: 2011/02/16 12:52:46 $
 *  $Revision: 1.1 $
 *  (last update by $Author: mussgill $)
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CondFormats/Common/interface/Time.h"

class Alignable;
class AlignableTracker;
class AlignableMuon;
class AlignableExtras;

/***************************************
****************************************/
class PedeLabelerBase
{
 public:

  typedef AlignmentAlgorithmBase::RunNumber  RunNumber;
  typedef AlignmentAlgorithmBase::RunRange   RunRange;
  typedef std::vector<RunRange>              RunRanges;

  class TopLevelAlignables
  {
  public:
    TopLevelAlignables(AlignableTracker *aliTracker,
		       AlignableMuon *aliMuon,
		       AlignableExtras *extras)
      :aliTracker_(aliTracker), aliMuon_(aliMuon), aliExtras_(extras) {}
    AlignableTracker *aliTracker_;
    AlignableMuon *aliMuon_;
    AlignableExtras *aliExtras_;
  };

  /// constructor from three Alignables (null pointers allowed )
  PedeLabelerBase(const TopLevelAlignables& alignables,
		  const edm::ParameterSet & config);
  /** non-virtual destructor: do not inherit from this class **/
  virtual ~PedeLabelerBase() {}
    
  /// uniqueId of Alignable, 0 if alignable not known
  /// between this ID and the next there is enough 'space' to add parameter
  /// numbers 0...nPar-1 to make unique IDs for the labels of active parameters
  virtual unsigned int alignableLabel(Alignable *alignable) const = 0;
  /// uniqueId of Alignable for a given parameter index and instance,
  /// 0 if alignable not known between this ID and the next there is enough
  /// 'space' to add parameter numbers 0...nPar-1 to make unique IDs for the
  /// labels of active parameters
  virtual unsigned int alignableLabelFromParamAndInstance(Alignable *alignable,
							  unsigned int param,
							  unsigned int instance) const = 0;
  virtual unsigned int lasBeamLabel(unsigned int lasBeamId) const = 0;
  /// returns the label for a given alignable parameter number combination
  virtual unsigned int parameterLabel(unsigned int aliLabel, unsigned int parNum) const = 0;
  /// returns the label for a given alignable parameter number combination
  /// in case the parameters are split into various instances
  virtual unsigned int parameterLabel(Alignable *alignable, unsigned int parNum,
				      const AlignmentAlgorithmBase::EventInfo &eventInfo,
				      const TrajectoryStateOnSurface &tsos) const = 0;
  /// returns true if the alignable has parameters that are split into various bins
  virtual bool hasSplitParameters(Alignable *alignable) const = 0;
  /// returns the number of instances for a given parameter
  virtual unsigned int numberOfParameterInstances(Alignable *alignable,
						  int param=-1) const = 0;

  /// parameter number, 0 <= .. < theMaxNumParam, belonging to unique parameter label
  virtual unsigned int paramNumFromLabel(unsigned int paramLabel) const = 0;
  /// alignable label from parameter label (works also for alignable label...)
  virtual unsigned int alignableLabelFromLabel(unsigned int label) const = 0;
  /// Alignable from alignable or parameter label,
  /// null if no alignable (but error only if not las beam, either!)
  virtual Alignable* alignableFromLabel(unsigned int label) const = 0;
  /// las beam id from las beam or parameter label
  /// zero and error if not a valid las beam label
  virtual unsigned int lasBeamIdFromLabel(unsigned int label) const = 0;
  virtual const RunRange& runRangeFromLabel(unsigned int label) const {
    return theOpenRunRange;
  }
  
  static const unsigned int theMaxNumParam;
  static const unsigned int theParamInstanceOffset;
  static const unsigned int theMinLabel;

 protected:

  const RunRange theOpenRunRange;
};

#endif
