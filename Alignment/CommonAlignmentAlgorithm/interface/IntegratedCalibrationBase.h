#ifndef Alignment_CommonAlignmentAlgorithm_IntegratedCalibrationBase_h
#define Alignment_CommonAlignmentAlgorithm_IntegratedCalibrationBase_h

/**
 * \file IntegratedCalibrationBase.cc
 *
 *  \author Gero Flucke
 *  \date August 2012
 *  $Revision: 1.2.2.1 $
 *  $Date: 2013/04/23 08:13:27 $
 *  (last update by $Author: jbehr $)
 *
 *  Base class for the calibrations that are integrated
 *  into the alignment algorithms.
 *  Note that not all algorithms support this...
 *  Limitations:
 *   o Hits are assumed to be (up to) 2D.
 *   o Derivatives depend only on local things (hit and track TSOS),
 *     EventSetup and AlignmentAlgorithmBase::EventInfo.
 */


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include <vector>
#include <utility>
#include <string>

class AlignableTracker;
class AlignableMuon;
class AlignableExtras;

class TrajectoryStateOnSurface;
class TrackingRecHit;

namespace edm { class EventSetup; class ParameterSet; } 

class IntegratedCalibrationBase
{
public:

  typedef AlignmentAlgorithmBase::EventInfo EventInfo;
  typedef std::pair<double,double> Values; /// x- and y-values
  typedef std::pair<Values, unsigned int> ValuesIndexPair; /// Values and their parameter index

  /// Constructor
  explicit IntegratedCalibrationBase(const edm::ParameterSet &cfg);
  
  /// Destructor
  virtual ~IntegratedCalibrationBase() {};

  /// How many parameters does this calibration define?
  virtual unsigned int numParameters() const = 0;

  /// Return all derivatives for x- (Values.first) and y-measurement (Values.second),
  /// default implementation uses other derivatives(..) method,
  /// but can be overwritten in derived class for efficiency.
  virtual std::vector<Values> derivatives(const TrackingRecHit &hit,
					  const TrajectoryStateOnSurface &tsos,
					  const edm::EventSetup &setup,
					  const EventInfo &eventInfo) const;

  /// Return non-zero derivatives for x- (ValuesIndexPair.first.first)
  /// and y-measurement (ValuesIndexPair.first.second) with their
  /// indices (ValuesIndexPair.second) by reference.
  /// Return value is their number.
  virtual unsigned int derivatives(std::vector<ValuesIndexPair> &outDerivInds,
				   const TrackingRecHit &hit,
				   const TrajectoryStateOnSurface &tsos,
				   const edm::EventSetup &setup,
				   const EventInfo &eventInfo) const = 0;

  /// Setting the determined parameter identified by index,
  /// should return false if out-of-bounds, true otherwise.
  virtual bool setParameter(unsigned int index, double value) = 0;

  /// Setting the determined parameter uncertainty identified by index,
  /// should return false if out-of-bounds or errors not treated, true otherwise.
  virtual bool setParameterError(unsigned int index, double value) = 0;

  /// Return current value of parameter identified by index.
  /// Should return 0. if index out-of-bounds.
  virtual double getParameter(unsigned int index) const = 0;

  /// Return current value of parameter identified by index.
  /// Should return 0. if index out-of-bounds or if errors not treated/undetermined.
  virtual double getParameterError(unsigned int index) const = 0;

  /// Call at beginning of job:
  /// default implementation is dummy, to be overwritten in derived class if useful.
  virtual void beginOfJob(AlignableTracker *tracker,
			  AlignableMuon *muon,
			  AlignableExtras *extras) {};

  /// Called at beginning of a loop of the AlignmentProducer,
  /// to be used for iterative algorithms, default does nothing.
  /// FIXME: move call to algorithm?
  virtual void startNewLoop() {};

  /// Called at end of a loop of the AlignmentProducer,
  /// to be used for iterative algorithms, default does nothing.
  /// FIXME: move call to algorithm?
  virtual void endOfLoop() {};

  /// Called at end of a the job of the AlignmentProducer.
  /// Do here the necessary stuff with the results that should have been passed
  /// by the algorithm to the calibration, e.g. write out to database.
  /// FIXME: How to deal with single jobs for an iterative algorithm?
  virtual void endOfJob() = 0;

  /* /// called at begin of run */
  /* virtual void beginRun(const edm::EventSetup &setup) {}; */

  /* /// called at end of run - order of arguments like in EDProducer etc. */
  /* virtual void endRun(const EndRunInfo &runInfo, const edm::EventSetup &setup) {}; */

  /* /// called at begin of luminosity block (no lumi block info passed yet) */
  /* virtual void beginLuminosityBlock(const edm::EventSetup &setup) {}; */

  /* /// called at end of luminosity block (no lumi block info passed yet) */
  /* virtual void endLuminosityBlock(const edm::EventSetup &setup) {}; */

  /// name of this calibration
  const std::string& name() const { return name_;} // non-virtual since refering to private member

 private:
  const std::string name_; /// name of this calibration (i.e. defining plugin)
};

#endif
