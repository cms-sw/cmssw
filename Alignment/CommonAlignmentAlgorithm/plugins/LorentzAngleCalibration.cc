/// \class LorentzAngleCalibration
///
/// Calibration of Lorentz angle (and strip deco mode backplane corrections?)
/// for the tracker, integrated in the alignment algorithms.
///
/// Note that not all algorithms support this...
///
///  \author    : Gero Flucke
///  date       : August 2012
///  $Revision: 1.35 $
///  $Date: 2011/09/06 13:46:07 $
///  (last update by $Author: mussgill $)

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include <iostream>
#include <sstream>

class LorentzAngleCalibration : public IntegratedCalibrationBase
{
public:
  /// Constructor
  explicit LorentzAngleCalibration(const edm::ParameterSet &cfg);
  
  /// Destructor
  virtual ~LorentzAngleCalibration();

  /// How many parameters does this calibration define?
  virtual unsigned int numParameters() const;

  // /// Return all derivatives,
  // /// default implementation uses other derivatives(..) method,
  // /// but can be overwritten in derived class for efficiency.
  // virtual std::vector<double> derivatives(const TransientTrackingRecHit &hit,
  // 					  const TrajectoryStateOnSurface &tsos,
  // 					  const edm::EventSetup &setup,
  // 					  const EventInfo &eventInfo) const;

  /// Return non-zero derivatives for x- and y-measurements with their indices by reference.
  /// Return value is their number.
  virtual unsigned int derivatives(std::vector<ValuesIndexPair> &outDerivInds,
				   const TransientTrackingRecHit &hit,
				   const TrajectoryStateOnSurface &tsos,
				   const edm::EventSetup &setup,
				   const EventInfo &eventInfo) const;

  /// Setting the determined parameter identified by index,
  /// should return false if out-of-bounds, true otherwise.
  virtual bool setParameter(unsigned int index, double value);

  /// Setting the determined parameter uncertainty identified by index,
  /// returns false if out-of-bounds, true otherwise.
  virtual bool setParameterError(unsigned int index, double error);

  /// Return current value of parameter identified by index.
  /// Should return 0. if index out-of-bounds.
  virtual double getParameter(unsigned int index) const;

  /// Return current value of parameter identified by index.
  /// Returns 0. if index out-of-bounds or if errors undetermined.
  virtual double getParameterError(unsigned int index) const;

  /// Call at beginning of job:
  /// default implementation is dummy, to be overwritten in derived class if useful.
//   virtual void beginOfJob(const AlignableTracker *tracker,
// 			  const AlignableMuon *muon,
// 			  const AlignableExtras *extras) {};

  /// Called at end of a the job of the AlignmentProducer.
  /// Write out determined parameters.
  virtual void endOfJob();

private:
  std::vector<double> parameters_;
  std::vector<double> paramUncertainties_;
};

//======================================================================
//======================================================================
//======================================================================

LorentzAngleCalibration::LorentzAngleCalibration(const edm::ParameterSet &cfg)
  : IntegratedCalibrationBase(cfg)
{
  // FIXME
  parameters_.resize(2, 0.); // mimic two parameters, start value 0.
  paramUncertainties_.resize(2, 0.); // dito
  edm::LogInfo("Alignment") << "@SUB=LorentzAngleCalibration" << "Created with name "
                            << this->name() << ",\n" << this->numParameters()
                            << " parameters to be determined.";
}
  
//======================================================================
LorentzAngleCalibration::~LorentzAngleCalibration()
{
  //  std::cout << "Destroy LorentzAngleCalibration named " << this->name() << std::endl;
}

//======================================================================
unsigned int LorentzAngleCalibration::numParameters() const
{
  return parameters_.size();
}

//======================================================================
unsigned int LorentzAngleCalibration::derivatives(std::vector<ValuesIndexPair> &outDerivInds,
						  const TransientTrackingRecHit &hit,
						  const TrajectoryStateOnSurface &tsos,
						  const edm::EventSetup &setup,
						  const EventInfo &eventInfo) const
{
  outDerivInds.clear();
  // FIXME: fake x-derivative of 2nd parameter, y-derivative is 0.
  //        (dito for both of 1st parameter)
  outDerivInds.push_back(ValuesIndexPair(Values(0.1, 0.), 1));

  return outDerivInds.size();
}

//======================================================================
bool LorentzAngleCalibration::setParameter(unsigned int index, double value)
{
  if (index >= parameters_.size()) {
    return false;
  } else {
    parameters_[index] = value;
    return true;
  }
}

//======================================================================
bool LorentzAngleCalibration::setParameterError(unsigned int index, double error)
{
  if (index >= paramUncertainties_.size()) {
    return false;
  } else {
    paramUncertainties_[index] = error;
    return true;
  }
}

//======================================================================
double LorentzAngleCalibration::getParameter(unsigned int index) const
{
  //   if (index >= parameters_.size()) {
  //     return 0.;
  //   } else {
  //     return parameters_[index];
  //   }
  return (index >= parameters_.size() ? 0. : parameters_[index]);
}

//======================================================================
double LorentzAngleCalibration::getParameterError(unsigned int index) const
{
  //   if (index >= paramUncertainties_.size()) {
  //     return 0.;
  //   } else {
  //     return paramUncertainties_[index];
  //   }
  return (index >= paramUncertainties_.size() ? 0. : paramUncertainties_[index]);
}

//======================================================================
void LorentzAngleCalibration::endOfJob()
{
  std::stringstream out("Parameters results are\n");

  for (unsigned int iPar = 0; iPar < parameters_.size(); ++iPar) {
    out << iPar << ": " << parameters_[iPar] << " +- " << paramUncertainties_[iPar] << "\n";
  }

  edm::LogInfo("Alignment") << "@SUB=LorentzAngleCalibration::endOfJob" << out.str();

  //FIXME: Write out to DB (after adding input values)...
}

//======================================================================
//======================================================================
// Plugin definition

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationPluginFactory.h"

DEFINE_EDM_PLUGIN(IntegratedCalibrationPluginFactory,
		   LorentzAngleCalibration, "LorentzAngleCalibration");
