#ifndef CalibTracker_SiStripESProducers_SiStripLorentzAngleGenerator_H
#define CalibTracker_SiStripESProducers_SiStripLorentzAngleGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include <string>

/**
 * Generator of the ideal/fake conditions for the LorentzAngle.<br>
 * Depending on the parameters passed via cfg, it is able to generate the values per DetId
 * with a gaussian distribution and a uniform distribution. When setting the sigma of the gaussian to 0
 * and passing a single value the generated values are fixed.
 */

class SiStripLorentzAngleGenerator : public SiStripCondObjBuilderBase<SiStripLorentzAngle> {
 public:

  explicit SiStripLorentzAngleGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripLorentzAngleGenerator();
  
  void getObj(SiStripLorentzAngle* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();
  /// Fills the estimatedValues array. Returns true if the generation must be done with the uniform distribution.
  bool setEstimatedValues(const vector<double> & estimatedValueMinMax, double * estimatedValues) const;
  /**
   * This method fills the hallMobility_ variable with different values according to the parameters passed in the cfg.<br>
   * - If a min and max value were passed it takes the value from a uniform distribution.
   * - If only a single value was passed and the error is set != 0 it takes the value from a gaussian distribution.
   * - If the error is 0 and only one value is passed it takes the fixed value.
   */
  void setHallMobility(const double * estimatedValue, const double & stdDev, const bool generateUniform);
  float hallMobility_;
  
};

#endif 
