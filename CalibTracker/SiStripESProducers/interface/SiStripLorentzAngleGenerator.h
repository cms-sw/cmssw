
#ifndef CalibTracker_SiStripESProducers_SiStripLorentzAngleGenerator_H
#define CalibTracker_SiStripESProducers_SiStripLorentzAngleGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripDepCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include <string>

/**
 * Generator of the ideal/fake conditions for the LorentzAngle.<br>
 * It receives input values with layer granularity and it is able to perform gaussian smearing or
 * use a uniform distribution at the module level. <br>
 * Depending on the parameters passed via cfg, it is able to generate the values per DetId
 * with a gaussian distribution and a uniform distribution. When setting the sigma of the gaussian to 0
 * and passing a single value the generated values are fixed. <br>
 * For TID and TEC the decision to generate with a uniform distribution comes from the setting
 * for the first layers of TIB and TOB.
 */

class SiStripLorentzAngleGenerator : public SiStripDepCondObjBuilderBase<SiStripLorentzAngle,TrackerTopology> {
 public:

  explicit SiStripLorentzAngleGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripLorentzAngleGenerator();
  
  void getObj(SiStripLorentzAngle* & obj, const TrackerTopology* tTopo){obj=createObject(tTopo);}

 private:

  SiStripLorentzAngle* createObject(const TrackerTopology* tTopo);
  float hallMobility_;
  /**
   * This method fills the hallMobility_ variable with different values according to the parameters passed in the cfg. <br>
   * - If a min and max value were passed it takes the value from a uniform distribution.
   * - If only a single value was passed and the error is set != 0 it takes the value from a gaussian distribution.
   * - If the error is 0 and only one value is passed it takes the fixed min value.
   */
  void setHallMobility(const double & meanMin, const double & meanMax, const double & sigma, const bool uniform);
  /// Method used to determine whether to generate with a uniform distribution for each layer
  void setUniform(const std::vector<double> & TIB_EstimatedValuesMin, const std::vector<double> & TIB_EstimatedValuesMax, std::vector<bool> & uniformTIB);
};

#endif 
