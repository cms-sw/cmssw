#ifndef CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationStruct_h
#define CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationStruct_h

// system includes
#include <map>
#include <vector>

// user includes
#include "DQMServices/Core/interface/DQMStore.h"

struct SiStripLorentzAngleCalibrationHistograms {
public:
  SiStripLorentzAngleCalibrationHistograms() = default;

  // B field
  std::string bfield_;

  // APV mode
  std::string apvmode_;

  std::map<uint32_t, int> orientation_;
  std::map<uint32_t, float> la_db_;
  std::map<uint32_t, std::string> moduleLocationType_;

  // histogramming
  std::map<std::string, dqm::reco::MonitorElement*> h1_;
  std::map<std::string, dqm::reco::MonitorElement*> h2_;
  std::map<std::string, dqm::reco::MonitorElement*> p_;

  // These are vectors since std:map::find is expensive
  // we're going to profi of the dense indexing offered by
  // SiStripHashedDetId and index the histogram position
  // with the natural booking order
  std::vector<dqm::reco::MonitorElement*> h2_ct_w_m_;
  std::vector<dqm::reco::MonitorElement*> h2_ct_var2_m_;
  std::vector<dqm::reco::MonitorElement*> h2_ct_var3_m_;

  std::vector<dqm::reco::MonitorElement*> h2_t_w_m_;
  std::vector<dqm::reco::MonitorElement*> h2_t_var2_m_;
  std::vector<dqm::reco::MonitorElement*> h2_t_var3_m_;

  std::map<std::string, dqm::reco::MonitorElement*> hp_;

  // info
  std::map<std::string, int> nlayers_;
  std::vector<std::string> modtypes_;
  std::map<std::string, float> la_;
};

#endif
