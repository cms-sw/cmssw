#ifndef CalibTracker_SiStripLorentzAngle_MeasureLA_h
#define CalibTracker_SiStripLorentzAngle_MeasureLA_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CalibTracker/SiStripLorentzAngle/interface/Book.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"

namespace sistrip {

class MeasureLA : public edm::ESProducer {

 public:

  explicit MeasureLA(const edm::ParameterSet&);
  boost::shared_ptr<SiStripLorentzAngle> produce(const SiStripLorentzAngleRcd&);
  
 private:

  void store_calibrations();
  void store_methods_and_granularity(const edm::VParameterSet&);

  void process_reports();
  template<class T> 
  void write_report_text(std::string, LA_Filler_Fitter::Method, std::map<T,LA_Filler_Fitter::Result>);
  void write_report_plots(std::string, LA_Filler_Fitter::Method, bool);

  void calibrate(std::pair<uint32_t,LA_Filler_Fitter::Method>, LA_Filler_Fitter::Result&);
  std::pair<uint32_t,LA_Filler_Fitter::Method> calibration_key(std::string layer,LA_Filler_Fitter::Method method);
  std::pair<uint32_t,LA_Filler_Fitter::Method> calibration_key(uint32_t detid,LA_Filler_Fitter::Method method);

  std::vector<std::string> inputFiles;
  std::string inFileLocation;
  edm::FileInPath fp_;
  unsigned maxEvents;
  edm::VParameterSet reports, measurementPreferences, calibrations;
  std::map<std::pair<uint32_t,LA_Filler_Fitter::Method>,float> slope, offset, error_scaling;
  int32_t methods;
  bool byModule, byLayer;
  Book book;

};

}
#endif
