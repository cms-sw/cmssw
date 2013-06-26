#ifndef CalibTracker_SiStripLorentzAngle_MeasureLA_h
#define CalibTracker_SiStripLorentzAngle_MeasureLA_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "CalibTracker/SiStripCommon/interface/Book.h"

namespace sistrip {

class MeasureLA : public edm::ESProducer {

 public:

  explicit MeasureLA(const edm::ParameterSet&);
  boost::shared_ptr<SiStripLorentzAngle> produce(const SiStripLorentzAngleRcd&);
  
 private:

  enum GRANULARITY {LAYER=0, MODULE=1, MODULESUMMARY=2};
  static std::string granularity(int32_t g) { switch(g) {
    case LAYER: return "_layer"; 
    case MODULE: return "_module"; 
    case MODULESUMMARY: return "_moduleSummary";
  } return "";};

  void store_calibrations();
  void store_methods_and_granularity(const edm::VParameterSet&);

  void summarize_module_muH_byLayer();
  void process_reports() const;
  template<class T>
  void write_report_text(const std::string, const LA_Filler_Fitter::Method&, const std::map<T,LA_Filler_Fitter::Result>&) const;
  void write_report_text_ms(const std::string, const LA_Filler_Fitter::Method ) const;
  void write_report_plots(const std::string, const LA_Filler_Fitter::Method, const GRANULARITY) const;

  void calibrate(const std::pair<unsigned,LA_Filler_Fitter::Method>, LA_Filler_Fitter::Result&) const;
  static std::pair<unsigned,LA_Filler_Fitter::Method> calibration_key(const std::string layer, const LA_Filler_Fitter::Method);
  static std::pair<unsigned,LA_Filler_Fitter::Method> calibration_key(const uint32_t detid, const LA_Filler_Fitter::Method);

  const std::vector<std::string> inputFiles;
  const std::string inFileLocation;
  const edm::FileInPath fp_;
  const edm::VParameterSet reports, measurementPreferences, calibrations;
  std::map<std::pair<uint32_t,LA_Filler_Fitter::Method>,float> slope, offset, error_scaling;
  int32_t methods;
  bool byModule, byLayer;
  const float localybin;
  const unsigned stripsperbin,maxEvents;
  Book book;

};

}
#endif
