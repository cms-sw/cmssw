#ifndef CalibTracker_SiStripLorentzAngle_MeasureLA_h
#define CalibTracker_SiStripLorentzAngle_MeasureLA_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CalibTracker/SiStripLorentzAngle/interface/Book.h"

namespace sistrip {

class MeasureLA : public edm::ESProducer {

 public:

  explicit MeasureLA(const edm::ParameterSet&);
  boost::shared_ptr<SiStripLorentzAngle> produce(const SiStripLorentzAngleRcd&);
  
 private:

  std::vector<std::string> inputFiles;
  std::string inFileLocation;
  bool byLayer,byModule;
  float chi2ndof_cut;
  unsigned maxEvents,nEntries_cut;
  edm::FileInPath fp_;
  Book book;

};

}
#endif
