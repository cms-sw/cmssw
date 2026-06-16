#ifndef CalibTracker_SiStripESProducers_SiStripFedCablingFakeESSource_H
#define CalibTracker_SiStripESProducers_SiStripFedCablingFakeESSource_H

#include "CalibTracker/SiStripESProducers/interface/SiStripFedCablingESProducer.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class SiStripFedCabling;
class SiStripFedCablingRcd;

/**
   @class SiStripFedCablingFakeESSource
   @author R.Bainbridge
   @brief Builds cabling map based on list of DetIds and FedIds read from ascii files
*/
class SiStripFedCablingFakeESSource : public SiStripFedCablingESProducer,
                                      public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit SiStripFedCablingFakeESSource(const edm::ParameterSet&);
  ~SiStripFedCablingFakeESSource() override;

private:
  /** Builds cabling map based on ascii files. */
  SiStripFedCabling* make(const SiStripFedCablingRcd&) override;

  /** Location of ascii file containing FedIds. */
  edm::FileInPath fedIds_;
  edm::ParameterSet pset_;
  SiStripDetInfo m_detInfo;
};

#endif  // CalibTracker_SiStripESProducers_SiStripFedCablingFakeESSource_H
