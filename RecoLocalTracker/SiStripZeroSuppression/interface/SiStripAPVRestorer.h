#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPAPVRESTORER_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPAPVRESTORER_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"

#include <vector>
#include <stdint.h>

class SiStripAPVRestorer {

 friend class SiStripRawProcessingFactory;

 public:
  
  virtual ~SiStripAPVRestorer() {};

  virtual void     init(const edm::EventSetup& es) = 0;
  virtual int16_t  inspect(const uint32_t&, std::vector<int16_t>&) = 0;
  virtual int16_t  inspect(const uint32_t&, std::vector<float>&) = 0;
  virtual void     restore(std::vector<int16_t>&) = 0;
  virtual void     restore(std::vector<float>&) = 0;
  virtual void     fixAPVsCM(edm::DetSet<SiStripProcessedRawDigi>& cmdigis);
  
 protected:

  SiStripAPVRestorer(){};

  std::vector<bool> apvFlags;

};

#endif
