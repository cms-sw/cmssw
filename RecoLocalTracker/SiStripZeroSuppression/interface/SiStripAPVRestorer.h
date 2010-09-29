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

  virtual void init(const edm::EventSetup& es) = 0;
  virtual void inspect(const uint32_t&, std::vector<int16_t>&) = 0;
  virtual void inspect(const uint32_t&, std::vector<float>&) = 0;
  virtual void restore(std::vector<int16_t>&) = 0;
  virtual void restore(std::vector<float>&) = 0;
  virtual void fixAPVsCM(edm::DetSet<SiStripProcessedRawDigi>& cmdigis) {
  
    // cmdigis should be the same size as apvFlags
    // otherwise something pathological has happened and we do nothing
    if ( cmdigis.size() != apvFlags.size() ) return;
    
    edm::DetSet<SiStripProcessedRawDigi>::iterator cm_iter = cmdigis.begin();
    std::vector<bool>::const_iterator apvf_iter = apvFlags.begin();
    
    // No way to change the adc value of a SiStripProcessedRawDigi
    // so we just extract the values, clear the DetSet, and
    // replace with the proper values.
    
    std::vector<float> cmvalues;
    for( ; cm_iter != cmdigis.end(); ++cm_iter  ) cmvalues.push_back( (*cm_iter).adc() );
    cmdigis.clear();
    
    std::vector<float>::const_iterator cmv_iter = cmvalues.begin();
    while( apvf_iter != apvFlags.end() )
      {
	if( *apvf_iter) {
	  std::cout << "  apvFlag was " << *apvf_iter << std::endl;
	  std::cout << "  baseline was " << *cmv_iter << std::endl;
	  cmdigis.push_back( SiStripProcessedRawDigi( -999.) );
	}
	else
	  cmdigis.push_back( SiStripProcessedRawDigi( *cmv_iter ) );
	apvf_iter++;
	cmv_iter++;
      }
  };
  
  
 protected:

  SiStripAPVRestorer(){};

  std::vector<bool> apvFlags;

};

#endif
