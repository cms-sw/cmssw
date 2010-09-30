#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripAPVRestorer.h"

#include <cmath>
#include <iostream>

void SiStripAPVRestorer::fixAPVsCM(edm::DetSet<SiStripProcessedRawDigi>& cmdigis) {
  
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
	  //std::cout << "  apvFlag was " << *apvf_iter << std::endl;
	  //std::cout << "  baseline was " << *cmv_iter << std::endl;
	  cmdigis.push_back( SiStripProcessedRawDigi( -999.) );
	}
	else
	  cmdigis.push_back( SiStripProcessedRawDigi( *cmv_iter ) );
	apvf_iter++;
	cmv_iter++;
      }
  }
