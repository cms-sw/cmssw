#ifndef CalibTracker_SiStripESProducers_SiStripPedestalsFakeESSource_H
#define CalibTracker_SiStripESProducers_SiStripPedestalsFakeESSource_H

#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsESSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/cstdint.hpp"
#include <memory>


/** 
    @class SiStripPedestalsFakeESSource
    @brief Fake source of SiStripPedestals object.
    @author D. Giordano
*/
class SiStripPedestalsFakeESSource : public SiStripPedestalsESSource {

 public:

  SiStripPedestalsFakeESSource( const edm::ParameterSet& );
  virtual ~SiStripPedestalsFakeESSource() {;}
  
     
private:
  

  SiStripPedestals* makePedestals();


private:

  //parameters for strip length proportional noise generation. not used if random mode is chosen
  uint32_t PedestalValue_;
  double LowThValue_;
  double HighThValue_;

  bool printdebug_;
  edm::FileInPath fp_;

};


#endif // CalibTracker_SiStripESProducers_SiStripPedestalsFakeESSource_H

