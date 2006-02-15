#ifndef SiPixelMonitorDigi_SiPixelDigiModule_h
#define SiPixelMonitorDigi_SiPixelDigiModule_h
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include <boost/cstdint.hpp>

class SiPixelDigiModule {                                                                                                                                                        
 public:
  
  SiPixelDigiModule();
  
  SiPixelDigiModule(uint32_t id);
  
  ~SiPixelDigiModule();

  void book();

  void fill(const PixelDigiCollection* digiCollection);
  
 private:
  uint32_t id_;
  MonitorElement* meNDigis_;
  MonitorElement* meADC_;
  MonitorElement* meCol_;
  MonitorElement* meRow_;
  
};
#endif
