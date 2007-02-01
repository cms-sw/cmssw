#ifndef SiPixelMonitorDigi_SiPixelDigiModule_h
#define SiPixelMonitorDigi_SiPixelDigiModule_h

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/cstdint.hpp>

class SiPixelDigiModule {        

 public:
  
  SiPixelDigiModule();
  
  SiPixelDigiModule(uint32_t id);
  
  ~SiPixelDigiModule();

  typedef edm::DetSet<PixelDigi>::const_iterator    DigiIterator;

  void book();

  //void fill(const PixelDigiCollection* digiCollection);
  void fill(const edm::DetSetVector<PixelDigi> & input);
  
 private:
  uint32_t id_;
  MonitorElement* meNDigis_;
  MonitorElement* meADC_;
  MonitorElement* meCol_;
  MonitorElement* meRow_;
  MonitorElement* mePixDigis_;
  
};
#endif
