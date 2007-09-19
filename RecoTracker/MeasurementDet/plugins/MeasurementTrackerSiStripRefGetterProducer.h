#ifndef RecoTracker_MeasurementDet_MeasurementTrackerSiStripRefGetterProducer_H
#define RecoTracker_MeasurementDet_MeasurementTrackerSiStripRefGetterProducer_H


/** \class MeasurementTrackerSiStripRefGetterProducer
 * module to allow for unpack on demand
 * retrieves a SiStripLazyGetter from the event
 * retrieves a MeasurementTrackerOD from the event 
 * produces a SiStripRefGetter, defined by MeasurementTrackerOD::define(...)
 * 
 * $Dates: 2007/09/21 13:28 $
 * $Revision: 1.1 $
 *
 * \author Jean-Roch Vlimant  UCSB
 *
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

#include <string>
#include <memory>
#include "boost/bind.hpp"

class MeasurementTrackerSiStripRefGetterProducer : public edm::EDProducer {
  
 public:

  typedef edm::SiStripLazyGetter<SiStripCluster> LazyGetter;
  typedef edm::SiStripRefGetter<SiStripCluster> RefGetter;

  MeasurementTrackerSiStripRefGetterProducer( const edm::ParameterSet& );
  ~MeasurementTrackerSiStripRefGetterProducer();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  /// Input module label of SiStripLazyGetter
  edm::InputTag inputModuleLabel_;

  /// Cabling 
  edm::ESHandle<SiStripRegionCabling> cabling_;

  /// name of the defined Measurement Tracker On Demand
  std::string measurementTrackerName_;
};

#endif 

