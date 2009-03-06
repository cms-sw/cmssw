#ifndef CalibTracker_SiStripESProducers_SiStripQualityConfigurableFakeESSource
#define CalibTracker_SiStripESProducers_SiStripQualityConfigurableFakeESSource

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/DataRecord/interface/SiStripBadModuleRcd.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

//
// class declaration
//


class SiStripQualityConfigurableFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripQualityConfigurableFakeESSource(const edm::ParameterSet&);
  ~SiStripQualityConfigurableFakeESSource(){};
  
  
  std::auto_ptr<SiStripBadStrip> produce(const SiStripBadModuleRcd&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripQualityConfigurableFakeESSource( const SiStripQualityConfigurableFakeESSource& );

  void selectDetectors(const std::vector<uint32_t>& , std::vector<uint32_t>& );

  bool isTIBDetector(const uint32_t & therawid,
		     uint32_t requested_layer,
		     uint32_t requested_bkw_frw,
		     uint32_t requested_int_ext,
		     uint32_t requested_string,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;
  
  bool isTOBDetector(const uint32_t & therawid,
		     uint32_t requested_layer,
		     uint32_t requested_bkw_frw,
		     uint32_t requested_rod,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;

  bool isTIDDetector(const uint32_t & therawid,
		     uint32_t requested_side,
		     uint32_t requested_wheel,
		     uint32_t requested_ring,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;

  bool isTECDetector(const uint32_t & therawid,
		     uint32_t requested_side,
		     uint32_t requested_wheel,
		     uint32_t requested_petal_bkw_frw,
		     uint32_t requested_petal,			
		     uint32_t requested_ring,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters BadComponentList_;


  const edm::ParameterSet& iConfig_;
  edm::FileInPath fp_;
  bool printdebug_;
  };

#endif
