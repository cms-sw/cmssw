// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

class CastorDbService;
class CastorDbRecord;

#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"  	 
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h" 
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"

class CastorDbProducer : public edm::ESProducer {
public:
  CastorDbProducer( const edm::ParameterSet& );
  ~CastorDbProducer() override;
  
  std::shared_ptr<CastorDbService> produce( const CastorDbRecord& );

private:
  // ----------member data ---------------------------
  using HostType = edm::ESProductHost<CastorDbService,
                                      CastorPedestalsRcd,
                                      CastorPedestalWidthsRcd,
                                      CastorGainsRcd,
                                      CastorGainWidthsRcd,
                                      CastorQIEDataRcd,
                                      CastorChannelQualityRcd,
                                      CastorElectronicsMapRcd>;

  void setupPedestals(const CastorPedestalsRcd&, CastorDbService*);
  void setupPedestalWidths(const CastorPedestalWidthsRcd&, CastorDbService*);
  void setupGains(const CastorGainsRcd&, CastorDbService*);
  void setupGainWidths(const CastorGainWidthsRcd&, CastorDbService*);
  void setupQIEData(const CastorQIEDataRcd&, CastorDbService*);
  void setupChannelQuality(const CastorChannelQualityRcd&, CastorDbService*);
  void setupElectronicsMap(const CastorElectronicsMapRcd&, CastorDbService*);

  edm::ReusableObjectHolder<HostType> holder_;

  std::vector<std::string> mDumpRequest;
  std::ostream* mDumpStream;
};
