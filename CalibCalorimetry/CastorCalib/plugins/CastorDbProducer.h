// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

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

  // callbacks
  void pedestalsCallback (const CastorPedestalsRcd& fRecord);
  void pedestalWidthsCallback (const CastorPedestalWidthsRcd& fRecord);
  void gainsCallback (const CastorGainsRcd& fRecord);
  void gainWidthsCallback (const CastorGainWidthsRcd& fRecord);
  void QIEDataCallback (const CastorQIEDataRcd& fRecord);
  void channelQualityCallback (const CastorChannelQualityRcd& fRecord);
  void electronicsMapCallback (const CastorElectronicsMapRcd& fRecord);

   private:
      // ----------member data ---------------------------
  std::shared_ptr<CastorDbService> mService;
  std::vector<std::string> mDumpRequest;
  std::ostream* mDumpStream;
};
