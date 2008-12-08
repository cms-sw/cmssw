#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"

class CSCTFConfigOnlineProd : public L1ConfigOnlineProdBase< L1MuCSCTFConfigurationRcd, L1MuCSCTFConfiguration > {
   public:
      CSCTFConfigOnlineProd(const edm::ParameterSet& iConfig)
         : L1ConfigOnlineProdBase< L1MuCSCTFConfigurationRcd, L1MuCSCTFConfiguration >( iConfig ) {}
      ~CSCTFConfigOnlineProd() {}

      virtual boost::shared_ptr< L1MuCSCTFConfiguration > newObject( const std::string& objectKey ) ;
   private:
};

