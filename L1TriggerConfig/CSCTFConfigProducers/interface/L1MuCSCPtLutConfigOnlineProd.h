#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"

class L1MuCSCPtLutConfigOnlineProd : public L1ConfigOnlineProdBase< L1MuCSCPtLutRcd, L1MuCSCPtLut > {
   public:
      L1MuCSCPtLutConfigOnlineProd(const edm::ParameterSet& iConfig)
         : L1ConfigOnlineProdBase< L1MuCSCPtLutRcd, L1MuCSCPtLut >( iConfig ) {}
      ~L1MuCSCPtLutConfigOnlineProd() {}

      virtual boost::shared_ptr< L1MuCSCPtLut > newObject( const std::string& objectKey ) ;
   private:
};

