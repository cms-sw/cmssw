#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"


class CSCTFObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      CSCTFObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
         : L1ObjectKeysOnlineProdBase( iConfig ) {}
      ~CSCTFObjectKeysOnlineProd() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
};

