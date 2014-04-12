#include "FWCore/Framework/interface/MakeDataException.h"

// forward declarations
namespace edm {
   namespace eventsetup {

MakeDataException::MakeDataException( const EventSetupRecordKey& iRecordKey, const DataKey& iDataKey) : 
cms::Exception("MakeDataException"),
message_(standardMessage(iRecordKey,iDataKey))
{
  this->append(myMessage());
}

      // ---------- static member functions --------------------
std::string MakeDataException::standardMessage( const EventSetupRecordKey& iRecordKey, const DataKey& iDataKey) 
{
 std::string returnValue = std::string("Error while making data \"") 
   + iDataKey.type().name()
   + "\" \""
   + iDataKey.name().value()
   + "\" in Record "
   + iRecordKey.type().name();
   return returnValue;
}
   

   }
}
