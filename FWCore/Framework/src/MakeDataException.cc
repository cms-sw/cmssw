#include "FWCore/Framework/interface/MakeDataException.h"

// forward declarations
namespace edm {
   namespace eventsetup {

MakeDataException::MakeDataException(const MakeDataExceptionInfoBase &info) : 
cms::Exception("MakeDataException"),
message_(standardMessage(info))
{
  this->append(myMessage());
}

MakeDataException::MakeDataException(const std::string& iAdditionalInfo,
                     const MakeDataExceptionInfoBase& info) : 
cms::Exception("MakeDataException"),
message_(messageWithInfo(info, iAdditionalInfo))
{
  this->append(this->myMessage());
}

      // ---------- static member functions --------------------
std::string MakeDataException::standardMessage(const MakeDataExceptionInfoBase& info) 
{
 std::string returnValue = std::string("Error while making data \"") 
 + info.dataClassName()  
 + "\" \""
 + info.key().name().value()
 + "\" in Record "
 + info.recordClassName();
 return returnValue;
}
   
std::string MakeDataException::messageWithInfo(const MakeDataExceptionInfoBase& info,
                                               const std::string& iInfo) 
{
  return standardMessage(info) +"\n"+iInfo;
}

   }
}
