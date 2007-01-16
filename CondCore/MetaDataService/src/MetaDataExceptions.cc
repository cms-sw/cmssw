#include "CondCore/MetaDataService/interface/MetaDataExceptions.h"
cond::MetaDataDuplicateEntryException::MetaDataDuplicateEntryException(const std::string& source, const std::string& name)
:cond::Exception(source+std::string(": metadata entry \"")+name+std::string("\" already exists")){
}
