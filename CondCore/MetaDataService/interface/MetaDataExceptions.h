#ifndef COND_METADATAEXCEPTIONS_H
#define COND_METADATAEXCEPTIONS_H
#include "CondCore/DBCommon/interface/Exception.h"
#include <string>
namespace cond{
  class MetaDataDuplicateEntryException : public Exception{
  public:
    MetaDataDuplicateEntryException(const std::string& source, 
				   const std::string& name);
    ~MetaDataDuplicateEntryException() throw(){}
  };
}
#endif
