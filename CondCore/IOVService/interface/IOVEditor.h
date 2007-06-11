#ifndef CondCore_IOVService_IOVEditor_h
#define CondCore_IOVService_IOVEditor_h
#include <string>
#include <vector>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class IOVEditor{
  public:
    virtual ~IOVEditor(){}
    virtual void insert( cond::Time_t tillTime,
			 const std::string& payloadToken
			 ) = 0;
    virtual void bulkInsert( std::vector< std::pair<cond::Time_t,std::string> >& values ) = 0;
    virtual void updateClosure( cond::Time_t newtillTime ) = 0;
    virtual void append(  cond::Time_t sinceTime,
			  const std::string& payloadToken
			  ) = 0;
    virtual void deleteEntries( bool withPayload=false ) = 0;
    virtual void import( const std::string& sourceIOVtoken ) = 0;
    virtual std::string token() const = 0;
  protected:
    IOVEditor(){}
  };
}//ns cond
#endif
