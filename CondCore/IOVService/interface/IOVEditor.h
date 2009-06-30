#ifndef CondCore_IOVService_IOVEditor_h
#define CondCore_IOVService_IOVEditor_h
#include <string>
#include <vector>
#include "CondCore/DBCommon/interface/Time.h"
//
// Package:     CondCore/IOVService
// Class  :     IOVEditor
//
/**\class IOVEditor IOVEditor.h CondCore/IOVService/interface/IOVEditor.h
   Description: Abstract interface for iov sequence manipulation
*/
//
// Author:      Zhen Xie
//
namespace cond{
  class IOVElement;

  class IOVEditor{
  public:
    /// Destructor
    virtual ~IOVEditor(){}

    virtual void create(cond::TimeType timetype,cond::Time_t lastTill) = 0;



    /// Assign a payload with till time. Returns the payload index in the iov sequence
    virtual unsigned int insert( cond::Time_t tillTime,
				 const std::string& payloadToken
				 ) = 0;

    /// Append a payload with known since time. The previous last payload's till time will be adjusted to the new payload since time. Returns the payload index in the iov sequence
    virtual unsigned int append(  cond::Time_t sinceTime,
				  const std::string& payloadToken
				  ) = 0;

    /// insert a payload with known since in any position
    virtual unsigned int 
    freeInsert( cond::Time_t sinceTime ,
		const std::string& payloadToken
		)=0;

    /// Bulk append of iov chunck
    virtual void bulkAppend( std::vector< std::pair<cond::Time_t,std::string> >& values ) = 0;

    virtual void bulkAppend(std::vector< cond::IOVElement >& values)=0;
    
    //stamp iov
    virtual void stamp(std::string const & icomment, bool append=false)=0;

    /// Update the closure of the iov sequence
    virtual void updateClosure( cond::Time_t newtillTime ) = 0;
    virtual void deleteEntries( bool withPayload=false ) = 0;
    virtual void import( const std::string& sourceIOVtoken ) = 0;
    /// Returns the token of the iov sequence associated with this editor
    virtual std::string token() const = 0;
  protected:
    /// Private constructor
    IOVEditor(){}
  };
}//ns cond
#endif
