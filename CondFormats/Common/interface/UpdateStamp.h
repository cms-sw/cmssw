#ifndef Cond_UpdateStamp_h
#define Cond_UpdateStamp_h

#include "CondFormats/Common/interface/Time.h"
#include <string>


namespace cond {
  
  /** class to "stamp" a new version of an updateble persistent object
      it includes a timestamp, a sequential revision number and a comment
      no history in mantained at the moment
   */
  class UpdateStamp {
  public:
    typedef enum { Unknown=-1, Obsolete, Tag, TagInGT, ChildTag, ChildTagInGT } ScopeType;
  public:
    // constrcutor creates and invalid stamp
    UpdateStamp();
    
    UpdateStamp(UpdateStamp const & rhs);
    
    virtual ~UpdateStamp();
    
    // stamp and return current revision number;
    int stamp( std::string const & icomment, bool append=false);
    
    int revision() const { return  m_revision;}
    
    cond::Time_t timestamp() const { return m_timestamp;}
    
    std::string const & comment() const  { return m_comment;}

    void setScope( ScopeType type ) { m_scope = type;}
    ScopeType scope() const { return m_scope;}
 
  private:
    
    int m_revision;
    cond::Time_t m_timestamp;
    std::string m_comment;
    ScopeType m_scope;
    
  };

} // nc cond

#endif
