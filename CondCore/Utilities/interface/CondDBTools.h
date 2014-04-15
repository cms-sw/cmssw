#ifndef Utilities_CondDBTools_h
#define Utilities_CondDBTools_h

#include "CondCore/CondDB/interface/Time.h"
//
#include <string>

namespace cond {

  namespace persistency {

    class Session;

    typedef enum { NEW=0, UPDATE, REPLACE } UpdatePolicy;

    size_t copyTag( const std::string& sourceTag, 
		    Session& sourceSession, 
		    const std::string& destTag, 
		    Session& destSession, 
		    UpdatePolicy policy,
		    bool log, 
		    bool forValidation );   

    size_t importIovs( const std::string& sourceTag, 
		       Session& sourceSession, 
		       const std::string& destTag, 
		       Session& destSession, 
		       cond::Time_t begin,
		       cond::Time_t end,
		       bool log );  

    bool copyIov( Session& session,
		  const std::string& sourceTag,
		  const std::string& destTag,
		  cond::Time_t souceSince,
		  cond::Time_t destSince,
		  bool log );
 
    bool compareTags( const std::string& firstTag,
		      Session& firstSession, 
		      const std::string& firstFileName, 
		      const std::string& secondTag, 
		      Session& secondSession,
		      const std::string& secondFileName );

    bool validateTag( const std::string& refTag, Session& refSession, const std::string& candTag, Session& candSession );

  }

}

#endif
