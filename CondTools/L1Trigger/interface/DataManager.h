#ifndef CondTools_L1Trigger_DataManager_h
#define CondTools_L1Trigger_DataManager_h

#include "FWCore/Framework/interface/DataKeyTags.h"
#include "CondCore/CondDB/interface/Session.h"

#include <string>

namespace l1t
{

/* Helper class that provides common objects required to access Pool and Coral DB's.
 * This class will initialize connections and makes sure that they are closed
 * when it is destroyed.
 * Connections are initialized, but user is still responsible for opening and commiting
 * them
 */
class DataManager
{
    public:
        DataManager() ;
        explicit DataManager (const std::string & connectString,
			      const std::string & authenticationPath,
			      bool isOMDS = false );
        virtual ~DataManager ();

        void connect(const std::string & connectString,
		     const std::string & authenticationPath,
		     bool isOMDS = false );
	void setDebug( bool debug ) ;

  cond::persistency::Session dbSession()
	  { return session ; }

    protected:
        //Returns type object for provided type name
        edm::eventsetup::TypeTag findType (const std::string & type);

        // Database connection management
  cond::persistency::Session  session;

private:
  bool debugFlag = false;
};

}

#endif
