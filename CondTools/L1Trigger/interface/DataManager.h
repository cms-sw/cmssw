#ifndef CondTools_L1Trigger_DataManager_h
#define CondTools_L1Trigger_DataManager_h

#include "FWCore/Framework/interface/DataKeyTags.h"

#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
/* #include "CondCore/DBCommon/interface/CoralTransaction.h" */
/* #include "CondCore/DBCommon/interface/PoolTransaction.h" */

#include "CondCore/MetaDataService/interface/MetaData.h"

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
        explicit DataManager (const std::string & connect,
			      const std::string & authenticationPath,
			      bool isOMDS = false );
        virtual ~DataManager ();

	void setDebug( bool debug ) ;

    protected:
        /* Returns type object for provided type name
         */
        edm::eventsetup::TypeTag findType (const std::string & type) const;

        // Database connection management
/*         cond::DBSession * poolSession; */
/*         cond::DBSession * coralSession; */
	cond::DBSession * session;
	cond::Connection * connection ;
/*         cond::CoralTransaction * coral; */
/*         cond::PoolTransaction * pool; */
        cond::MetaData * metadata;
};

}

#endif
