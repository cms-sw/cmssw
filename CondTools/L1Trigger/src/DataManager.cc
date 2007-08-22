#include "CondTools/L1Trigger/src/DataManager.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"

#include "CondCore/DBCommon/interface/Exception.h"

namespace l1t
{

DataManager::DataManager (const std::string & connect, const std::string & catalog)
{
    // Initialize session used with this object
    poolSession = new cond::DBSession ();
    poolSession->sessionConfiguration ().setAuthenticationMethod (cond::Env);
//    poolSession->sessionConfiguration ().setMessageLevel (cond::Info);
    poolSession->connectionConfiguration ().enableConnectionSharing ();
    poolSession->connectionConfiguration ().enableReadOnlySessionOnUpdateConnections ();

    coralSession = new cond::DBSession ();
    coralSession->sessionConfiguration ().setAuthenticationMethod (cond::Env);
//    coralSession->sessionConfiguration ().setMessageLevel (cond::Info);
    coralSession->connectionConfiguration ().enableConnectionSharing ();
    coralSession->connectionConfiguration ().enableReadOnlySessionOnUpdateConnections ();

    // create Coral connection and pool. This ones should be connected on required basis
    coral = new cond::RelationalStorageManager (connect, coralSession);
    pool = new cond::PoolStorageManager (connect, catalog, poolSession);

    // and data object
    metadata = new cond::MetaData (*coral);

    // can I use two sessions? One for pool and one for coral? So that I could connect to one and another....
    // will sqlite will be happy with transactions?
}

DataManager::~DataManager ()
{
    // detele all in reverse direction
    // closing and comminting may be a good idea here
    // in case we have exception in some other part between connect/close pairs

    delete metadata;
    delete pool;
    delete coral;
    delete coralSession;
    delete poolSession;
}

edm::eventsetup::TypeTag DataManager::findType (const std::string & type) const
{
     static edm::eventsetup::TypeTag defaultType;
     edm::eventsetup::TypeTag typeTag = edm::eventsetup::TypeTag::findType (type);

     if (typeTag == defaultType)
        throw cond::Exception ("DataReader::findType") << "Type " << type << " was not found";

     return typeTag;
}


}
