#include "CondTools/L1Trigger/src/DataManager.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"

namespace l1t
{

DataManager::DataManager (const std::string & connect, const std::string & catalog)
{
    // Initialize session used with this object
    session = new cond::DBSession ();
    session->sessionConfiguration ().setAuthenticationMethod (cond::Env);
//    session->sessionConfiguration ().setMessageLevel (cond::Info);
    session->connectionConfiguration ().enableConnectionSharing ();
    session->connectionConfiguration ().enableReadOnlySessionOnUpdateConnections ();

    // create Coral connection and pool. This ones should be connected on required basis
    coral = new cond::RelationalStorageManager (connect, session);
    pool = new cond::PoolStorageManager (connect, catalog, session);

    // and data object
    metadata = new cond::MetaData (*coral);
}

DataManager::~DataManager ()
{
    // detele all in reverse direction
    // closing and comminting may be a good idea here
    // in case we have exception in some other part between connect/close pairs

    delete metadata;
    delete pool;
    delete coral;
    delete session;
}

}
