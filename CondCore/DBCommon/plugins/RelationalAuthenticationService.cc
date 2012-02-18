#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "RelationalAccess/AuthenticationCredentials.h"
#include "RelationalAccess/AuthenticationServiceException.h"
#include "CondCore/DBCommon/interface/CoralServiceMacros.h"
#include "RelationalAuthenticationService.h"
//
#include "RelationalAccess/AuthenticationServiceException.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralKernel/Property.h"
#include "CoralKernel/Context.h"
//
#include <memory>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <boost/filesystem.hpp>
#include <boost/version.hpp>
#include <boost/bind.hpp>
//#include <iostream>
#include "CoralBase/MessageStream.h"

cond::RelationalAuthenticationService::RelationalAuthenticationService::RelationalAuthenticationService( const std::string& key )
  : coral::Service( key ),
    m_db(),
    m_cache()
{
}

cond::RelationalAuthenticationService::RelationalAuthenticationService::~RelationalAuthenticationService()
{
}

const coral::IAuthenticationCredentials&
cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials( const std::string& connectionString ) const
{
  const coral::IAuthenticationCredentials* creds = m_cache.get( connectionString );
  if( !creds ){
    m_db.setUpForConnectionString( connectionString );
    m_db.exportAll( m_cache );
  } 
  creds = m_cache.get( connectionString );
  if( ! creds ){
    std::string msg("No Authentication available for connection=\"");
    msg += connectionString + "\".";
    throw coral::AuthenticationServiceException( msg, "cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials", "");
  }
  return *creds;
}

const coral::IAuthenticationCredentials&
cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials( const std::string& connectionString,
										     const std::string& role ) const
{
  const coral::IAuthenticationCredentials* creds = m_cache.get( connectionString, role );
  if( !creds ){
    m_db.setUpForConnectionString( connectionString );
    m_db.exportAll( m_cache );
  } 
  creds = m_cache.get( connectionString, role );
  if( ! creds ){
    std::string msg("No Authentication available for connection=\"");
    msg += connectionString + "\".";
    msg += " and role=\"" + role + "\".";
    throw coral::AuthenticationServiceException( msg, "cond::RelationalAuthenticationService::RelationalAuthenticationService::credentials","");
  }
  return *creds;
}

DEFINE_CORALSERVICE(cond::RelationalAuthenticationService::RelationalAuthenticationService,"COND/Services/RelationalAuthenticationService");
