#ifndef RUNNUMBER_HANDLER_H
#define RUNNUMBER_HANDLER_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/RunNumber.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"



/*
 *  \class RunNumberHandler
 *  
 *  popcon handler for holding  runnumber info from Run Control
 *  
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (June-25-2008)
 *
*/


using namespace std;



  class RunNumberHandler : public popcon::PopConSourceHandler<RunNumber>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~RunNumberHandler();
    RunNumberHandler(const edm::ParameterSet& pset); 
 
    
  private:
    std::string m_name;
    unsigned long long m_since;
    
    // for reading from omds 
  
    std::string  m_connectionString;
 
    std::string m_authpath;
    std::string m_host;
    std::string m_sid;
    std::string m_user;
    std::string m_pass;
    int m_port;
};

#endif 
