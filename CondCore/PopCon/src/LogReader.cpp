#include "CondCore/PopCon/interface/LogReader.h"

#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"

//#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"

//static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
popcon::LogReader::LogReader (const std::string& pop_connect) : m_connect(pop_connect) {
}

popcon::LogReader::~LogReader ()
{
}

void  popcon::LogReader::initialize()
{		
}

coral::TimeStamp  popcon::LogReader::lastRun(std::string& name, std::string& cs)
{
  coral::TimeStamp ts;
  //cond::CoralTransaction& coraldb=conHandler.getConnection(m_connect)->coralTransaction();
  session=new cond::DBSession;
  session->configuration().setAuthenticationMethod( cond::XML );
  session->configuration().setMessageLevel( cond::Error );
  cond::Connection con(m_connect,0);
  session->open();
  con.connect(session);
  cond::CoralTransaction& coraldb=con.coralTransaction();
  coraldb.start(true);
  coral::IQuery* query = coraldb.coralSessionProxy().nominalSchema().newQuery();
  query->addToOutputList("max(P_CON_EXECUTION.EXEC_START)","mes");
  query->addToTableList("P_CON_EXECUTION");
  query->addToTableList("P_CON_PAYLOAD_STATE");
  
  std::string condition = "P_CON_EXECUTION.OBJ_ID = P_CON_PAYLOAD_STATE.OBJ_ID AND P_CON_PAYLOAD_STATE.NAME =:nm AND P_CON_PAYLOAD_STATE.CONNECT_STRING =:cs";
  coral::AttributeList conditionData;
  conditionData.extend<std::string>( "nm" );
  conditionData.extend<std::string>( "cs" );
  query->setCondition( condition, conditionData );
  conditionData[0].data<std::string>() = name;
  conditionData[1].data<std::string>() = cs;
  
  query->setMemoryCacheSize( 100 );
  coral::ICursor& cursor = query->execute();
  if( cursor.next() ){
    const coral::AttributeList& row = cursor.currentRow();
    row.toOutputStream( std::cout ) << std::endl;
    ts =  row["mes"].data<coral::TimeStamp>();
    //std::cout << ts.day()  << " "<< ts.month() << " " << ts.year() << " " << ts.hour() << " " << ts.minute() << " " << ts.second() << " " << ts.nanosecond();
    //std::cout << " " << as << " " << id << std::endl;
  }
  cursor.close();
  coraldb.commit();
  con.disconnect();
  delete session;
  return ts;
}
