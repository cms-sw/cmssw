#include "CoralBase/Attribute.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ISessionProxy.h"

#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/SiPixel/interface/PixelDCSBase.h"

const std::string PixelDCSBase::theUser  = "CMS_PXL_PIXEL_R";
const std::string PixelDCSBase::theOwner = "CMS_PXL_PVSS_COND";

PixelDCSBase::PixelDCSBase(const edm::ParameterSet& cfg):
  m_connectStr("oracle://"),
  m_table( cfg.getParameter<std::string>("table") ),
  m_column( cfg.getParameter<std::string>("column") )
{
  cond::SessionConfiguration& conf = m_dbSession.configuration();

  conf.setAuthenticationMethod(cond::XML);
  conf.setAuthenticationPath( cfg.getUntrackedParameter<std::string>("authenticationPath", ".") );
  conf.setMessageLevel(cond::Error);

  m_connectStr += cfg.getParameter<std::string>("dbName");
  m_connectStr += '/';
  m_connectStr += theUser;
}

void PixelDCSBase::getData()
{
  m_dbSession.open();

  cond::Connection dbConn(m_connectStr);

  dbConn.connect(&m_dbSession);

  cond::CoralTransaction& trans = dbConn.coralTransaction();

  trans.start(true);

  const coral::ISchema& schema = trans.coralSessionProxy().schema(theOwner);

  typedef std::set<std::string> Strings;
  typedef Strings::const_iterator CIter;

  Strings tables = schema.listTables();

  CIter last = tables.end();

  for (CIter i = tables.begin(); i != last; ++i)
  {
    if ( !isLVTable(*i) ) tables.erase(i);
  }

  if (tables.find(m_table) == last)
  {
    std::string error = "Cannot find last value table ";

    error += m_table;
    error += "\nAvailable last value tables are:\n";

    for (CIter i = tables.begin(); i != last; ++i) (error += *i) += '\n';

    throw cms::Exception("PixelDCSBase") << error;
  }

  coral::IQuery* query = schema.newQuery();

  coral::AttributeList output = outputDefn();

  output.extend("name", typeid(std::string) ); 

  query->defineOutput(output);
  query->addToTableList("DP_NAME2ID", "D");
  query->addToTableList("ALIASES", "A");
  query->addToTableList(m_table, "T");
  query->addToOutputList( "T." + m_column, output[0].specification().name() );
  query->addToOutputList("A.ALIAS", "name");
  query->setCondition( "T.DPID = D.ID and A.DPE_NAME = D.DPNAME || '.'",
      coral::AttributeList() );

  coral::ICursor& cursor = query->execute();

  fillObject(cursor);

  cursor.close();
  delete query;
  dbConn.disconnect();
}
