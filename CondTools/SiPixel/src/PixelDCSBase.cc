#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "RelationalAccess/IColumn.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDescription.h"

#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/SiPixel/interface/PixelDCSBase.h"

const std::string PixelDCSBase::theUser  = "CMS_PXL_PIXEL_R";
const std::string PixelDCSBase::theOwner = "CMS_PXL_PVSS_COND";

PixelDCSBase::PixelDCSBase(const edm::ParameterSet& cfg):
  m_connectStr("oracle://"),
  m_tables( cfg.getParameter<Strings>("tables") )
{
  cond::SessionConfiguration& conf = m_dbSession.configuration();

  conf.setAuthenticationMethod(cond::XML);
  conf.setAuthenticationPath( cfg.getParameter<std::string>("authenticationPath") );
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

  coral::ISchema& schema = trans.coralSessionProxy().schema(theOwner);

  typedef std::set<std::string>::const_iterator CIter;

  std::set<std::string> lvTables = schema.listTables();

  CIter last = lvTables.end();

  for (CIter i = lvTables.begin(); i != last; ++i)
  {
    if ( !isLVTable(*i) ) lvTables.erase(i); // remove non-last value tables
  }

  unsigned int nTable = m_tables.size();

  for (unsigned int t = 0; t < nTable; ++t)

    if (lvTables.find(m_tables[t]) == last)
    {
      std::string error = "Cannot find last value table ";

      error += m_tables[t];
      error += "\nAvailable last value tables are:\n";

      for (CIter i = lvTables.begin(); i != last; ++i) (error += *i) += '\n';

      throw cms::Exception("PixelDCSBase") << error;
    }

  coral::IQuery* query = schema.newQuery();

  std::string condition = "A.DPE_NAME = D.DPNAME || '.'";

  for (unsigned int t = 0; t < nTable; ++t)
  {
    const std::string& table = m_tables[t];

    condition += " and " + table + ".DPID = D.ID";

    // the last values are contained in the 2nd column of each table
    query->addToOutputList( schema.tableHandle(table).description().columnDescription(1).name() );
    query->addToTableList(table);
  }

  query->addToOutputList("A.ALIAS", "name");
  query->addToTableList("DP_NAME2ID", "D");
  query->addToTableList("ALIASES", "A");
  query->setCondition( condition, coral::AttributeList() );

  fillObject( query->execute() );

  delete query;
  dbConn.disconnect();
}
