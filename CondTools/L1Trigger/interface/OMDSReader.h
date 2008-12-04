#ifndef CondTools_L1Trigger_OMDSReader_h
#define CondTools_L1Trigger_OMDSReader_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     OMDSReader
// 
/**\class OMDSReader OMDSReader.h CondTools/L1Trigger/interface/OMDSReader.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Sun Mar  2 01:36:06 CET 2008
// $Id: OMDSReader.h,v 1.3 2008/07/10 20:58:01 wsun Exp $
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondTools/L1Trigger/interface/DataManager.h"
#include "RelationalAccess/IQuery.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

// forward declarations

namespace l1t
{

  class OMDSReader : public DataManager
  {

  public:
    OMDSReader( const std::string& connectString,
		const std::string& authenticationPath ) ;

    virtual ~OMDSReader();

      // ---------- const member functions ---------------------

      // std::vector< std::string > is the list of attribute names.
      // We need this typedef because there is no way to ask AttributeList
      // for its attribute names.  We have a vector of AttributeLists because
      // the query may return more than one row, each of which is encapsulated
      // in an AttributeList.
      typedef
	std::pair< std::vector< std::string >,
	std::vector< coral::AttributeList > >
	QueryResults ;

      // These functions encapsulate basic SQL queries of the form
      //
      // SELECT <columns> FROM <schema.table> WHERE <conditionLHS> = <conditionRHS>
      //
      // where
      //
      // <columns> can be one or many column names
      // <conditionRHS> can be a string or the result of another query

      const QueryResults basicQuery(
	const std::vector< std::string >& columnNames,
	const std::string& schemaName, // for nominal schema, use ""
	const std::string& tableName,
	const std::string& conditionLHS = "",
	const QueryResults conditionRHS = QueryResults(),
	                                           // must have only one row
	const std::string& conditionRHSName = ""
	                 // if empty, conditionRHS must have only one column
	) const ;

      const QueryResults basicQuery(
	const std::string& columnName,
	const std::string& schemaName, // for nominal schema, use ""
	const std::string& tableName,
	const std::string& conditionLHS = "",
	const QueryResults conditionRHS = QueryResults(),
	                                           // must have only one row
	const std::string& conditionRHSName = ""
	                 // if empty, conditionRHS must have only one column
	) const ;

      const QueryResults singleAttribute( const std::string& data ) const ;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      OMDSReader(const OMDSReader&); // stop default

      const OMDSReader& operator=(const OMDSReader&); // stop default

      // ---------- member data --------------------------------
      cond::CoralTransaction* m_coralTransaction ;
};

}
#endif
