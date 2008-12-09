// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     OMDSReader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Mar  2 01:46:46 CET 2008
// $Id: OMDSReader.cc,v 1.8 2008/12/04 20:43:49 wsun Exp $
//

// system include files

// user include files
#include "CondTools/L1Trigger/interface/OMDSReader.h"
#include "RelationalAccess/ITableDescription.h"
#include "RelationalAccess/IColumn.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

namespace l1t
{

//
// constructors and destructor
//
  OMDSReader::OMDSReader( const std::string& connectString,
			  const std::string& authenticationPath )
    : DataManager( connectString, authenticationPath, true )
  {
    m_coralTransaction = &( connection->coralTransaction() ) ;
    m_coralTransaction->start( true ) ;
  }

// OMDSReader::OMDSReader(const OMDSReader& rhs)
// {
//    // do actual copying here;
// }

OMDSReader::~OMDSReader()
{
}

//
// assignment operators
//
// const OMDSReader& OMDSReader::operator=(const OMDSReader& rhs)
// {
//   //An exception safe implementation is
//   OMDSReader temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

  const OMDSReader::QueryResults
  OMDSReader::basicQuery(
    const std::vector< std::string >& columnNames,
    const std::string& schemaName,
    const std::string& tableName,
    const std::string& conditionLHS,
    const QueryResults conditionRHS,
    const std::string& conditionRHSName ) const
  {
    coral::ISchema& schema = schemaName.empty() ?
      m_coralTransaction->nominalSchema() :
      m_coralTransaction->coralSessionProxy().schema( schemaName ) ;

    coral::ITable& table = schema.tableHandle( tableName ) ;

    // Pointer is deleted automatically at end of function.
    boost::shared_ptr< coral::IQuery > query( table.newQuery() ) ;

    // Construct query
    std::vector< std::string >::const_iterator it = columnNames.begin() ;
    std::vector< std::string >::const_iterator end = columnNames.end() ;
    for( ; it != end ; ++it )
      {
	query->addToOutputList( *it ) ;
      }

    // Only apply condition if RHS has one row.
    if( !conditionLHS.empty() && conditionRHS.numberRows() == 1 )
      {
	if( !conditionRHSName.empty() )
	  {
	    // Assume all RHS types are strings.
	    coral::AttributeList attList ;
	    attList.extend( conditionRHSName, typeid( std::string ) ) ;
	    std::string tmp ;
	    conditionRHS.fillVariable( conditionRHSName, tmp ) ;
	    attList[ conditionRHSName ].data< std::string >() = tmp ;

	    query->setCondition( conditionLHS + " = :" + conditionRHSName,
				 attList ) ;
	  }
	else if( conditionRHS.columnNames().size() == 1 )
	  // check for only one column
	  {
	    query->setCondition( conditionLHS + " = :" +
				   conditionRHS.columnNames().front(),
				 conditionRHS.attributeLists().front() ) ;
	  }
      }

    coral::ICursor& cursor = query->execute() ;

    // Copy AttributeLists for external use because the cursor is deleted
    // when the query goes out of scope.
    std::vector< coral::AttributeList > atts ;
    while( cursor.next() )
      {
	atts.push_back( cursor.currentRow() ) ;
      } ;

    return QueryResults( columnNames, atts ) ;
  }

  const OMDSReader::QueryResults
  OMDSReader::basicQuery(
    const std::string& columnName,
    const std::string& schemaName,
    const std::string& tableName,
    const std::string& conditionLHS,
    const QueryResults conditionRHS,
    const std::string& conditionRHSName ) const
  {
    std::vector< std::string > columnNames ;
    columnNames.push_back( columnName ) ;
    return basicQuery( columnNames, schemaName, tableName,
			conditionLHS, conditionRHS, conditionRHSName ) ;
  }

  std::vector< std::string >
  OMDSReader::columnNames(
    const std::string& schemaName,
    const std::string& tableName ) const
  {
    coral::ISchema& schema = schemaName.empty() ?
      m_coralTransaction->nominalSchema() :
      m_coralTransaction->coralSessionProxy().schema( schemaName ) ;

    coral::ITable& table = schema.tableHandle( tableName ) ;
    const coral::ITableDescription& tableDesc = table.description() ;

    std::vector< std::string > names ;
    int nCols = tableDesc.numberOfColumns() ;

    for( int i = 0 ; i < nCols ; ++i )
      {
	const coral::IColumn& column = tableDesc.columnDescription( i ) ;
	names.push_back( column.name() ) ;
      }

    return names ;
  }

//
// static member functions
//
}
