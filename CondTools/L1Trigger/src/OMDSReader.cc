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
// $Id: OMDSReader.cc,v 1.1 2008/03/03 21:52:19 wsun Exp $
//

// system include files

// user include files
#include "CondTools/L1Trigger/interface/OMDSReader.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ISchema.h"

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

  boost::shared_ptr< coral::IQuery >
  OMDSReader::newQuery( const std::string& tableString,
			const std::vector< std::string >& queryStrings,
			const std::string& conditionString,
			const coral::AttributeList& conditionAttributes ) const
  {
    // need CoralSessionProxy?
    coral::ITable& table =
      m_coralTransaction->nominalSchema().tableHandle( tableString ) ;

    // Ownership is transferred to calling function
    boost::shared_ptr< coral::IQuery > query( table.newQuery() ) ;

    // Construct query
    std::vector< std::string >::const_iterator it = queryStrings.begin() ;    
    std::vector< std::string >::const_iterator end = queryStrings.end() ;
    for( ; it != end ; ++it )
      {
	query->addToOutputList( *it ) ;
      }

    query->setCondition( conditionString, conditionAttributes ) ;

    return query ;
  }

//
// const member functions
//

//
// static member functions
//
}
