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
// $Id: OMDSReader.h,v 1.1 2008/03/03 21:52:10 wsun Exp $
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondTools/L1Trigger/interface/DataManager.h"
#include "RelationalAccess/IQuery.h"

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
      boost::shared_ptr< coral::IQuery >
	newQuery( const std::string& tableString,
		  const std::vector< std::string >& queryStrings,
		  const std::string& conditionString,
		  const coral::AttributeList& conditionAttributes ) const ;

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
