#ifndef CondCore_CondDB_PayloadReader_h
#define CondCore_CondDB_PyloadReader_h
//
// Package:     CondDB
// Class  :     PayloadReader
// 
/**\class PayloadReader PayloadReader.h CondCore/CondDB/interface/PayloadReader.h
   Description: service for accessing conditions payloads from DB.  
*/
//
// Author:      Giacomo Govi
// Created:     Jul 2015
//

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include <memory>

namespace cond {

  namespace persistency {

    class PayloadReader {
    public:

      //static constexpr const char* const PRODUCTION_DB = "oracle://cms_orcon_adg/CMS_CONDITIONS";
      static constexpr const char* const PRODUCTION_DB = "oracle://cms_orcoff_prep/CMS_CONDITIONS_002";

    public:

      // default constructor
      PayloadReader();
      
      // 
      PayloadReader( const PayloadReader& rhs );
      
      // 
      virtual ~PayloadReader();
      
      //
      PayloadReader& operator=( const PayloadReader& rhs );

      //
      ConnectionPool& connection();

      //
      void open( const std::string& connectionString );

      //
      void open();
      
      // 
      void close();
      
      //
      template <typename T> boost::shared_ptr<T> fetch( const cond::Hash& payloadHash );
      
   private:
      
      std::shared_ptr<ConnectionPool> m_connection;
      Session m_session;
    };
        
    template <typename T> inline boost::shared_ptr<T> PayloadReader::fetch( const cond::Hash& payloadHash ){
      boost::shared_ptr<T> ret;
      if(m_session.connectionString().empty()) open( PRODUCTION_DB );
      m_session.transaction().start( true );
      ret = m_session.fetchPayload<T>( payloadHash );
      m_session.transaction().commit();
      return ret;
    }

  }
}
#endif
