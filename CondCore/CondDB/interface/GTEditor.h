#ifndef CondCore_CondDB_GTEditor_h
#define CondCore_CondDB_GTEditor_h
//
// Package:     CondDB
// Class  :     GTEditor
// 
/**\class GTEditor GTEditor.h CondCore/CondDB/interface/GTEditor.h
   Description: service for update access to the condition IOVs.  
*/
//
// Author:      Giacomo Govi
// Created:     Jul 2013
//

#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class SessionImpl;
    class GTEditorData;

    // value semantics...
    class GTEditor {
    public:
      
      // ctor
      explicit GTEditor( const std::shared_ptr<SessionImpl>& session );
      
      // ctor used after new tag creation. the specified params are assumed and passed directly to the object.
      GTEditor( const std::shared_ptr<SessionImpl>& session, const std::string& gtName );
      
      //
      GTEditor( const GTEditor& rhs );
      
      //
      GTEditor& operator=( const GTEditor& rhs );
      
      // loads to tag to edit
      void load( const std::string& gtName );
      
      // read only getters. they could be changed to return references...
      std::string name() const;
      
      // getters/setters for the updatable parameters 
      cond::Time_t validity() const;
      void setValidity( cond::Time_t validity );
      
      std::string description() const;
      void setDescription( const std::string& description );
      
      std::string release() const;
      void setRelease( const std::string& release );
      
      boost::posix_time::ptime snapshotTime() const;
      void setSnapshotTime( const boost::posix_time::ptime& snapshotTime );
      
      // register a new insertion.
      // if checkType==true, the object type declared for the tag is verified to be the same type as the iov payloadObjectType record object type.
      void insert( const std::string& recordName, const std::string& tagName, bool checkType=false );
      void insert( const std::string& recordName, const std::string& recordLabel, const std::string& tagName, bool checkType=false );
      
      // execute the update/intert queries and reset the buffer
      bool flush();
      // execute the update/intert queries and reset the buffer
      bool flush( const boost::posix_time::ptime& operationTime );
    private:
      void checkTransaction( const std::string& ctx );
      
    private:
      std::shared_ptr<GTEditorData> m_data;
      std::shared_ptr<SessionImpl> m_session;
      
    };

  }
}

#endif

