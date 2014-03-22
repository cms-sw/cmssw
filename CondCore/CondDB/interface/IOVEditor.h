#ifndef CondCore_CondDB_IOVEditor_h
#define CondCore_CondDB_IOVEditor_h
//
// Package:     CondDB
// Class  :     IOVEditor
// 
/**\class IOVEditor IOVEditor.h CondCore/CondDB/interface/IOVEditor.h
   Description: service for update access to the condition IOVs.  
*/
//
// Author:      Giacomo Govi
// Created:     Apr 2013
//

#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class SessionImpl;
    class IOVEditorData;

    // value semantics...
    class IOVEditor {
    public:

      IOVEditor();
      // ctor
      explicit IOVEditor( const std::shared_ptr<SessionImpl>& session );

      // ctor used after new tag creation. the specified params are assumed and passed directly to the object.
      IOVEditor( const std::shared_ptr<SessionImpl>& session, const std::string& tag, cond::TimeType timeType, 
		 const std::string& payloadType, cond::SynchronizationType synchronizationType );

      //
      IOVEditor( const IOVEditor& rhs );
      
      //
      IOVEditor& operator=( const IOVEditor& rhs );
      
      // loads to tag to edit
      void load( const std::string& tag );
      
      // read only getters. they could be changed to return references...
      std::string tag() const;
      cond::TimeType timeType() const;
      std::string payloadType() const;
      cond::SynchronizationType synchronizationType() const;
      
      // getters/setters for the updatable parameters 
      cond::Time_t endOfValidity() const;
      void setEndOfValidity( cond::Time_t validity );
      
      std::string description() const;
      void setDescription( const std::string& description );
      
      cond::Time_t lastValidatedTime() const;
      void setLastValidatedTime( cond::Time_t time );  
      
      // register a new insertion.
      // if checkType==true, the payload corresponding to the specified id is verified to be the same type as the iov payloadObjectType 
      void insert( cond::Time_t since, const cond::Hash& payloadHash, bool checkType=false );
      void insert( cond::Time_t since, const cond::Hash& payloadHash, const boost::posix_time::ptime& insertionTime, bool checkType=false ); 

      // execute the update/intert queries and reset the buffer
      bool flush();
      bool flush( const boost::posix_time::ptime& operationTime );
      
    private:
      void checkTransaction( const std::string& ctx );
      
    private:
      std::shared_ptr<IOVEditorData> m_data;
      std::shared_ptr<SessionImpl> m_session;

    };

  }
}

#endif

