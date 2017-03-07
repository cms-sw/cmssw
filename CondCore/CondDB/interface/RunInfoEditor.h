#ifndef CondCore_CondDB_RunInfoEditor_h
#define CondCore_CondDB_RunInfoEditor_h
//
// Package:     CondDB
// Class  :     RunInfoEditor
// 
/**\class RunInfoEditor RunInfoEditor.h CondCore/CondDB/interface/RunInfoEditor.h
   Description: service for update access to the runInfo entries.  
*/
//
// Author:      Giacomo Govi
// Created:     March 2017
//

#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class SessionImpl;
    class RunInfoEditorData;

    // value semantics...
    class RunInfoEditor {
    public:

      RunInfoEditor();
      // ctor
      explicit RunInfoEditor( const std::shared_ptr<SessionImpl>& session );

      //
      RunInfoEditor( const RunInfoEditor& rhs );
      
      //
      RunInfoEditor& operator=( const RunInfoEditor& rhs );

      //
      void init();
      
      // register an insertion.
      void insert( cond::Time_t runNumber, const boost::posix_time::ptime& start, const boost::posix_time::ptime& end );

      // register a new insertion.
      void insertNew( cond::Time_t runNumber, const boost::posix_time::ptime& start );

      // set the end of a new started run
      void updateEnd( cond::Time_t runNumber, const boost::posix_time::ptime& end );

      // execute the update/intert queries and reset the buffer
      bool flush();

    private:
      void checkTransaction( const std::string& ctx );
      
    private:
      std::shared_ptr<RunInfoEditorData> m_data;
      std::shared_ptr<SessionImpl> m_session;

    };

  }
}

#endif

