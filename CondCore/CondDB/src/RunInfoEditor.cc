#include "CondCore/CondDB/interface/RunInfoEditor.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "SessionImpl.h"
//

namespace cond {

  namespace persistency {

    // implementation details. holds only data.
    class RunInfoEditorData {
    public:
      explicit RunInfoEditorData():
	runBuffer(),
        updateBuffer(){
      }
      // update buffers
      std::vector<std::tuple<cond::Time_t,boost::posix_time::ptime,boost::posix_time::ptime> > runBuffer;
      std::vector<std::pair<cond::Time_t,boost::posix_time::ptime> > updateBuffer;
    };

    RunInfoEditor::RunInfoEditor():
      m_data(),
      m_session(){
    }

    RunInfoEditor::RunInfoEditor( const std::shared_ptr<SessionImpl>& session ):
      m_data( new RunInfoEditorData ),
      m_session( session ){
      
    }

    RunInfoEditor::RunInfoEditor( const RunInfoEditor& rhs ):
      m_data( rhs.m_data ),
      m_session( rhs.m_session ){
    }
    
    RunInfoEditor& RunInfoEditor::operator=( const RunInfoEditor& rhs ){
      m_data = rhs.m_data;
      m_session = rhs.m_session;
      return *this;
    }

    void  RunInfoEditor::init(){
      if( m_data.get() ){
	checkTransaction( "RunInfoEditor::init" );
	if( !m_session->runInfoSchema().exists() ) m_session->runInfoSchema().create();
      }
    }
    
    void RunInfoEditor::insert( cond::Time_t runNumber, const boost::posix_time::ptime& start, const boost::posix_time::ptime& end ){
      if( m_data.get() ) m_data->runBuffer.push_back( std::tie( runNumber, start, end ) );
    }
    
    void RunInfoEditor::insertNew( cond::Time_t runNumber, const boost::posix_time::ptime& start){
      if( m_data.get() ) m_data->runBuffer.push_back( std::tie( runNumber, start, start ) );
    }

    void RunInfoEditor::updateEnd( cond::Time_t runNumber, const boost::posix_time::ptime& end ){
      if( m_data.get() ) m_data->updateBuffer.push_back( std::make_pair( runNumber, end ) );
    }

    bool RunInfoEditor::flush(){
      bool ret = false;
      if( m_data.get() ){
	checkTransaction( "RunInfoEditor::flush" );
	m_session->runInfoSchema().runInfoTable().insert( m_data->runBuffer );
        for( auto update: m_data->updateBuffer ) m_session->runInfoSchema().runInfoTable().updateEnd( update.first, update.second );
        m_data->runBuffer.clear();
        m_data->updateBuffer.clear();
        ret = true;
      }
      return ret;
    }
    
    void RunInfoEditor::checkTransaction( const std::string& ctx ){
      if( !m_session.get() ) throwException("The session is not active.",ctx );
      if( !m_session->isTransactionActive( false ) ) throwException("The transaction is not active.",ctx );
    }
    
  }
}
