#include "CondCore/CondDB/interface/PayloadReader.h"

namespace cond {

  namespace persistency {

    PayloadReader::PayloadReader(){
      m_connection.reset( new ConnectionPool );
    }
 
    PayloadReader::PayloadReader( const PayloadReader& rhs ):
      m_connection( rhs.m_connection ),
      m_session( rhs.m_session ){
    }

    PayloadReader::~PayloadReader(){
    }
    
    PayloadReader& PayloadReader::operator=( const PayloadReader& rhs ){
      m_connection = rhs.m_connection;
      m_session = rhs.m_session;
      return *this;
    }

    ConnectionPool& PayloadReader::connection(){
      return *m_connection;
    }

    void PayloadReader::open( const std::string& connectionString ){
      m_session = m_connection->createSession( connectionString );
    }

    void PayloadReader::open(){
      open( PRODUCTION_DB );
    }

    void PayloadReader::close(){
      m_session.close();
    }

  }
}
