#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"

#include <sstream>
#include <iostream>

namespace cond {

  namespace payloadInspector {

    PlotAnnotations::PlotAnnotations():m(){}

    std::string PlotAnnotations::get( const std::string& key ) const{
      std::string ret("");
      auto im = m.find( key );
      if( im != m.end() ) ret = im->second;
      return ret;
    } 

    constexpr const char* const ModuleVersion::label;

    PlotBase::PlotBase():
        m_plotAnnotations(),m_data(""){
    }

    std::string PlotBase::payloadType() const {
      return m_plotAnnotations.get(PlotAnnotations::PAYLOAD_TYPE_K);
    }

    std::string PlotBase::title() const {
      return m_plotAnnotations.get(PlotAnnotations::TITLE_K);
    }

    std::string PlotBase::type() const {
      return m_plotAnnotations.get(PlotAnnotations::PLOT_TYPE_K);
    }

    bool PlotBase::isSingleIov() const {
      return m_plotAnnotations.singleIov;
    }

    std::string PlotBase::data() const {
      return m_data;
    }

    bool PlotBase::process( const std::string& connectionString,  const std::string& tag, const std::string& timeType, cond::Time_t begin, cond::Time_t end ){
      init();
      m_tagTimeType = cond::time::timeTypeFromName(timeType);
      cond::persistency::ConnectionPool connection;
      m_dbSession = connection.createSession( connectionString );
      m_dbSession.transaction().start();
      std::vector<std::tuple<cond::Time_t,cond::Hash> > iovs;
      m_dbSession.getIovRange( tag, begin, end, iovs );
      m_data = processData( iovs );
      m_dbSession.transaction().commit();
      // fixme...
      return true;
    }

    void PlotBase::init(){
    }

    std::string PlotBase::processData( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& ){
      return ""; 
    }

    void PlotBase::setSingleIov( bool flag ) {
      m_plotAnnotations.singleIov = flag;
    }

    cond::TimeType PlotBase::tagTimeType() const {
      return m_tagTimeType;
    }

    cond::persistency::Session PlotBase::dbSession(){
      return m_dbSession;
    }

  }

}
