#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondFormats/Common/interface/BasicPayload.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"
#include <memory>
#include <sstream>

namespace {

  class BasicPayload_data0 : public cond::payloadInspector::HistoryPlot<cond::BasicPayload,float> {
  public:
    BasicPayload_data0() : cond::payloadInspector::HistoryPlot<cond::BasicPayload,float>( "Example Trend", "data0"){
    }

    float getFromPayload( cond::BasicPayload& payload ){
      return payload.m_data0;
    }
  };

  class BasicPayload_data1 : public cond::payloadInspector::RunHistoryPlot<cond::BasicPayload,float> {
  public:
    BasicPayload_data1() : cond::payloadInspector::RunHistoryPlot<cond::BasicPayload,float>( "Example Run-based Trend", "data0"){
    }

    float getFromPayload( cond::BasicPayload& payload ){
      return payload.m_data0;
    }
  };

  class BasicPayload_data2 : public cond::payloadInspector::TimeHistoryPlot<cond::BasicPayload,float> {
  public:
    BasicPayload_data2() : cond::payloadInspector::TimeHistoryPlot<cond::BasicPayload,float>( "Example Time-based Trend", "data0"){
    }

    float getFromPayload( cond::BasicPayload& payload ){
      return payload.m_data0;
    }
  };

  class BasicPayload_data3 : public cond::payloadInspector::ScatterPlot<cond::BasicPayload,float,float> {
  public:
    BasicPayload_data3() : cond::payloadInspector::ScatterPlot<cond::BasicPayload,float,float>( "Example Scatter", "data0","data1"){
    }

    std::tuple<float,float> getFromPayload( cond::BasicPayload& payload ){
      return std::make_tuple(payload.m_data0,payload.m_data1);
    }
  };

  class BasicPayload_data4 : public cond::payloadInspector::Histogram1D<cond::BasicPayload> {
  public:
    BasicPayload_data4() : cond::payloadInspector::Histogram1D<cond::BasicPayload>( "Example Histo1d", "x",10,0,10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<cond::BasicPayload> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  for( size_t j=0;j<100;j++ ) {
	    fillWithValue( j, payload->m_vec[j] );
	  }
	}
      }
      return true;
    }
  };

  class BasicPayload_data5 : public cond::payloadInspector::Histogram2D<cond::BasicPayload> {
  public:
    BasicPayload_data5() : cond::payloadInspector::Histogram2D<cond::BasicPayload>( "Example Histo2d", "x",10,0,10,"y",10,0,10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<cond::BasicPayload> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
          for( size_t i=0;i<10;i++ )
	    for( size_t j=0;j<10;j++ ) {
	      fillWithValue( j, i, payload->m_vec[i*10+j] );
	    }
	}
      }
      return true;
    }
  };


}

PAYLOAD_INSPECTOR_MODULE( BasicPayload ){
  PAYLOAD_INSPECTOR_CLASS( BasicPayload_data0 );
  PAYLOAD_INSPECTOR_CLASS( BasicPayload_data1 );
  PAYLOAD_INSPECTOR_CLASS( BasicPayload_data2 );
  PAYLOAD_INSPECTOR_CLASS( BasicPayload_data3 );
  PAYLOAD_INSPECTOR_CLASS( BasicPayload_data4 );
  PAYLOAD_INSPECTOR_CLASS( BasicPayload_data5 );
}
