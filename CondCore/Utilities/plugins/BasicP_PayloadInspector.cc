#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/JsonPrinter.h"
#include "CondFormats/Common/interface/BasicPayload.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"
#include <memory>
#include <sstream>

namespace {

  class BasicPayloadPlot_data0 {
  public:
    BasicPayloadPlot_data0(){
    }

    // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
    std::string objectType() {
      return "BasicPayload";
    }

    // return a title string to be used in the PayloadInspector
    std::string title() {
      return "Data0 vs run number";
    }

    std::string info() {
      return title();
    }

    std::string data( const boost::python::list& iovs ){
      cond::persistency::PayloadReader reader;
      // TO DO: add try /catch block                                                                                                                                                   
      cond::utilities::JsonPrinter jprint("Run","data0");
      for( int i=0; i< len( iovs ); i++ ) {
	cond::Iov_t iov = boost::python::extract<cond::Iov_t>( iovs[i] );
	std::shared_ptr<cond::BasicPayload> obj = reader.fetch<cond::BasicPayload>( iov.payloadId );
	jprint.append(boost::lexical_cast<std::string>(iov.since),boost::lexical_cast<std::string>(obj->m_data0 ));
      }
      return jprint.print();
    }
  };

  class BasicPayloadPlot_data1 {
  public:
    BasicPayloadPlot_data1(){
    }

    // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
    std::string objectType() {
      return "BasicPayload";
    }

    // return a title string to be used in the PayloadInspector
    std::string title() {
      return "Data1 trend";
    }

    std::string info() {
      return title();
    }

    std::string data( const boost::python::list& iovs ){
      cond::persistency::PayloadReader reader;
      // TO DO: add try /catch block                                                                                                                                                 
      cond::utilities::JsonPrinter jprint("Run","data1");
      for( int i=0; i< len( iovs ); i++ ) {
	cond::Iov_t iov = boost::python::extract<cond::Iov_t>( iovs[i] );
	std::shared_ptr<cond::BasicPayload> obj = reader.fetch<cond::BasicPayload>( iov.payloadId );
        jprint.append(boost::lexical_cast<std::string>(iov.since),boost::lexical_cast<std::string>(obj->m_data1 ));
      }
      return jprint.print();
    }
  };

}

PAYLOAD_INSPECTOR_MODULE( BasicPayload ){
  PAYLOAD_INSPECTOR_CLASS( BasicPayloadPlot_data0 );
  PAYLOAD_INSPECTOR_CLASS( BasicPayloadPlot_data1 );
}
