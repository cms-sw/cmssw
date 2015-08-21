#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/JsonPrinter.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <sstream>

namespace {

  class BeamSpotPlot_x {
  public:
    BeamSpotPlot_x(){
    }

    // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
    std::string objectType() {
      return "BeamSpotObjects";
    }

    // return a title string to be used in the PayloadInspector
    std::string title() {
      return "x vs run number";
    }

    std::string info() {
      return title();
    }

    std::string data( const boost::python::list& iovs ){
      cond::persistency::PayloadReader reader;
      // TO DO: add try /catch block                                                                                                                                                   
      cond::utilities::JsonPrinter jprint("Run","x");
      for( int i=0; i< len( iovs ); i++ ) {
	cond::Iov_t iov = boost::python::extract<cond::Iov_t>( iovs[i] );
	boost::shared_ptr<BeamSpotObjects> obj = reader.fetch<BeamSpotObjects>( iov.payloadId );
	jprint.append(boost::lexical_cast<std::string>( iov.since ),
		      boost::lexical_cast<std::string>( obj->GetX() ),
		      boost::lexical_cast<std::string>( obj->GetXError() ) );
      }
      return jprint.print();
    }
  };

  class BeamSpotPlot_y {
  public:
    BeamSpotPlot_y(){
    }

    // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
    std::string objectType() {
      return "BeamSpotObjects";
    }

    // return a title string to be used in the PayloadInspector
    std::string title() {
      return "y vs run number";
    }

    std::string info() {
      return title();
    }

    std::string data( const boost::python::list& iovs ){
      cond::persistency::PayloadReader reader;
      // TO DO: add try /catch block                                                                                                                                                   
      cond::utilities::JsonPrinter jprint("Run","y");
      for( int i=0; i< len( iovs ); i++ ) {
	cond::Iov_t iov = boost::python::extract<cond::Iov_t>( iovs[i] );
	boost::shared_ptr<BeamSpotObjects> obj = reader.fetch<BeamSpotObjects>( iov.payloadId );
	jprint.append(boost::lexical_cast<std::string>( iov.since ),
		      boost::lexical_cast<std::string>( obj->GetY() ),
		      boost::lexical_cast<std::string>( obj->GetYError() ) );
      }
      return jprint.print();
    }
  };

  class BeamSpotPlot_xy {
  public:
    BeamSpotPlot_xy(){
    }

    // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
    std::string objectType() {
      return "BeamSpotObjects";
    }

    // return a title string to be used in the PayloadInspector
    std::string title() {
      return "BeamSpot x vs y";
    }

    std::string info() {
      return title();
    }

    std::string data( const boost::python::list& iovs ){
      cond::persistency::PayloadReader reader;
      // TO DO: add try /catch block                                                                                                                                                   
      cond::utilities::JsonPrinter jprint("x","y");
      for( int i=0; i< len( iovs ); i++ ) {
	cond::Iov_t iov = boost::python::extract<cond::Iov_t>( iovs[i] );
	boost::shared_ptr<BeamSpotObjects> obj = reader.fetch<BeamSpotObjects>( iov.payloadId );
	jprint.append(boost::lexical_cast<std::string>( obj->GetX() ), 
		      boost::lexical_cast<std::string>( obj->GetY() ) );
      }
      return jprint.print();
    }
  };


}

PAYLOAD_INSPECTOR_MODULE( BeamSpot ){
  PAYLOAD_INSPECTOR_CLASS( BeamSpotPlot_x );
  PAYLOAD_INSPECTOR_CLASS( BeamSpotPlot_y );
  PAYLOAD_INSPECTOR_CLASS( BeamSpotPlot_xy );
}
