#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/JsonPrinter.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"

#include <sstream>

namespace {

  class SiStripDetVOff_LV {
  public:
    SiStripDetVOff_LV(){
    }

    // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
    std::string objectType() {
      return "SiStripDetVOff";
    }

    // return a title string to be used in the PayloadInspector
    std::string title() {
      return "Nr of mod with LV OFF vs time";
    }

    std::string info() {
      return title();
    }

    std::string data( const boost::python::list& iovs ){
      cond::persistency::PayloadReader reader;
      // TO DO: add try /catch block                                                                                                                                                    
      reader.open();
      cond::utilities::JsonPrinter jprint("Time","nLVOff");
      for( int i=0; i< len( iovs ); i++ ) {
	cond::Iov_t iov = boost::python::extract<cond::Iov_t>( iovs[i] );
	boost::shared_ptr<SiStripDetVOff> obj = reader.fetch<SiStripDetVOff>( iov.payloadId );
	jprint.append(boost::lexical_cast<std::string>( iov.since ),
		      boost::lexical_cast<std::string>( obj->getLVoffCounts() ),
		      boost::lexical_cast<std::string>( 0. ) );
      }
      return jprint.print();
    }
  };

  class SiStripDetVOff_HV {
  public:
    SiStripDetVOff_HV(){
    }

    // return the type-name of the objects we handle, so the PayloadInspector can find corresponding tags
    std::string objectType() {
      return "SiStripDetVOff";
    }

    // return a title string to be used in the PayloadInspector
    std::string title() {
      return "Nr of mod with HV OFF vs time";
    }

    std::string info() {
      return title();
    }

    std::string data( const boost::python::list& iovs ){
      cond::persistency::PayloadReader reader;
      // TO DO: add try /catch block                                                                                                                                                    
      reader.open();
      cond::utilities::JsonPrinter jprint("Time","nHVOff");
      for( int i=0; i< len( iovs ); i++ ) {
	cond::Iov_t iov = boost::python::extract<cond::Iov_t>( iovs[i] );
	boost::shared_ptr<SiStripDetVOff> obj = reader.fetch<SiStripDetVOff>( iov.payloadId );
	jprint.append(boost::lexical_cast<std::string>( iov.since ),
		      boost::lexical_cast<std::string>( obj->getHVoffCounts() ),
		      boost::lexical_cast<std::string>( 0. ) );
      }
      return jprint.print();
    }
  };

}

PAYLOAD_INSPECTOR_MODULE( SiStrip ){
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_LV );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_HV );
}
