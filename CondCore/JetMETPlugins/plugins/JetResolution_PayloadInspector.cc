#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/JetMETObjects/interface/JetResolution.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include <CondFormats/JetMETObjects/interface/Utilities.h>

#include <memory>
#include <sstream>

namespace JME{

  /*******************************************************
 *    
 *         1d histogram of JetResolution in Eta of 1 IOV 
 *
   *******************************************************/

  // inherit from one of the predefined plot class: Histogram1D

  class JetResolutionEta : public cond::payloadInspector::Histogram1D<JetResolutionObject> {
    public:
      static const int MIN_ETA = -5.5;
      static const int MAX_ETA =  5.5;

    JetResolutionEta() : cond::payloadInspector::Histogram1D<JetResolutionObject>( "Jet Energy Resolution", "#eta", 50, MIN_ETA, MAX_ETA, "Resolution"){
      Base::setSingleIov( true );
    }

    // Histogram1D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override{
      std::cout << "filling iovs\n";
      for(size_t idx = 0; idx<50; idx++)
        fillWithBinAndValue(idx, -1.);
      //fillWithValue(-1.2);
      for(auto const &iov: iovs){
        //std::cout << "looping iovs" << std::endl;
        fillWithValue( 1.);
        std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload( std::get<1>(iov) );
        if( payload.get() ){
          //std::cout << "payload fetched" << std::endl;
          fillWithValue( 1.2);
          // test dump
          payload->dump();
          payload->saveToFile("output.txt");
          // looping over the 
//          return true;
        } // payload
      } // iov
      return true;
    } // fill
  };  // class

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( JetResolutionObject ){
  PAYLOAD_INSPECTOR_CLASS( JetResolutionEta );
}

}  // namespace
