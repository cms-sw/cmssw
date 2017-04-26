#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include <memory>
#include <sstream>



namespace {

  /*******************************************************
   
     2d histogram of ECAL barrel channel status of 1 IOV 

  *******************************************************/


  // inherit from one of the predefined plot class: Histogram2D
  class EcalChannelStatusEBMap : public cond::payloadInspector::Histogram2D<EcalChannelStatus> {

  public:
    EcalChannelStatusEBMap() : cond::payloadInspector::Histogram2D<EcalChannelStatus>( "ECAL Barrel channel status - map ",
										       "iphi", 360, 1., 361., "ieta", 171, -85., 86.) {
     Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {
	std::shared_ptr<EcalChannelStatus> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL channel status, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].getEncodedStatusCode()) continue;

	    // fill the Histogram2D here
	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), (*payload)[rawid].getEncodedStatusCode());
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)

      return true;
    }// fill method
  };

  class EcalChannelStatusEEMap : public cond::payloadInspector::Histogram2D<EcalChannelStatus> {

  public:
    EcalChannelStatusEEMap() : cond::payloadInspector::Histogram2D<EcalChannelStatus>( "ECAL Barrel channel status - map ",
										       "ix", 220, 1, 221, "iy", 100, 1, 101) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {
	std::shared_ptr<EcalChannelStatus> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL channel status, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].getEncodedStatusCode()) continue;
		    if(iz == 0)
		      fillWithValue( ix, iy, (*payload)[rawid].getEncodedStatusCode());
		    else
		      fillWithValue( ix + 120, iy, (*payload)[rawid].getEncodedStatusCode());

		}  // validDetId 
	} // payload
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( EcalChannelStatus ){
  PAYLOAD_INSPECTOR_CLASS( EcalChannelStatusEBMap );
  PAYLOAD_INSPECTOR_CLASS( EcalChannelStatusEEMap );
}
