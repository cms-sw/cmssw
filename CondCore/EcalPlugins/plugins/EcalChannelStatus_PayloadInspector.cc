#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <memory>
#include <sstream>



namespace {

  /*******************************************************
   
     2d histogram of ECAL barrel channel status of 1 IOV 

  *******************************************************/


  // inherit from one of the predefined plot class: Histogram2D
  class EcalChannelStatusEBMap : public cond::payloadInspector::Histogram2D<EcalChannelStatus> {

  public:
      static const int MIN_IETA = 1;
      static const int MIN_IPHI = 1;
      static const int MAX_IETA = 85;
      static const int MAX_IPHI = 360;

    EcalChannelStatusEBMap() : cond::payloadInspector::Histogram2D<EcalChannelStatus>( "ECAL Barrel channel status - map ",
										       "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {
     Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for ( auto const & iov: iovs) {
	std::shared_ptr<EcalChannelStatus> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  // set to -1 for ieta 0 (no crystal)
	  for(int iphi = MIN_IPHI; iphi < MAX_IPHI+1; iphi++) fillWithValue(iphi, 0, -1);
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL channel status, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    //	    if (!(*payload)[rawid].getEncodedStatusCode()) continue;
	    float weight = (float)(*payload)[rawid].getEncodedStatusCode();

	    // fill the Histogram2D here
	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), weight);
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)

      return true;
    }// fill method
  };

  class EcalChannelStatusEEMap : public cond::payloadInspector::Histogram2D<EcalChannelStatus> {

  public:
    static const int IX_MIN =1;
  
  /** Lower bound of EE crystal y-index
   */
    static const int IY_MIN =1;
  
  /** Upper bound of EE crystal y-index
   */
    static const int IX_MAX =100;
  
  /** Upper bound of EE crystal y-index
   */
    static const int IY_MAX =100;
    EcalChannelStatusEEMap() : cond::payloadInspector::Histogram2D<EcalChannelStatus>( "ECAL Endcap channel status - map ",
										       "ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for ( auto const & iov: iovs) {
	std::shared_ptr<EcalChannelStatus> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // set to -1 everywhwere
	  for(int ix = IX_MIN; ix < 2.2*IX_MAX+1; ix++)
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      fillWithValue(ix, iy, -1);
	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL channel status, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    //		    if (!(*payload)[rawid].getEncodedStatusCode()) continue;
		    float weight = (float)(*payload)[rawid].getEncodedStatusCode();
		    if(iz == -1)
		      fillWithValue( ix, iy, weight);
		    else
		      fillWithValue( ix+IX_MAX+20, iy, weight);

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
