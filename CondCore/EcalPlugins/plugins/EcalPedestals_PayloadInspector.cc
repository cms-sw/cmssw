#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

#include <memory>
#include <sstream>



namespace {



  /************************************************
   
     1d histogram of ECAL barrel pedestal of 1 IOV 

  *************************************************/


  // inherit from one of the predefined plot class: Histogram1D
  class EcalPedestalEBM12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEBM12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel pedestal average gain12",
									      "ECAL Barrel pedestal average gain12", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->barrelItems().size()) return false;

	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
	    
	    // to be used to fill the histogram
	    fillWithValue( (*payload)[rawid].mean_x12 );

	  }// loop over EB cells 
	}// payload
      }// iovs

      return true;
    }// fill
  };

  class EcalPedestalEBM6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEBM6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel pedestal average gain6",
									     "ECAL Barrel pedestal average gain6", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {

	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->barrelItems().size()) return false;

	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
	    
	    // to be used to fill the histogram
	    fillWithValue( (*payload)[rawid].mean_x6);

	  }// loop over EB cells 
	}// payload
      }// iovs

      return true;
    }// fill
  };

  class EcalPedestalEBM1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEBM1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Barrel pedestal average gain1",
									    "ECAL Barrel pedestal average gain1", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->barrelItems().size()) return false;
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
	    
	    // to be used to fill the histogram
	    fillWithValue( (*payload)[rawid].mean_x1);

	  }// loop over EB cells 
	}// payload
      }// iovs
      return true;
    }// fill
  };

  class EcalPedestalEEM12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEEM12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap pedestal average gain12", 
									      "ECAL Endcap pedestal average gain12", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    fillWithValue( (*payload)[rawid].mean_x12 );

		}  // validDetId 
	} // payload
      }  // loop over iovs

      return true;
    }// fill
  };

  class EcalPedestalEEM6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEEM6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap pedestal average gain6",
									     "ECAL Endcap pedestal average gain6", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {

	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    fillWithValue( (*payload)[rawid].mean_x6);

		}  // validDetId 
	} // payload
      }  // loop over iovs

      return true;
    }// fill
  };

  class EcalPedestalEEM1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEEM1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Endcap pedestal average gain1", 
									    "ECAL Endcap pedestal average gain1", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
		    fillWithValue( (*payload)[rawid].mean_x1);

		}  // validDetId 
	} // payload
      }  // loop over iovs
      return true;
    }// fill
  };

 class EcalPedestalEBR12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEBR12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel noise average gain12",
									      "ECAL Barrel noise average gain12", 100, 0, 10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->barrelItems().size()) return false;

	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
	    
	    // to be used to fill the histogram
	    fillWithValue( (*payload)[rawid].rms_x12 );

	  }// loop over EB cells 
	}// payload
      }// iovs

      return true;
    }// fill
  };

  class EcalPedestalEBR6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEBR6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel noise average gain6",
									     "ECAL Barrel noise average gain6", 100, 0, 10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {

	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->barrelItems().size()) return false;

	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
	    
	    // to be used to fill the histogram
	    fillWithValue( (*payload)[rawid].rms_x6);

	  }// loop over EB cells 
	}// payload
      }// iovs

      return true;
    }// fill
  };

  class EcalPedestalEBR1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEBR1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Barrel noise average gain1",
									    "ECAL Barrel noise average gain1", 100, 0, 10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->barrelItems().size()) return false;
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
	    
	    // to be used to fill the histogram
	    fillWithValue( (*payload)[rawid].rms_x1);

	  }// loop over EB cells 
	}// payload
      }// iovs
      return true;
    }// fill
  };

  class EcalPedestalEER12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEER12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap noise average gain12", 
									      "ECAL Endcap noise average gain12", 100, 0, 10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    fillWithValue( (*payload)[rawid].rms_x12 );

		}  // validDetId 
	} // payload
      }  // loop over iovs

      return true;
    }// fill
  };

  class EcalPedestalEER6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEER6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap noise average gain6",
									     "ECAL Endcap noise average gain6", 100, 0, 10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {

	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    fillWithValue( (*payload)[rawid].rms_x6);

		}  // validDetId 
	} // payload
      }  // loop over iovs

      return true;
    }// fill
  };

  class EcalPedestalEER1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalEER1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Endcap noise average gain1", 
									    "ECAL Endcap noise average gain1", 100, 0, 10){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
		    fillWithValue( (*payload)[rawid].rms_x1);

		}  // validDetId 
	} // payload
      }  // loop over iovs
      return true;
    }// fill
  };

  /*************************************************
   
     2d histogram of ECAL barrel pedestal of 1 IOV 

  *************************************************/


  // inherit from one of the predefined plot class: Histogram2D
  class EcalPedestalEBM12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEBM12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain12 - map",
										 "iphi", 360, 1, 361, "ieta", 170, -85, 86) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
	    
	    // there's no ieta==0 in the EB numbering
	    //	    int delta = (EBDetId(rawid)).ieta() > 0 ? -1 : 0 ;
	    // fill the Histogram2D here
	    //	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta()+0.5+delta, (*payload)[rawid].mean_x12 );
	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), (*payload)[rawid].mean_x12 );
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalEBM6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEBM6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain6 - map",
										"iphi", 360, 1, 361, "ieta", 170, -85, 86) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
	    
	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), (*payload)[rawid].mean_x6 );
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalEBM1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEBM1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain1 - map",
										"iphi", 360, 1, 361, "ieta", 170, -85, 86) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
	    
	    fillWithValue(  (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), (*payload)[rawid].mean_x1 );
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };
  class EcalPedestalEEM12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEEM12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain12 - map", 
										 "ix", 220, 1, 221, "iy", 100, 1, 101) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    if(iz == 0)
		      fillWithValue( ix, iy, (*payload)[rawid].mean_x12 );
		    else
		      fillWithValue( ix + 120, iy, (*payload)[rawid].mean_x12 );

		}  // validDetId 
	} // payload
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalEEM6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEEM6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain6 - map",
										"ix", 220, 1, 221, "iy", 100, 1, 101) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    if(iz == 0)
		      fillWithValue( ix, iy, (*payload)[rawid].mean_x6);
		    else
		      fillWithValue( ix + 120, iy, (*payload)[rawid].mean_x6);

		}  // validDetId 
	} // payload
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalEEM1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEEM1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain1 - map",
										"ix", 220, 1, 221, "iy", 100, 1, 101) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12) continue;
		    if(iz == 0)
		      fillWithValue( ix, iy, (*payload)[rawid].mean_x1);
		    else
		      fillWithValue( ix + 120, iy, (*payload)[rawid].mean_x1);

		}  // validDetId 
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  // inherit from one of the predefined plot class: Histogram2D
  class EcalPedestalEBR12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEBR12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain12 - map",
										 "iphi",360,1,361, "ieta",170,-85,86) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
	    
	    // there's no ieta==0 in the EB numbering
	    //	    int delta = (EBDetId(rawid)).ieta() > 0 ? -1 : 0 ;
	    // fill the Histogram2D here
	    //	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta()+0.5+delta, (*payload)[rawid].mean_x12 );
	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), (*payload)[rawid].rms_x12 );
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalEBR6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEBR6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain6 - map",
										"iphi",360,1,361, "ieta",170,-85,86) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
	    
	    fillWithValue(  (EBDetId(rawid)).iphi() , (EBDetId(rawid)).ieta(), (*payload)[rawid].rms_x6 );
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalEBR1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEBR1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain1 - map",
										"iphi",360,1,361, "ieta",170,-85,86) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  // looping over the EB channels, via the dense-index, mapped into EBDetId's
	  if (!payload->barrelItems().size()) return false;
	  for(int cellid = EBDetId::MIN_HASH;
	      cellid < EBDetId::kSizeForDenseIndexing;
	      ++cellid) {
	    uint32_t rawid = EBDetId::unhashIndex(cellid);
	    
	    // check the existence of ECAL pedestal, for a given ECAL barrel channel
	    if (payload->find(rawid) == payload->end()) continue;
	    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x1) continue;
	    
	    fillWithValue(  (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), (*payload)[rawid].rms_x1 );
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };
  class EcalPedestalEER12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEER12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain12 - map", 
										 "ix", 220, 1, 221, "iy", 100, 1, 101) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    if(iz == 0)
		      fillWithValue( ix, iy, (*payload)[rawid].rms_x12 );
		    else
		      fillWithValue( ix + 120, iy, (*payload)[rawid].rms_x12 );

		}  // validDetId 
	} // payload
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalEER6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEER6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain6 - map",
										"ix", 220, 1, 221, "iy", 100, 1, 101) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    if(iz == 0)
		      fillWithValue( ix, iy, (*payload)[rawid].rms_x6);
		    else
		      fillWithValue( ix + 120, iy, (*payload)[rawid].rms_x6);

		}  // validDetId 
	} // payload
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalEER1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalEER1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain1 - map",
										"ix", 220, 1, 221, "iy", 100, 1, 101) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = 0; iz < 2; iz++)
	    for(int iy = 1; iy < 101; iy++)
	      for(int ix = 1; ix < 101; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = EEDetId::unhashIndex(myEEId);
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12) continue;
		    if(iz == 0)
		      fillWithValue( ix, iy, (*payload)[rawid].rms_x1);
		    else
		      fillWithValue( ix + 120, iy, (*payload)[rawid].rms_x1);

		}  // validDetId 
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( EcalPed ){
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBM12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBM6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBM1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEEM12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEEM6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEEM1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBR12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBR6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBR1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEER12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEER6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEER1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBM12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBM6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBM1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEEM12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEEM6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEEM1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBR12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBR6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEBR1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEER12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEER6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalEER1Map );
}
