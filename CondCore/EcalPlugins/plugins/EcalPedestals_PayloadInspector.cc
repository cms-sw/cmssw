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
  class EcalPedestalsEBM12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBM12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel pedestal average gain12",
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

  class EcalPedestalsEBM6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBM6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel pedestal average gain6",
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

  class EcalPedestalsEBM1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBM1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Barrel pedestal average gain1",
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

  class EcalPedestalsEEM12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEEM12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap pedestal average gain12", 
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

  class EcalPedestalsEEM6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEEM6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap pedestal average gain6",
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

  class EcalPedestalsEEM1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEEM1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Endcap pedestal average gain1", 
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

 class EcalPedestalsEBR12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBR12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel noise average gain12",
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

  class EcalPedestalsEBR6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBR6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel noise average gain6",
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

  class EcalPedestalsEBR1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBR1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Barrel noise average gain1",
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

  class EcalPedestalsEER12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEER12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap noise average gain12", 
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

  class EcalPedestalsEER6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEER6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap noise average gain6",
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

  class EcalPedestalsEER1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEER1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Endcap noise average gain1", 
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
  class EcalPedestalsEBM12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBM12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain12 - map",
										 "iphi", 360, 1, 361, "ieta", 171, -85, 86) {
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

  class EcalPedestalsEBM6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBM6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain6 - map",
										"iphi", 360, 1, 361, "ieta", 171, -85, 86) {
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

  class EcalPedestalsEBM1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBM1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain1 - map",
										"iphi", 360, 1, 361, "ieta", 171, -85, 86) {
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
  class EcalPedestalsEEM12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEEM12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain12 - map", 
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

  class EcalPedestalsEEM6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEEM6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain6 - map",
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

  class EcalPedestalsEEM1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEEM1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain1 - map",
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
  class EcalPedestalsEBR12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBR12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain12 - map",
										 "iphi",360,1,361, "ieta",171, -85, 86) {
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

  class EcalPedestalsEBR6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBR6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain6 - map",
										"iphi",360,1,361, "ieta",171, -85, 86) {
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

  class EcalPedestalsEBR1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEBR1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain1 - map",
										"iphi",360,1,361, "ieta",171, -85, 86) {
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
  class EcalPedestalsEER12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEER12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain12 - map", 
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

  class EcalPedestalsEER6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEER6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain6 - map",
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

  class EcalPedestalsEER1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
    EcalPedestalsEER1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain1 - map",
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
PAYLOAD_INSPECTOR_MODULE( EcalPedestals){
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBM12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBM6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBM1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEM12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEM6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEM1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBR12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBR6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBR1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEER12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEER6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEER1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBM12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBM6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBM1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEM12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEM6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEM1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBR12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBR6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBR1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEER12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEER6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEER1Map );
}
