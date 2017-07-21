#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <memory>
#include <sstream>



namespace {



  /************************************************
   
     1d histogram of ECAL barrel pedestal of 1 IOV 

  *************************************************/


  // inherit from one of the predefined plot class: Histogram1D
  class EcalPedestalsEBMean12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBMean12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel pedestal average gain12",
									      "ECAL Barrel pedestal average gain12", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for ( auto const & iov: iovs) {
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

  class EcalPedestalsEBMean6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBMean6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel pedestal average gain6",
									     "ECAL Barrel pedestal average gain6", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {

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

  class EcalPedestalsEBMean1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBMean1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Barrel pedestal average gain1",
									    "ECAL Barrel pedestal average gain1", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
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

  class EcalPedestalsEEMean12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

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
    EcalPedestalsEEMean12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap pedestal average gain12", 
									      "ECAL Endcap pedestal average gain12", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
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

  class EcalPedestalsEEMean6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

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
    EcalPedestalsEEMean6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap pedestal average gain6",
									     "ECAL Endcap pedestal average gain6", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {

	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
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

  class EcalPedestalsEEMean1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

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
    EcalPedestalsEEMean1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Endcap pedestal average gain1", 
									    "ECAL Endcap pedestal average gain1", 200, 150, 250){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
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

 class EcalPedestalsEBRMS12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBRMS12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel noise average gain12",
									      "ECAL Barrel noise average gain12", 100, 0, 6){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
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

  class EcalPedestalsEBRMS6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBRMS6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Barrel noise average gain6",
									     "ECAL Barrel noise average gain6", 100, 0, 3){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {

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

  class EcalPedestalsEBRMS1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

  public:
    EcalPedestalsEBRMS1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Barrel noise average gain1",
									    "ECAL Barrel noise average gain1", 100, 0, 2){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
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

  class EcalPedestalsEERMS12 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

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
    EcalPedestalsEERMS12() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap noise average gain12", 
									      "ECAL Endcap noise average gain12", 100, 0, 8){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
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

  class EcalPedestalsEERMS6 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

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
    EcalPedestalsEERMS6() : cond::payloadInspector::Histogram1D<EcalPedestals>( "ECAL Endcap noise average gain6",
									     "ECAL Endcap noise average gain6", 100, 0, 4){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {

	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){

	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
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

  class EcalPedestalsEERMS1 : public cond::payloadInspector::Histogram1D<EcalPedestals> {

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
    EcalPedestalsEERMS1() : cond::payloadInspector::Histogram1D<EcalPedestals>("ECAL Endcap noise average gain1", 
									    "ECAL Endcap noise average gain1", 100, 0, 3){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
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
  class EcalPedestalsEBMean12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
      static const int MIN_IETA = 1;
      static const int MIN_IPHI = 1;
      static const int MAX_IETA = 85;
      static const int MAX_IPHI = 360;

    EcalPedestalsEBMean12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain12 - map",
										     "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
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
	    // set min and max on 2d plots
	    float valped = (*payload)[rawid].mean_x12;
	    if(valped > 250.) valped = 250.;
	    //	    if(valped < 150.) valped = 150.;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEBMean6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
      static const int MIN_IETA = 1;
      static const int MIN_IPHI = 1;
      static const int MAX_IETA = 85;
      static const int MAX_IPHI = 360;
    EcalPedestalsEBMean6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain6 - map",
										    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1){
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
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
	    
	    // set min and max on 2d plots
	    float valped = (*payload)[rawid].mean_x6;
	    if(valped > 250.) valped = 250.;
	    //	    if(valped < 150.) valped = 150.;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
  	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEBMean1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
      static const int MIN_IETA = 1;
      static const int MIN_IPHI = 1;
      static const int MAX_IETA = 85;
      static const int MAX_IPHI = 360;
    EcalPedestalsEBMean1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel pedestal gain1 - map",
										    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
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
	    
	    // set min and max on 2d plots
	    float valped = (*payload)[rawid].mean_x1;
	    if(valped > 250.) valped = 250.;
	    //	    if(valped < 150.) valped = 150.;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valped);
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEEMean12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

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
    EcalPedestalsEEMean12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain12 - map", 
										"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    // set min and max on 2d plots
		    float valped = (*payload)[rawid].mean_x12;
		    if(valped > 250.) valped = 250.;
		    //		    if(valped < 150.) valped = 150.;
		    if(iz == -1)
		      fillWithValue(ix, iy, valped);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valped);

		}  // validDetId 
	} // payload
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEEMean6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

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
    EcalPedestalsEEMean6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain6 - map",
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
       Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    // set min and max on 2d plots
		    float valped = (*payload)[rawid].mean_x6;
		    if(valped > 250.) valped = 250.;
		    //		    if(valped < 150.) valped = 150.;
		    if(iz == -1)
		      fillWithValue(ix, iy, valped);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valped);
		}  // validDetId 
	} // payload
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEEMean1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

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
    EcalPedestalsEEMean1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap pedestal gain1 - map",
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {		Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12) continue;
		    // set min and max on 2d plots
		    float valped = (*payload)[rawid].mean_x1;
		    if(valped > 250.) valped = 250.;
		    //		    if(valped < 150.) valped = 150.;
		    if(iz == -1)
		      fillWithValue(ix, iy, valped);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valped);
		}  // validDetId 
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  // inherit from one of the predefined plot class: Histogram2D
  class EcalPedestalsEBRMS12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
      static const int MIN_IETA = 1;
      static const int MIN_IPHI = 1;
      static const int MAX_IETA = 85;
      static const int MAX_IPHI = 360;
    EcalPedestalsEBRMS12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain12 - map",
    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {								      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
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
	    // set max on noise 2d plots
	    float valrms = (*payload)[rawid].rms_x12;
	    if(valrms > 2.2) valrms = 2.2;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
	  }// loop over cellid
	}// if payload.get()
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEBRMS6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
      static const int MIN_IETA = 1;
      static const int MIN_IPHI = 1;
      static const int MAX_IETA = 85;
      static const int MAX_IPHI = 360;
    EcalPedestalsEBRMS6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain6 - map",
    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {										Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
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
	    
	    // set max on noise 2d plots
	    float valrms = (*payload)[rawid].rms_x6;
	    if(valrms > 1.5) valrms = 1.5;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEBRMS1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

  public:
      static const int MIN_IETA = 1;
      static const int MIN_IPHI = 1;
      static const int MAX_IETA = 85;
      static const int MAX_IPHI = 360;
    EcalPedestalsEBRMS1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Barrel noise gain1 - map",
    "iphi", MAX_IPHI, MIN_IPHI, MAX_IPHI+MIN_IPHI, "ieta", 2*MAX_IETA+1, -1*MAX_IETA, MAX_IETA+1) {								      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
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
	    
	    // set max on noise 2d plots
	    float valrms = (*payload)[rawid].rms_x1;
	    if(valrms > 1.0) valrms = 1.0;
	    fillWithValue( (EBDetId(rawid)).iphi(), (EBDetId(rawid)).ieta(), valrms);
	  }  // loop over cellid
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };
  class EcalPedestalsEERMS12Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

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
    EcalPedestalsEERMS12Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain12 - map", 
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    // Histogram2D::fill (virtual) needs be overridden - the implementation should use fillWithValue
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){

      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x12 && !(*payload)[rawid].rms_x12) continue;
		    // set max on noise 2d plots
		    float valrms = (*payload)[rawid].rms_x12;
		    if(valrms > 3.5) valrms = 3.5;
		    if(iz == -1)
		      fillWithValue(ix, iy, valrms);
		    else
		      fillWithValue(ix + IX_MAX + 20, iy, valrms);

		}  // validDetId 
	} // payload
      }// loop over IOV's (1 in this case)
      return true;
    }// fill method
  };

  class EcalPedestalsEERMS6Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

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
    EcalPedestalsEERMS6Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain6 - map",
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x6 && !(*payload)[rawid].rms_x6) continue;
		    // set max on noise 2d plots
		    float valrms = (*payload)[rawid].rms_x6;
		    if(valrms > 2.0) valrms = 2.0;
		    if(iz == -1)
		      fillWithValue( ix, iy, valrms);
		    else
		      fillWithValue( ix + IX_MAX + 20, iy, valrms);
		}  // validDetId 
	} // payload
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

  class EcalPedestalsEERMS1Map : public cond::payloadInspector::Histogram2D<EcalPedestals> {

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
  EcalPedestalsEERMS1Map() : cond::payloadInspector::Histogram2D<EcalPedestals>( "ECAL Endcap noise gain1 - map",
"ix", 2.2*IX_MAX, IX_MIN, 2.2*IX_MAX+1, "iy", IY_MAX, IY_MIN, IY_MAX+IY_MIN) {
      Base::setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for( auto const & iov : iovs ) {
	std::shared_ptr<EcalPedestals> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  if (!payload->endcapItems().size()) return false;

	  // looping over the EE channels
	  for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
	    for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
	      for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
		if(EEDetId::validDetId(ix, iy, iz)) {
		    EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
		    uint32_t rawid = myEEId.rawId();
		    // check the existence of ECAL pedestal, for a given ECAL endcap channel
		    if (payload->find(rawid) == payload->end()) continue;
		    if (!(*payload)[rawid].mean_x1 && !(*payload)[rawid].rms_x12) continue;
		    // set max on noise 2d plots
		    float valrms = (*payload)[rawid].rms_x1;
		    if(valrms > 1.5) valrms = 1.5;
		    if(iz == -1)
		      fillWithValue( ix, iy, valrms);
		    else
		      fillWithValue( ix + IX_MAX + 20, iy, valrms);
		}  // validDetId 
	}   // if payload.get()
      }    // loop over IOV's (1 in this case)
      return true;
    }    // fill method
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( EcalPedestals){
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS12);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS6 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS1 );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBMean1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEEMean1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEBRMS1Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS12Map);
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS6Map );
  PAYLOAD_INSPECTOR_CLASS( EcalPedestalsEERMS1Map );
}
