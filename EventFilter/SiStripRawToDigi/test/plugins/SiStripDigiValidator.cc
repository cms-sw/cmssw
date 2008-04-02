#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiValidator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool operator == (const SiStripDigi& lhs, const SiStripDigi& rhs)
{
  return ( (lhs.strip() == rhs.strip()) && (lhs.adc() == rhs.adc()) );
}

bool operator == (const SiStripRawDigi& lhs, const SiStripRawDigi& rhs)
{
  return ( (lhs.adc() == rhs.adc()) );
}

SiStripDigiValidator::SiStripDigiValidator(const edm::ParameterSet& conf)
  : collection1Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection1")),
    collection2Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection2")),
    raw_(conf.getUntrackedParameter<bool>("FedRawDataMode")),
    errors_(false)
{
}

SiStripDigiValidator::~SiStripDigiValidator()
{  
}

void SiStripDigiValidator::beginJob(const edm::EventSetup& setup)
{
  errors_ = false;
}

void SiStripDigiValidator::endJob()
{
  if (errors_) edm::LogInfo("SiStripDigiValidator") << "Differences were found" << std::endl;
  else edm::LogInfo("SiStripDigiValidator") << "Collections are identical in every event" << std::endl;
}

void SiStripDigiValidator::analyze(const edm::Event& event,const edm::EventSetup& setup)
{
  if ( !raw_ ) { errors_ = Compare<SiStripDigi>( collection1Tag_, collection2Tag_, event, setup ); }
  else { errors_ = Compare<SiStripRawDigi>( collection1Tag_, collection2Tag_, event, setup ); }
}

template <typename T>
bool Compare( edm::InputTag collection1Tag_,
	      edm::InputTag collection2Tag_,
	      const edm::Event& event,
	      const edm::EventSetup& setup )
{
  
  bool errors = false;
  
  // get collection 1 from event
  edm::Handle< edm::DetSetVector<SiStripDigi> > collection1Handle;
  bool gotCollection1 = event.getByLabel(collection1Tag_,collection1Handle); 
  if (!gotCollection1) { 
    edm::LogWarning("SiStripDigiValidator") << "Failed to get collection 1 from event!";
    return true;
  }
  const edm::DetSetVector<SiStripDigi>& collection1 = *collection1Handle;
  
  // get collection 2 from event
  edm::Handle< edm::DetSetVector<T> > collection2Handle;
  bool gotCollection2 = event.getByLabel(collection2Tag_,collection2Handle); 
  if (!gotCollection2) {
    edm::LogWarning("SiStripDigiValidator") << "Failed to get collection 2 from event!";
    return true;
  }
  const edm::DetSetVector<T>& collection2 = *collection2Handle;
  
  //check number of DetSets
  if (collection1.size() != collection2.size()) {
    edm::LogWarning("SiStripDigiValidator") 
      << "Collection sizes do not match! ("
      << collection1.size() << " and " << collection2.size() <<  ")";
    errors = true;
  }

  // Debug for collection 1
  {
    std::stringstream ss;
    ss << " Collection1: " << std::endl;
    typename edm::DetSetVector<SiStripDigi>::const_iterator ids = collection1.begin(); 
    typename edm::DetSetVector<SiStripDigi>::const_iterator jds = collection1.end(); 
    for ( ; ids != jds; ++ids ) {
      std::stringstream sss;
      uint16_t nd = 0;
      typename edm::DetSetVector<SiStripDigi>::value_type::const_iterator id = ids->begin();
      typename edm::DetSetVector<SiStripDigi>::value_type::const_iterator jd = ids->end();
      for ( ; id != jd; ++id ) {
	if ( id->adc() ) { nd++; }
	//if ( !( uint16_t( id - ids->begin() ) % 128 ) ) {
	if ( uint16_t( id - ids->begin() ) < 10 ) {
	  sss << uint16_t( id - ids->begin() ) << "/" << id->adc() << ", ";
	}
      }
      ss << " DetId: " << ids->detId()
	 << " size: " << ids->size()
	 << " digis: " << nd
	 << " index/adc: " << sss.str()
	 << std:: endl;
    }
    LogTrace("SiStripDigiValidator") << ss.str();
  }

  // Debug for collection 2
  {
    std::stringstream ss;
    ss << " Collection2: " << std::endl;
    typename edm::DetSetVector<T>::const_iterator ids = collection2.begin(); 
    typename edm::DetSetVector<T>::const_iterator jds = collection2.end(); 
    for ( ; ids != jds; ++ids ) {
      std::stringstream sss;
      uint16_t nd = 0;
      typename edm::DetSetVector<T>::value_type::const_iterator id = ids->begin();
      typename edm::DetSetVector<T>::value_type::const_iterator jd = ids->end();
      for ( ; id != jd; ++id ) {
	if ( id->adc() ) { nd++; }
	//if ( !( uint16_t( id - ids->begin() ) % 128 ) ) {
	if ( uint16_t( id - ids->begin() ) < 10 ) {
	  sss << uint16_t( id - ids->begin() ) << "/" << id->adc() << ", ";
	}
      }
      ss << " DetId: " << ids->detId()
	 << " size: " << ids->size()
	 << " digis: " << nd
	 << " index/adc: " << sss.str()
	 << std::endl;
    }
    LogTrace("SiStripDigiValidator") << ss.str();
  }
  
  //loop over first collection DetSets comparing them to same DetSet in other collection
  typename edm::DetSetVector<SiStripDigi>::const_iterator iDetSet1 = collection1.begin(); 
  typename edm::DetSetVector<SiStripDigi>::const_iterator iDetSet2 = collection1.end(); 
  for ( ; iDetSet1 != iDetSet2; ++iDetSet1 ) {
    
    //check that it exists
    edm::det_id_type id = iDetSet1->detId();
    typename edm::DetSetVector<T>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      edm::LogWarning("SiStripDigiValidator")
	<< "DetSet in collection 1 with id " << id
	<< " is missing from collection 2!";
      return true;
    }

    //check that it is the same size
    if (iDetSet1->data.size() != iDetSet2->data.size()) {
      edm::LogWarning("SiStripDigiValidator") 
	<< "Sizes of DetSets with id " << id << " do not match! ("
	<< iDetSet1->data.size() << " and " << iDetSet2->data.size() <<  ")";
      return true;
    }
    
    //check that the digis are identical (not necessarily in the same order)
    typename edm::DetSetVector<SiStripDigi>::value_type::const_iterator iDigi1 = iDetSet1->begin();
    typename edm::DetSetVector<SiStripDigi>::value_type::const_iterator jDigi1 = iDetSet1->end();
    for ( ; iDigi1 != jDigi1; ++iDigi1 ) {

      bool matchFound = false;
      typename edm::DetSetVector<T>::value_type::const_iterator iDigi2 = iDetSet2->begin();
      typename edm::DetSetVector<T>::value_type::const_iterator jDigi2 = iDetSet2->end();
      for ( ; iDigi2 != jDigi2; ++iDigi2 ) {
	if ( ( iDigi1->adc() == iDigi2->adc() ) && ( iDigi1 - iDetSet1->begin() == iDigi2 - iDetSet2->begin() ) ) {
	  //if ( *iDigi1 == *iDigi2 ) { 
	  matchFound = true;
	  break;
	}
      }
      
      if (!matchFound) {
	edm::LogWarning("SiStripDigiValidator") 
	  << "No match for digi in detector " << id << "!";
	errors = true;
      }

    }

  }

  return errors;

}

// Specialized methods

template bool Compare<SiStripDigi>( edm::InputTag,
				    edm::InputTag,
				    const edm::Event&,
				    const edm::EventSetup& );

template bool Compare<SiStripRawDigi>( edm::InputTag, 
				       edm::InputTag, 
				       const edm::Event&,
				       const edm::EventSetup& );

