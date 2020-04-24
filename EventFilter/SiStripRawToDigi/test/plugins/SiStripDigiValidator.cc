#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiValidator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiStripDigiValidator::SiStripDigiValidator(const edm::ParameterSet& conf)
  : tag1_(conf.getUntrackedParameter<edm::InputTag>("TagCollection1")),
    tag2_(conf.getUntrackedParameter<edm::InputTag>("TagCollection2")),
    raw1_(conf.getUntrackedParameter<bool>("RawCollection1")),
    raw2_(conf.getUntrackedParameter<bool>("RawCollection2")),
    errors_(false),
    header_()
{
  std::stringstream ss;
  ss << " Collection1: " 
     << " Type:Label:Instance:Process=\""
     << ( raw1_ ? "SiStripRawDigi" : "SiStripDigi" )
     << ":" << tag1_.label()
     << ":" << tag1_.instance()
     << ":" << tag1_.process()
     << "\""
     << std::endl
     << " Collection2: " 
     << " Type:Label:Instance:Process=\""
     << ( raw2_ ? "SiStripRawDigi" : "SiStripDigi" )
     << ":" << tag2_.label()
     << ":" << tag2_.instance()
     << ":" << tag2_.process()
     << "\""
     << std::endl;
  header_ = ss.str();
  
  mayConsume< edm::DetSetVector<SiStripDigi> >( tag1_ );
  mayConsume< edm::DetSetVector<SiStripRawDigi> >( tag1_ );
  mayConsume< edm::DetSetVector<SiStripDigi> >( tag2_ );
  mayConsume< edm::DetSetVector<SiStripRawDigi> >( tag2_ );
}

SiStripDigiValidator::~SiStripDigiValidator()
{  
}

void SiStripDigiValidator::beginJob()
{
}

void SiStripDigiValidator::endJob()
{

  std::stringstream ss;
  ss << "[SiStripDigiValidator::" << __func__ << "]"
     << std::endl
     << header();
  
  if (!errors_) { ss << "Collections are identical in every event (assuming no exceptions thrown!)" << std::endl; }
  else { ss << "Differences were found" << std::endl; }

  if (!errors_) { edm::LogVerbatim("SiStripDigiValidator") << ss.str(); }
  else { edm::LogError("SiStripDigiValidator") << ss.str(); }

}

void SiStripDigiValidator::analyze( const edm::Event& event,
				    const edm::EventSetup& setup )
{
  if ( !raw1_ && !raw2_ ) {
    edm::Handle< edm::DetSetVector<SiStripDigi> > collection1Handle;
    event.getByLabel(tag1_,collection1Handle); 
    edm::Handle< edm::DetSetVector<SiStripDigi> > collection2Handle;
    event.getByLabel(tag2_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  } else if ( !raw1_ && raw2_ ) {
    edm::Handle< edm::DetSetVector<SiStripDigi> > collection1Handle;
    event.getByLabel(tag1_,collection1Handle);     
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > collection2Handle;
    event.getByLabel(tag2_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  } else if ( raw1_ && !raw2_ ) {
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > collection1Handle;
    event.getByLabel(tag1_,collection1Handle);     
    edm::Handle< edm::DetSetVector<SiStripDigi> > collection2Handle;
    event.getByLabel(tag2_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  } else if ( raw1_ && raw2_ ) {
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > collection1Handle;
    event.getByLabel(tag1_,collection1Handle);     
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > collection2Handle;
    event.getByLabel(tag2_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  }
}

void SiStripDigiValidator::validate( const edm::DetSetVector<SiStripDigi>& collection1, 
				     const edm::DetSetVector<SiStripDigi>& collection2 )
{

  //check number of DetSets
  if (collection1.size() != collection2.size()) {
    std::stringstream ss;
    ss << "[SiStripDigiValidator::" << __func__ << "]"
       << std::endl
       << header()
       << "Collection sizes do not match! ("
       << collection1.size() 
       << " and " 
       << collection2.size() 
       <<  ")";
    edm::LogError("SiStripDigiValidator") << ss.str();
    errors_ = true;
    return;
  }

  //loop over first collection DetSets comparing them to same DetSet in other collection
  edm::DetSetVector<SiStripDigi>::const_iterator iDetSet1 = collection1.begin(); 
  edm::DetSetVector<SiStripDigi>::const_iterator jDetSet1 = collection1.end(); 
  for ( ; iDetSet1 != jDetSet1; ++iDetSet1 ) {
    
    //check that it exists
    edm::det_id_type id = iDetSet1->detId();
    edm::DetSetVector<SiStripDigi>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      std::stringstream ss;
      ss << "[SiStripDigiValidator::" << __func__ << "]"
	 << std::endl
	 << header()
	 << "DetSet in collection 1 with id " 
	 << id
	 << " is missing from collection 2!";
      edm::LogError("SiStripDigiValidator") << ss.str();
      errors_ = true;
      return;
    }
    
    //check that the digis are identical 
    edm::DetSet<SiStripDigi>::const_iterator iDigi1 = iDetSet1->begin();
    edm::DetSet<SiStripDigi>::const_iterator iDigi2 = iDetSet2->begin();
    edm::DetSet<SiStripDigi>::const_iterator jDigi2 = iDetSet2->end();
    for ( ; iDigi2 != jDigi2; ++iDigi2 ) { 
      if ((iDigi1->adc()==iDigi2->adc())&&
	  (iDigi1->strip()==iDigi2->strip())) iDigi1++;
    }
    
    if (iDigi1!=iDetSet1->end()) {
      std::stringstream ss;
      ss << "[SiStripDigiValidator::" << __func__ << "]"
	 << std::endl
	 << header()
	 << "No match for digi in detector " 
	 << id 
	 << " with strip number " 
	 << iDigi1->strip();
      edm::LogError("SiStripDigiValidator") << ss.str();
      errors_ = true;
    }
  }
}

void SiStripDigiValidator::validate( const edm::DetSetVector<SiStripDigi>& collection1, 
				     const edm::DetSetVector<SiStripRawDigi>& collection2 )
{

  //check number of DetSets
  if (collection1.size() != collection2.size()) {
      std::stringstream ss;
      ss << "[SiStripDigiValidator::" << __func__ << "]"
	 << std::endl
	 << header()
	 << "Collection sizes do not match! ("
	 << collection1.size() 
	 << " and " 
	 << collection2.size() 
	 <<  ")";
      edm::LogError("SiStripDigiValidator") << ss.str();
    errors_ = true;
    return;
  }

  //loop over first collection DetSets comparing them to same DetSet in other collection
  edm::DetSetVector<SiStripDigi>::const_iterator iDetSet1 = collection1.begin(); 
  edm::DetSetVector<SiStripDigi>::const_iterator jDetSet1 = collection1.end(); 
  for ( ; iDetSet1 != jDetSet1; ++iDetSet1 ) {
    
    //check that it exists
    edm::det_id_type id = iDetSet1->detId();
    edm::DetSetVector<SiStripRawDigi>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      std::stringstream ss;
      ss << "[SiStripDigiValidator::" << __func__ << "]"
	 << std::endl
	 << header()
	 << "DetSet in collection 1 with id " 
	 << id
	 << " is missing from collection 2!";
      edm::LogError("SiStripDigiValidator") << ss.str();
      errors_ = true;
      return;
    }
    
    //check that the digis are identical 
    edm::DetSet<SiStripDigi>::const_iterator iDigi1 = iDetSet1->begin();
    edm::DetSet<SiStripRawDigi>::const_iterator iDigi2 = iDetSet2->begin();
    edm::DetSet<SiStripRawDigi>::const_iterator jDigi2 = iDetSet2->end();

    for ( ; iDigi2 != jDigi2; ++iDigi2 ) {
      if ((iDigi1->adc()==iDigi2->adc()) && 
	  (iDigi1->strip()==iDigi2-iDetSet2->begin())) iDigi1++;
    }
    
    if (iDigi1!=iDetSet1->end()) {
      std::stringstream ss;
      ss << "[SiStripDigiValidator::" << __func__ << "]"
	 << std::endl
	 << header()
	 << "No match for digis in detector " 
	 << id 
	 << " with strip/adc: " 
	 << iDigi1->strip() << "/" << iDigi1->adc()
	 << " and "
	 << uint32_t( iDigi2 - iDetSet2->begin() ) << "/" << iDigi2->adc();
      edm::LogError("SiStripDigiValidator") << ss.str();
      errors_ = true;
    }

  }
}

void SiStripDigiValidator::validate( const edm::DetSetVector<SiStripRawDigi>& collection1, 
				     const edm::DetSetVector<SiStripDigi>& collection2 )
{
  validate(collection2,collection1);
}

void SiStripDigiValidator::validate( const edm::DetSetVector<SiStripRawDigi>& collection1, 
				     const edm::DetSetVector<SiStripRawDigi>& collection2 )
{

  //check number of DetSets
  if (collection1.size() != collection2.size()) {
    std::stringstream ss;
    ss << "[SiStripDigiValidator::" << __func__ << "]"
       << std::endl
       << header()
       << "Collection sizes do not match! ("
       << collection1.size() 
       << " and " 
       << collection2.size() 
       <<  ")";
    edm::LogError("SiStripDigiValidator") << ss.str();
    errors_ = true;
    return;
  }

  //loop over first collection DetSets comparing them to same DetSet in other collection
  edm::DetSetVector<SiStripRawDigi>::const_iterator iDetSet1 = collection1.begin(); 
  edm::DetSetVector<SiStripRawDigi>::const_iterator jDetSet1 = collection1.end(); 
  for ( ; iDetSet1 != jDetSet1; ++iDetSet1 ) {
    
    //check that it exists
    edm::det_id_type id = iDetSet1->detId();
    edm::DetSetVector<SiStripRawDigi>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      std::stringstream ss;
      ss << "[SiStripDigiValidator::" << __func__ << "]"
	 << std::endl
	 << header()
	 << "DetSet in collection 1 with id " 
	 << id
	 << " is missing from collection 2!";
      edm::LogError("SiStripDigiValidator") << ss.str();
      errors_ = true;
      return;
    }
    
    //check that the digis are identical 
    edm::DetSet<SiStripRawDigi>::const_iterator iDigi1 = iDetSet1->begin();
    edm::DetSet<SiStripRawDigi>::const_iterator iDigi2 = iDetSet2->begin();
    edm::DetSet<SiStripRawDigi>::const_iterator jDigi2 = iDetSet2->end();
    for ( ; iDigi2 != jDigi2; ++iDigi2 ) { 
      if ( iDigi1->adc() != iDigi2->adc() ||
	   iDigi1 - iDetSet1->begin() != 
	   iDigi2 - iDetSet2->begin() ) { break; }
      iDigi1++;
    }
    
    if ( iDigi1 != iDetSet1->end() ) {
      std::stringstream ss;
      ss << "[SiStripDigiValidator::" << __func__ << "]"
	 << std::endl
	 << header()
	 << "No match for digi in detector " 
	 << id 
	 << " with strip number " 
	 << ( iDigi1 - iDetSet1->begin() );
      edm::LogError("SiStripDigiValidator") << ss.str();
      errors_ = true;
    }
  }

}
  
  
  
