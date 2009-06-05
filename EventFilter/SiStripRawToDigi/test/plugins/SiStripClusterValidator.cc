#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripClusterValidator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

SiStripClusterValidator::SiStripClusterValidator(const edm::ParameterSet& conf)
  : collection1Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection1")),
    collection2Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection2")),
    dsvnew_(conf.getUntrackedParameter<bool>("DetSetVectorNew",true)),
    errors_(false)
{
}

SiStripClusterValidator::~SiStripClusterValidator()
{  
}

void SiStripClusterValidator::beginJob(const edm::EventSetup& setup)
{
}

void SiStripClusterValidator::endJob()
{
  if (errors_) edm::LogInfo("SiStripClusterValidator") << "Differences were found" << std::endl;
  else edm::LogInfo("SiStripClusterValidator") << "Collections are identical in every event" << std::endl;
}

void SiStripClusterValidator::analyze(const edm::Event& event,const edm::EventSetup& setup)
{
  if (dsvnew_) {
    edm::Handle< edmNew::DetSetVector<SiStripCluster> > collection1Handle;
    event.getByLabel(collection1Tag_,collection1Handle); 
    edm::Handle< edmNew::DetSetVector<SiStripCluster> > collection2Handle;
    event.getByLabel(collection2Tag_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  } else {
    edm::Handle< edm::DetSetVector<SiStripCluster> > collection1Handle;
    event.getByLabel(collection1Tag_,collection1Handle); 
    edm::Handle< edm::DetSetVector<SiStripCluster> > collection2Handle;
    event.getByLabel(collection2Tag_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  }
}

void SiStripClusterValidator::validate(const edmNew::DetSetVector<SiStripCluster>& collection1, const edmNew::DetSetVector<SiStripCluster>& collection2)
{
  /// check number of DetSets

  if (collection1.size() != collection2.size()) {
    edm::LogWarning("SiStripClusterValidator") 
      << "Collection sizes do not match! ("
      << collection1.size() 
      << " and " 
      << collection2.size() 
      <<  ")";
    errors_ = true;
    return;
  }

  /// loop over first collection DetSets comparing them to same DetSet in other collection

  edmNew::DetSetVector<SiStripCluster>::const_iterator iDetSet1 = collection1.begin(); 
  edmNew::DetSetVector<SiStripCluster>::const_iterator jDetSet1 = collection1.end(); 
  for ( ; iDetSet1 != jDetSet1; ++iDetSet1 ) {
    
    /// check that it exists

    edm::det_id_type id = iDetSet1->detId();
    edmNew::DetSetVector<SiStripCluster>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      edm::LogWarning("SiStripClusterValidator")
	<< "DetSet in collection 1 with id " 
	<< id
	<< " is missing from collection 2!";
      errors_ = true;
      return;
    }
    
    /// check that the clusters are identical 

    edmNew::DetSet<SiStripCluster>::const_iterator iCluster1 = iDetSet1->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator iCluster2 = iDetSet2->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator jCluster2 = iDetSet2->end();
    for ( ; iCluster2 != jCluster2; ++iCluster2 ) { 
      if ( iCluster1->geographicalId() == iCluster2->geographicalId() &&
	   iCluster1->amplitudes() == iCluster2->amplitudes() &&
	   iCluster1->firstStrip() == iCluster2->firstStrip() ) iCluster1++;
    }
    
    if (iCluster1!=iDetSet1->end()) {
      edm::LogWarning("SiStripClusterValidator") 
	<< "No match for cluster in detector " 
	<< id 
	<< " with first strip number " 
	<< iCluster1->firstStrip();
      errors_ = true;
    }
  }
}

void SiStripClusterValidator::validate(const edm::DetSetVector<SiStripCluster>& collection1, const edm::DetSetVector<SiStripCluster>& collection2)
{
  /// check number of DetSets

  if (collection1.size() != collection2.size()) {
    edm::LogWarning("SiStripClusterValidator") 
      << "Collection sizes do not match! ("
      << collection1.size() 
      << " and " 
      << collection2.size() 
      <<  ")";
    errors_ = true;
    return;
  }

  /// loop over first collection DetSets comparing them to same DetSet in other collection

  edm::DetSetVector<SiStripCluster>::const_iterator iDetSet1 = collection1.begin(); 
  edm::DetSetVector<SiStripCluster>::const_iterator jDetSet1 = collection1.end(); 
  for ( ; iDetSet1 != jDetSet1; ++iDetSet1 ) {
    
    /// check that it exists

    edm::det_id_type id = iDetSet1->detId();
    edm::DetSetVector<SiStripCluster>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      edm::LogWarning("SiStripClusterValidator")
	<< "DetSet in collection 1 with id " 
	<< id
	<< " is missing from collection 2!";
      errors_ = true;
      return;
    }
    
    /// check that the clusters are identical 

    edm::DetSet<SiStripCluster>::const_iterator iCluster1 = iDetSet1->begin();
    edm::DetSet<SiStripCluster>::const_iterator iCluster2 = iDetSet2->begin();
    edm::DetSet<SiStripCluster>::const_iterator jCluster2 = iDetSet2->end();
    for ( ; iCluster2 != jCluster2; ++iCluster2 ) { 
      if ( iCluster1->geographicalId() == iCluster2->geographicalId() &&
	   iCluster1->amplitudes() == iCluster2->amplitudes() &&
	   iCluster1->firstStrip() == iCluster2->firstStrip() ) iCluster1++;
    }
    
    if (iCluster1!=iDetSet1->end()) {
      edm::LogWarning("SiStripClusterValidator") 
	<< "No match for cluster in detector " 
	<< id 
	<< " with first strip number " 
	<< iCluster1->firstStrip();
      errors_ = true;
    }
  }
}

/// Debug for SiStripCluster collection

std::ostream& operator<<(std::ostream& ss, const edmNew::DetSetVector<SiStripCluster>& clusters) {
  
  edmNew::DetSetVector<SiStripCluster>::const_iterator ids = clusters.begin(); 
  edmNew::DetSetVector<SiStripCluster>::const_iterator jds = clusters.end(); 
  for ( ; ids != jds; ++ids ) {
    std::stringstream sss;
    uint16_t nd = 0;
    edmNew::DetSet<SiStripCluster>::const_iterator id = ids->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator jd = ids->end();
    for ( ; id != jd; ++id ) {
      nd++;
      //if ( !( uint16_t( id - ids->begin() ) % 128 ) ) {
	if ( uint16_t( id - ids->begin() ) < 10 ) {
	  sss << uint16_t( id - ids->begin() ) 
	      << "/" 
	      << id->firstStrip() 
	      << "/" 
	      << id->amplitudes().size() 
	      <<", ";
	}
    }
    edm::LogWarning("SiStripClusterValidator") 
      << " DetId: " << ids->detId()
      << " size: " << ids->size()
      << " clusters: " << nd
      << " index/first-strip/width: " << sss.str()
      << std::endl;
  }
  return ss;
}

/// Debug for SiStripCluster collection

std::ostream& operator<<(std::ostream& ss, const edm::DetSetVector<SiStripCluster>& clusters) {
  
  edm::DetSetVector<SiStripCluster>::const_iterator ids = clusters.begin(); 
  edm::DetSetVector<SiStripCluster>::const_iterator jds = clusters.end(); 
  for ( ; ids != jds; ++ids ) {
    std::stringstream sss;
    uint16_t nd = 0;
    edm::DetSet<SiStripCluster>::const_iterator id = ids->begin();
    edm::DetSet<SiStripCluster>::const_iterator jd = ids->end();
    for ( ; id != jd; ++id ) {
      nd++;
      //if ( !( uint16_t( id - ids->begin() ) % 128 ) ) {
	if ( uint16_t( id - ids->begin() ) < 10 ) {
	  sss << uint16_t( id - ids->begin() ) 
	      << "/" 
	      << id->firstStrip() 
	      << "/" 
	      << id->amplitudes().size() 
	      <<", ";
	}
    }
    edm::LogWarning("SiStripClusterValidator") 
      << " DetId: " << ids->detId()
      << " size: " << ids->size()
      << " clusters: " << nd
      << " index/first-strip/width: " << sss.str()
      << std::endl;
  }
  return ss;
}
