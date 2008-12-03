#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiValidator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
}

void SiStripDigiValidator::endJob()
{
  if (errors_) edm::LogInfo("SiStripDigiValidator") << "Differences were found" << std::endl;
  else edm::LogInfo("SiStripDigiValidator") << "Collections are identical in every event" << std::endl;
}

void SiStripDigiValidator::analyze(const edm::Event& event,const edm::EventSetup& setup)
{

  if (raw_) {

    edm::Handle< edm::DetSetVector<SiStripDigi> > collection1Handle;
    event.getByLabel(collection1Tag_,collection1Handle);     
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > collection2Handle;
    event.getByLabel(collection2Tag_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  }
  
  else {
    
    edm::Handle< edm::DetSetVector<SiStripDigi> > collection1Handle;
    event.getByLabel(collection1Tag_,collection1Handle); 
    edm::Handle< edm::DetSetVector<SiStripDigi> > collection2Handle;
    event.getByLabel(collection2Tag_,collection2Handle);
    validate(*collection1Handle,*collection2Handle);
  }
}

void SiStripDigiValidator::validate(const edm::DetSetVector<SiStripDigi>& collection1, const edm::DetSetVector<SiStripDigi>& collection2)
{

  //check number of DetSets
  if (collection1.size() != collection2.size()) {
    edm::LogWarning("SiStripDigiValidator") 
      << "Collection sizes do not match! ("
      << collection1.size() 
      << " and " 
      << collection2.size() 
      <<  ")";
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
      edm::LogWarning("SiStripDigiValidator")
	<< "DetSet in collection 1 with id " 
	<< id
	<< " is missing from collection 2!";
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
      edm::LogWarning("SiStripDigiValidator") 
	<< "No match for digi in detector " 
	<< id 
	<< " with strip number " 
	<< iDigi1->strip();
      errors_ = true;
    }
  }
}

void SiStripDigiValidator::validate(const edm::DetSetVector<SiStripDigi>& collection1, const edm::DetSetVector<SiStripRawDigi>& collection2)
{

  //check number of DetSets
  if (collection1.size() != collection2.size()) {
    edm::LogWarning("SiStripDigiValidator") 
      << "Collection sizes do not match! ("
      << collection1.size() 
      << " and " 
      << collection2.size() 
      <<  ")";
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
      edm::LogWarning("SiStripDigiValidator")
	<< "DetSet in collection 1 with id " 
	<< id
	<< " is missing from collection 2!";
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
      edm::LogWarning("SiStripDigiValidator") 
	<< "No match for digi in detector " 
	<< id 
	<< " with strip number " 
	<< iDigi1->strip();
      errors_ = true;
    }
  }
}




