#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiValidator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool operator == (const SiStripDigi& lhs, const SiStripDigi& rhs)
{
  return ( (lhs.strip() == rhs.strip()) && (lhs.adc() == rhs.adc()) );
}

SiStripDigiValidator::SiStripDigiValidator(const edm::ParameterSet& conf)
  : collection1Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection1")),
    collection2Tag_(conf.getUntrackedParameter<edm::InputTag>("Collection2")),
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
  //get collections from event
  edm::Handle< edm::DetSetVector<SiStripDigi> > collection1Handle, collection2Handle;
  bool gotCollection1 = event.getByLabel(collection1Tag_,collection1Handle);
  if (!gotCollection1) edm::LogError("SiStripDigiValidator") << "Failed to get collection 1 from event!";
  bool gotCollection2 = event.getByLabel(collection2Tag_,collection2Handle);
  if (!gotCollection2) edm::LogError("SiStripDigiValidator") << "Failed to get collection 2 from event!";
  const edm::DetSetVector<SiStripDigi>& collection1 = *collection1Handle;
  const edm::DetSetVector<SiStripDigi>& collection2 = *collection2Handle;
  
  //check number of DetSets
  if (collection1.size() != collection2.size()) {
    edm::LogWarning("SiStripDigiValidator") << "Collection sizes do not match!";
    errors_ = true;
  }
  
  //loop over first collection DetSets comparing them to same DetSet in other collection
  for (edm::DetSetVector<SiStripDigi>::const_iterator iDetSet1 = collection1.begin(); iDetSet1 != collection1.end(); iDetSet1++) {
    //check that it exists
    edm::det_id_type id = iDetSet1->detId();
    edm::DetSetVector<SiStripDigi>::const_iterator iDetSet2 = collection2.find(id);
    if (iDetSet2 == collection2.end()) {
      edm::LogWarning("SiStripDigiValidator") << "DetSet in collection 1 with id " << id << " is missing from collection 2!";
      errors_ = true;
    }
    //check that it is the same size
    if (iDetSet1->size() != iDetSet2->size()) {
      edm::LogWarning("SiStripDigiValidator") << "Sizes of DetSets with id " << id << " do not match!";
      errors_ = true;
    }
    //check that the digis are identical (not necessarily in the same order)
    for (edm::DetSetVector<SiStripDigi>::value_type::const_iterator iDigi1 = iDetSet1->begin(); iDigi1 != iDetSet1->end(); iDigi1++) {
      bool matchFound = false;
      for (edm::DetSetVector<SiStripDigi>::value_type::const_iterator iDigi2 = iDetSet2->begin(); iDigi2 != iDetSet2->end(); iDigi2++) {
        if (*iDigi1 == *iDigi2) {
          matchFound = true;
          break;
        }
      }
      if (!matchFound) {
        edm::LogWarning("SiStripDigiValidator") << "No match for digi with strip " << iDigi1->strip() << " in detector " << id << "!";
        errors_ = true;
      }
    }
  }
}
