#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialClusterSource.h"
// edm 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
// cabling
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// clhep
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
// std
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripTrivialClusterSource::SiStripTrivialClusterSource( const edm::ParameterSet& pset ) :

  minOcc_( pset.getUntrackedParameter<double>("MinOccupancy",0.001) ),
  maxOcc_( pset.getUntrackedParameter<double>("MaxOccupancy",0.03) ),
  minCluster_( pset.getUntrackedParameter<unsigned int>("MinCluster",1) ),
  maxCluster_( pset.getUntrackedParameter<unsigned int>("MaxCluster",5) ),
  mixClusters_( pset.getUntrackedParameter<bool>("MixClusters",true) ),
  separation_( pset.getUntrackedParameter<unsigned int>("Separation",2) ),
  maxAdc_( pset.getUntrackedParameter<bool>("MaxAdc",false) ),
  detCabling_(),
  detids_(),
  nstrips_(0)
{
  LogTrace("TrivialClusterSource") 
    << "[SiStripTrivialClusterSource::" 
    << __func__
    << "]";
  
  produces< edm::DetSetVector<SiStripDigi> >();
}

// -----------------------------------------------------------------------------
/** */
SiStripTrivialClusterSource::~SiStripTrivialClusterSource() {

  LogTrace("TrivialClusterSource") 
    << "[SiStripTrivialClusterSource::" 
    << __func__
    << "]";
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialClusterSource::beginJob( const edm::EventSetup& setup) {  

  LogTrace("TrivialClusterSource") 
    << "[SiStripTrivialClusterSource::" 
    << __func__
    << "]";

  edm::ESHandle<SiStripFedCabling> fedcabling;
  setup.get<SiStripFedCablingRcd>().get( fedcabling );
  cabling_ = new SiStripDetCabling(*fedcabling.product());
  detCabling_ = cabling_->getDetCabling();

 //Store Det-ids and Calculate number of strips in tracker
  detids_.reserve(detCabling_.size());
  map< uint32_t, vector<FedChannelConnection> >::const_iterator iconn = detCabling_.begin();
  for (;iconn != detCabling_.end();iconn++) {
    if ((!iconn->first) || (iconn->first == sistrip::invalid32_)) {
    detids_.push_back(iconn->first);
    nstrips_+=(iconn->second.size()*256);
    }
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialClusterSource::endJob() {
  if (cabling_) delete cabling_;}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialClusterSource::produce( edm::Event& iEvent, 
					const edm::EventSetup& iSetup ) {

  LogDebug("TrivialClusterSource") 
    << __func__
    << "] "
    << "Event: "
    << iEvent.id();

  auto_ptr< edm::DetSetVector<SiStripDigi> > clusters( new edm::DetSetVector<SiStripDigi>() );

  //Occupancy
  double occupancy = RandFlat::shoot() * (maxOcc_ - minOcc_) + minOcc_;
  uint32_t ndigis = (uint32_t)(nstrips_ * occupancy);

  //Fill collection with clusters
  uint16_t csize = randflat(minCluster_,maxCluster_);

  while (ndigis > csize) {
 
    bool accepted = false;
    uint32_t safetyCounter = 0;
    while (!accepted && 
	   (safetyCounter < 10000)) {
      
      //Generate det-id and strip number
      uint32_t index = randflat(0,detCabling_.size()-1);
      uint32_t detid = detids_[index];
      uint16_t strip = randflat(0,256*detCabling_[detid].size() - csize);
      
      //Access detset for chosen detid from collection
      edm::DetSet<SiStripDigi>& detset = clusters->find_or_insert(detid);
      detset.data.reserve(768);
      
      //Update detset with cluster at chosen strip position if clear
      CandidateCluster candidate(strip,csize,separation_);
      if (find_if(detset.begin(),detset.end(),candidate)==detset.end()) {
	
	for (unsigned int istrip=0;istrip<csize;++istrip) { 
	  uint16_t adc = (maxAdc_) ? 255 : randflat(1,255);
	  detset.data.push_back(SiStripDigi(strip+istrip,adc));
	}
	accepted = true;
      }
      else safetyCounter++;
    }
    
    if (safetyCounter > 10000) {
      edm::LogError("TrivialClusterSource")
	<< "[SiStripTrivialClusterSource::"
	<< __func__
	<<"]"
	<< " Detector saturated - no room for new clusters.";
      break;
    }
    
    //Update cluster size and digis counter
    ndigis-=csize;
    if (mixClusters_) csize = randflat(minCluster_,maxCluster_);
  }
  

  iEvent.put( clusters );

 LogDebug("TrivialClusterSource")
   << "[SiStripTrivialClusterSource::"
   << __func__
   <<"]"
   << " Generated " << occupancy*nstrips_
   << " digis for " << nstrips_/256.
   << " channels with a mean occupancy of " 
   << occupancy *100.
   << " %";
  }

uint32_t SiStripTrivialClusterSource::randflat(uint32_t min,uint32_t max) {

  if (max < min)
    edm::LogError("TrivialClusterSource")
      << "[SiStripTrivialClusterSource::"
      << __func__
      <<"]"
      << " Min > Max";
  
  return (uint32_t)(RandFlat::shoot() * (max - min +1)) + min; 
}
