#include "OnDemandMeasurementTracker.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"  

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TkStripMeasurementDet.h"
#include "TkPixelMeasurementDet.h"
#include "TkGluedMeasurementDet.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

#include <iostream>
#include <typeinfo>
#include <map>

#include <DataFormats/GeometrySurface/interface/BoundPlane.h>
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/interface/UpdaterService.h"

using namespace std;

OnDemandMeasurementTracker::OnDemandMeasurementTracker(const edm::ParameterSet&              conf,
						       const PixelClusterParameterEstimator* pixelCPE,
						       const StripClusterParameterEstimator* stripCPE,
						       const SiStripRecHitMatcher*  hitMatcher,
						       const TrackerGeometry*  trackerGeom,
						       const GeometricSearchTracker* geometricSearchTracker,
						       const SiStripQuality *stripQuality,
                                                       int   stripQualityFlags,
                                                       int   stripQualityDebugFlags,
                                                       const SiPixelQuality *pixelQuality,
                                                       const SiPixelFedCabling *pixelCabling,
                                                       int   pixelQualityFlags,
                                                       int   pixelQualityDebugFlags,
						       const SiStripRegionCabling * stripRegionCabling,
						       bool isRegional):
  MeasurementTrackerImpl(conf,pixelCPE,stripCPE,hitMatcher,trackerGeom,geometricSearchTracker,
        stripQuality,stripQualityFlags,stripQualityDebugFlags,
        pixelQuality,pixelCabling,pixelQualityFlags,pixelQualityDebugFlags,
        isRegional)
  , category_("OnDemandMeasurementTracker")
  , StayPacked_(true)
  , StripOnDemand_(true)
  , PixelOnDemand_(false)
  , theStripRegionCabling(stripRegionCabling)
{
  //  the constructor does construct the regular MeasurementTracker
  //  then a smart copy of the DetMap is made into DetODMap: this could be avoided with modification to MeasurementDet interface
  //  the elementIndex to be defined in the refgetter is mapped to the detId
  //  flags are set to initialize the DetODMap
  
  std::map<SiStripRegionCabling::ElementIndex, std::vector< DetODContainer::const_iterator> > local_mapping;
  for (DetContainer::iterator it=theDetMap.begin(); it!= theDetMap.end();++it)
    {
      DetODContainer::iterator inserted = theDetODMap.insert(make_pair(it->first,DetODStatus(it->second))).first;


      GeomDet::SubDetector subdet = it->second->geomDet().subDetector();
      if (subdet == GeomDetEnumerators::PixelBarrel  || subdet == GeomDetEnumerators::PixelEndcap ){
        inserted->second.index = -1;
        inserted->second.kind  = DetODStatus::Pixel;
      }//pixel module
      else if (subdet == GeomDetEnumerators::TIB || subdet == GeomDetEnumerators::TOB ||
	  subdet == GeomDetEnumerators::TID || subdet == GeomDetEnumerators::TEC )
	{
	  //set flag to false
          if (typeid(*it->second) == typeid(TkStripMeasurementDet)) {
            const TkStripMeasurementDet & sdet = static_cast<const TkStripMeasurementDet &>(*it->second);
            inserted->second.index = sdet.index();
            inserted->second.kind = DetODStatus::Strip;
          } else {
            inserted->second.kind = DetODStatus::Glued;
          }

	  //what will be the element index in the refgetter
	  GlobalPoint center = it->second->geomDet().position();
	  double eta = center.eta();
	  double phi = center.phi();
	  uint32_t id = it->first;
	  SiStripRegionCabling::ElementIndex eIndex = theStripRegionCabling->elementIndex(SiStripRegionCabling::Position(eta,phi),
											  SiStripRegionCabling::subdetFromDetId(id),
											  SiStripRegionCabling::layerFromDetId(id));
	  LogDebug(category_)<<"region selected (from "<<id<<" center) is:\n"
			     <<"position: "<<center
			     <<"\n center absolute index: "<<theStripRegionCabling->region(theStripRegionCabling->positionIndex(SiStripRegionCabling::Position(eta,phi)))
			     <<"\n center position index: "<<theStripRegionCabling->positionIndex(SiStripRegionCabling::Position(eta,phi)).first<<
	    " "<<theStripRegionCabling->positionIndex(SiStripRegionCabling::Position(eta,phi)).second
			     <<"\n center postion: "<<theStripRegionCabling->position(theStripRegionCabling->positionIndex(SiStripRegionCabling::Position(eta,phi))).first<<
	    " "<<theStripRegionCabling->position(theStripRegionCabling->positionIndex(SiStripRegionCabling::Position(eta,phi))).second
			     <<"\n eta: "<<eta
			     <<"\n phi: "<<phi
			     <<"\n subedet: "<<SiStripRegionCabling::subdetFromDetId(id)
			     <<" layer: "<<SiStripRegionCabling::layerFromDetId(id);

	  //	  register those in a map
	  //to be able to know what are the detid in a given elementIndex
	  local_mapping[eIndex].push_back(inserted);
	}//strip module
      else{
	//abort
	edm::LogError(category_)<<"not a tracker geomdet in constructor: "<<it->first;
	throw MeasurementDetException("OnDemandMeasurementTracker dealing with a non tracker GeomDet.");
      }//abort
    }//loop over DetMap
  //move into a vector
  region_mapping.reserve(local_mapping.size());
  for( auto eIt= local_mapping.begin();
       eIt!=local_mapping.end();++eIt)
    region_mapping.push_back(std::make_pair((*eIt).first,(*eIt).second));
}

void OnDemandMeasurementTracker::define( const edm::Handle< LazyGetter> & aLazyGetterH,
					 RefGetter & aGetter, StMeasurementDetSet &stData ) const
{
  //  define is supposed to be call by an EDProducer module, which wil put the RefGetter in the event
  //  so that reference can be made to it.
  //  the lazy getter is retrieved by the calling module and passed along with the event
  //  the map is cleared, except for pixel
  //  then the known elementIndex are defined to the RefGetter. no unpacking is done at this time
  //  the defined region range is registered in the DetODMap for further use.
  stData.resetOnDemandStrips();
 
  //define all the elementindex in the refgetter
  for(auto eIt= region_mapping.begin();
       eIt!=region_mapping.end();++eIt){
    std::pair<unsigned int, unsigned int> region_range; 
    
    //before update of the refgetter
    region_range.first = aGetter.size();
    //update the refegetter with the elementindex
    theStripRegionCabling->updateSiStripRefGetter<SiStripCluster> (aGetter, aLazyGetterH, eIt->first);
    //after update of the refgetter
    region_range.second = aGetter.size();

    LogDebug(category_)<<"between index: "<<region_range.first<<" "<<region_range.second
		       <<"\n"<<dumpRegion(region_range,aGetter,StayPacked_);
    
    //now assign to each measurement det for that element index
    for (auto dIt=eIt->second.begin();
	 dIt!=eIt->second.end();++dIt){
      const DetODStatus & elem = (*dIt)->second;
      if (elem.kind == DetODStatus::Strip) {
          stData.defineStrip(elem.index, region_range);
      }
      LogDebug(category_)<<"detId: "<<(*dIt)->first<<" in region range: "<<region_range.first<<" "<<region_range.second;
    }//loop over MeasurementDet attached to that elementIndex
  }//loop over know elementindex
}

#include <sstream>

std::string OnDemandMeasurementTracker::dumpCluster(const std::vector<SiStripCluster> ::const_iterator & begin,const  std::vector<SiStripCluster> ::const_iterator & end)const
{
  //  dumpCluster is a printout of all the clusters between the iterator. returns a string
  std::string tab="      ";
  std::stringstream ss;
  std::vector<SiStripCluster> ::const_iterator it = begin;
  unsigned int i=0;
  for (;it!=end;++it){
    ss<<tab<<i++<<") center: "<<it->barycenter()<<",id: "<<it->geographicalId()<<" with: "<<it->amplitudes().size()<<" strips\n"<<tab<<tab<<"{";
    for (unsigned int is=0;is!=it->amplitudes().size();++is){
      ss<<it->amplitudes()[is]<<" ";
    }ss<<"}\n";
  }
  return ss.str();
}

std::string OnDemandMeasurementTracker::dumpRegion(std::pair<unsigned int,unsigned int> indexes,
					     const RefGetter & theGetter,
					     bool stayPacked)const
{
  //  dumpRegion is a printout of all the clusters in a region defined on the RefGetter. returns a string
  std::stringstream ss;
  ss<<"cluster between: "<<indexes.first<<" and: "<<indexes.second<<"\n";
  for (unsigned int iRegion = indexes.first; iRegion != indexes.second; ++iRegion){    
    uint32_t reg = SiStripRegionCabling::region((theGetter)[iRegion].region());
    SiStripRegionCabling::Position pos = theStripRegionCabling->position(reg);
    SiStripRegionCabling::PositionIndex posI = theStripRegionCabling->positionIndex(reg);
    
    ss<<"Clusters for region:["<<iRegion<<"]"
      <<"\n element index: "<<(theGetter)[iRegion].region()
      <<"\n region absolute index: "<<reg
      <<"\n region position index: "<<posI.first<<" "<<posI.second
      <<"\n region position: "<<pos.first<<" "<<pos.second
      <<"\n"<< (stayPacked? " hidden to avoid unpacking." : dumpCluster((theGetter)[iRegion].begin(),(theGetter)[iRegion].end()));
  }
  return ss.str();
}

void OnDemandMeasurementTracker::assign(const TkStripMeasurementDet * smdet,
                                        const MeasurementTrackerEvent &data) const {
  //  assign is using the handle to the refgetter and the region index range to update the MeasurementDet with their clusters
 
  //// --------- we don't need to const-cast the TkStripMeasurementDet 
  ////           but we now need to const-cast the MeasurementTrackerEvent 
  StMeasurementDetSet & rwdata = const_cast<StMeasurementDetSet &>(data.stripData());

  DetId id = smdet->rawId();
  
  LogDebug(category_)<<"assigning: "<<id.rawId();

    rwdata.setUpdated(smdet->index());

    if (!data.stripData().rawInactiveStripDetIds().empty() && std::binary_search(data.stripData().rawInactiveStripDetIds().begin(), data.stripData().rawInactiveStripDetIds().end(), id)) {
      smdet->setActiveThisEvent(rwdata, false); 
      return;
    }

    //retrieve the region range index for this module
    const std::pair<unsigned int,unsigned int> & indexes = data.stripData().regionRange(smdet->index());

    //this printout will trigger unpacking. no problem. it is done on the next regular line (find(id.rawId())
    LogDebug(category_)<<"between index: "<<indexes.first<<" and: "<<indexes.second
		       <<"\nretrieved for module: "<<id.rawId()
		       <<"\n"<<dumpRegion(indexes,data.stripData().refGetter());
    
    //look for iterator range in the regions defined for that module
    for (unsigned int iRegion = indexes.first; iRegion != indexes.second; ++iRegion){
      RefGetter::record_pair range = data.stripData().refGetter()[iRegion].find(id.rawId());
      if (range.first!=range.second){
	//	found something not empty
	//update the measurementDet
	smdet->update(rwdata, range.first, range.second);
	LogDebug(category_)<<"Valid clusters for: "<<id.rawId()
			   <<"\nnumber of regions defined here: "<< indexes.second-indexes.first
			   <<"\n"<<dumpCluster(range.first,range.second);
	/* since theStripsToSkip is a "static" pointer of the MT, no need to set it at all time.
	  if (selfUpdateSkipClusters_){
	  //assign skip clusters
	  smdet->setClusterToSkip(&theStripsToSkip);
	}
	*/
	//and you are done
	return;}
    }//loop over regions, between indexes

    //if reached. no cluster are found. set the TkStripMeasurementDet to be empty
    smdet->setEmpty(rwdata);

}



const MeasurementDet * 
OnDemandMeasurementTracker::idToDetBare(const DetId& id, const MeasurementTrackerEvent &data) const 
{
  //  overloaded from MeasurementTracker
  //  find the detid. if not found throw exception
  //  if already updated: always for pixel or strip already queried. return it
  DetODContainer::const_iterator it = theDetODMap.find(id);
  if ( it != theDetODMap.end()) {
    switch (it->second.kind) {
        case DetODStatus::Pixel:
                // nothing to do
            break;
        case DetODStatus::Strip: {
                const TkStripMeasurementDet*  theConcreteDet = static_cast<const TkStripMeasurementDet*>(it->second.mdet);
                assert(data.stripData().stripDefined(theConcreteDet->index()));
                if (!data.stripData().stripUpdated(theConcreteDet->index())) assign(theConcreteDet, data);
            }
            break;
        case DetODStatus::Glued: {
                const TkGluedMeasurementDet*  theConcreteDet = static_cast<const TkGluedMeasurementDet*>(it->second.mdet);
                int imono = theConcreteDet->monoDet()->index(), istereo = theConcreteDet->stereoDet()->index();
                assert(data.stripData().stripDefined(imono) && data.stripData().stripDefined(istereo));
                if (!data.stripData().stripUpdated(imono))   assign(theConcreteDet->monoDet(), data);
                if (!data.stripData().stripUpdated(istereo)) assign(theConcreteDet->stereoDet(), data);
            }
            break;
        }
        return it->second.mdet;
  } 
  else{
    //throw excpetion
    edm::LogError(category_)<<"failed to find the MeasurementDet for: "<<id.rawId();
    throw MeasurementDetException("failed to find the MeasurementDet for: <see message logger>");
  }
  return 0;
}


