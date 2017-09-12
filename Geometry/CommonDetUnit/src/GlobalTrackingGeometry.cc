/** \file GlobalTrackingGeometry.cc
 *
 *  \author M. Sani
 */

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>

GlobalTrackingGeometry::GlobalTrackingGeometry(std::vector<const TrackingGeometry*>& geos)
    : theGeometries(geos),
    theDetTypes(nullptr), theDetUnits(nullptr), theDets(nullptr), theDetUnitIds(nullptr), theDetIds(nullptr)
{}

GlobalTrackingGeometry::~GlobalTrackingGeometry()
{
    delete theDetTypes.load();
    theDetTypes = nullptr;
    delete theDetUnits.load();
    theDetUnits = nullptr;
    delete theDets.load();
    theDets = nullptr;
    delete theDetUnitIds.load();
    theDetUnitIds = nullptr;
    delete theDetIds.load();
    theDetIds = nullptr;
}

const GeomDetUnit* GlobalTrackingGeometry::idToDetUnit(DetId id) const {
    
    const TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != nullptr) {
      return tg->idToDetUnit(id);
    } else {
      return nullptr;
    }
}


const GeomDet* GlobalTrackingGeometry::idToDet(DetId id) const{
  
    const TrackingGeometry* tg = slaveGeometry(id);
    
    if (tg != nullptr) {
        return tg->idToDet(id);
    } else {
      return nullptr;
    }
}

const TrackingGeometry* GlobalTrackingGeometry::slaveGeometry(DetId id) const {  
  
    int idx = id.det()-1;
    if (id.det() == DetId::Muon) {
        
        idx+=id.subdetId()-1;
    }

    if (theGeometries[idx]==nullptr) throw cms::Exception("NoGeometry") << "No Tracking Geometry is available for DetId " << id.rawId() << std::endl;

    return theGeometries[idx];
}

const TrackingGeometry::DetTypeContainer&
GlobalTrackingGeometry::detTypes( void ) const
{    
   if (!theDetTypes.load(std::memory_order_acquire)) {
       std::unique_ptr<DetTypeContainer> ptr{new DetTypeContainer()};
       for(auto theGeometrie : theGeometries)
       {
        if( theGeometrie == nullptr ) continue;
        DetTypeContainer detTypes(theGeometrie->detTypes());
        if( detTypes.size() + ptr->size() < ptr->capacity()) ptr->resize( detTypes.size() + ptr->size());
        for(auto detType : detTypes)
          ptr->emplace_back( detType );
       }
       DetTypeContainer* expect = nullptr;
       if(theDetTypes.compare_exchange_strong(expect, ptr.get(), std::memory_order_acq_rel)) {
           ptr.release();
       }
   }
   return *theDetTypes.load(std::memory_order_acquire);
}

const TrackingGeometry::DetUnitContainer&
GlobalTrackingGeometry::detUnits( void ) const
{
   if (!theDetUnits.load(std::memory_order_acquire)) {
       std::unique_ptr<DetUnitContainer> ptr{new DetUnitContainer()};
       for(auto theGeometrie : theGeometries)
       {
        if( theGeometrie == nullptr ) continue;
        DetUnitContainer detUnits(theGeometrie->detUnits());
        if( detUnits.size() + ptr->size() < ptr->capacity()) ptr->resize( detUnits.size() + ptr->size());
        for(auto detUnit : detUnits)
          ptr->emplace_back( detUnit );
       }
       DetUnitContainer* expect = nullptr;
       if(theDetUnits.compare_exchange_strong(expect, ptr.get(), std::memory_order_acq_rel)) {
           ptr.release();
       }
   }
   return *theDetUnits.load(std::memory_order_acquire);
}

const TrackingGeometry::DetContainer&
GlobalTrackingGeometry::dets( void ) const
{
   if (!theDets.load(std::memory_order_acquire)) {
       std::unique_ptr<DetContainer> ptr{new DetContainer()};
       for(auto theGeometrie : theGeometries)
       {
        if( theGeometrie == nullptr ) continue;
        DetContainer dets(theGeometrie->dets());
        if( dets.size() + ptr->size() < ptr->capacity()) ptr->resize( dets.size() + ptr->size());
        for(auto det : dets)
          ptr->emplace_back( det );
       }
       DetContainer* expect = nullptr;
       if(theDets.compare_exchange_strong(expect, ptr.get(), std::memory_order_acq_rel)) {
           ptr.release();
       }
   }
   return *theDets.load(std::memory_order_acquire);
}

const TrackingGeometry::DetIdContainer&
GlobalTrackingGeometry::detUnitIds( void ) const
{
   if (!theDetUnitIds.load(std::memory_order_acquire)) {
       std::unique_ptr<DetIdContainer> ptr{new DetIdContainer()};
       for(auto theGeometrie : theGeometries)
       {
        if( theGeometrie == nullptr ) continue;
        DetIdContainer detUnitIds(theGeometrie->detUnitIds());
        if( detUnitIds.size() + ptr->size() < ptr->capacity()) ptr->resize( detUnitIds.size() + ptr->size());
        for(auto detUnitId : detUnitIds)
          ptr->emplace_back( detUnitId );
       }
       DetIdContainer* expect = nullptr;
       if(theDetUnitIds.compare_exchange_strong(expect, ptr.get(), std::memory_order_acq_rel)) {
           ptr.release();
       }
   }
   return *theDetUnitIds.load(std::memory_order_acquire);
}

const TrackingGeometry::DetIdContainer&
GlobalTrackingGeometry::detIds( void ) const
{
   if (!theDetIds.load(std::memory_order_acquire)) {
       std::unique_ptr<DetIdContainer> ptr{new DetIdContainer()};
       for(auto theGeometrie : theGeometries)
       {
        if( theGeometrie == nullptr ) continue;
        DetIdContainer detIds(theGeometrie->detIds());
        if( detIds.size() + ptr->size() < ptr->capacity()) ptr->resize( detIds.size() + ptr->size());
        for(auto detId : detIds)
          ptr->emplace_back( detId );
       }
       DetIdContainer* expect = nullptr;
       if(theDetIds.compare_exchange_strong(expect, ptr.get(), std::memory_order_acq_rel)) {
           ptr.release();
       }
   }
   return *theDetIds.load(std::memory_order_acquire);
}
