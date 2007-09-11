#ifndef CalibTracker_SiStripConnectivity_SiStripRegionCabling_H
#define CalibTracker_SiStripConnectivity_SiStripRegionCabling_H
#define _USE_MATH_DEFINES

#include <boost/cstdint.hpp>
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"
#include "DataFormats/Common/interface/Handle.h"
#include <vector>
#include <map>
#include <cmath>

/**
   Author: pwing
   Package: CalibFormats/SiStripObjects
   Class: SiStripRegionCabling
   Description: Gives a regional view of the silicon strip tracker cabling. 
   Cabling is divided into (eta,phi) "regions". A "region" within a given 
   sub-detector is called a "wedge". A layer within a given wedge is called 
   an "element".
*/

class SiStripRegionCabling {

 public:

  /** enums */
  enum SubDet {TIB = 0, TOB = 1, TID = 2, TEC = 3, ALLSUBDETS = 4}; 
  enum Layer {TIBLAYERS = 4, TOBLAYERS = 6, TIDLAYERS = 3, TECLAYERS = 9, ALLLAYERS = 10}; 

  /** Cabling typedefs */
  typedef std::map< uint32_t, std::vector<FedChannelConnection> > ElementCabling;
  typedef std::vector< ElementCabling > WedgeCabling;
  typedef std::vector< WedgeCabling > RegionCabling;
  typedef std::vector< RegionCabling > Cabling;

  /** Position typedefs */
  typedef std::pair<double,double> Position;
  typedef std::pair<uint32_t,uint32_t> PositionIndex;

  /** Encoded information typedefs */
  typedef uint32_t ElementIndex;

  SiStripRegionCabling(const uint32_t,const uint32_t, const double);

  ~SiStripRegionCabling() {}

  /** Set and get methods for cabling. */

  inline void setRegionCabling(const Cabling&);

  inline const Cabling& getRegionCabling() const;

  inline const uint32_t etadivisions() const;

  inline const uint32_t phidivisions() const;

  /** Methods for interchanging between region, region-index and 
      eta/phi-position. */

  inline const std::pair<double,double> regionDimensions() const;

  inline const Position position(const uint32_t) const;

  inline const Position position(const PositionIndex) const;

  inline const PositionIndex positionIndex(const uint32_t) const;

  const PositionIndex positionIndex(const Position) const;

  const uint32_t region(const Position) const;
  
  inline const uint32_t region(const PositionIndex) const;

  /** Method for incrementing position index. */

  PositionIndex increment(const PositionIndex, int, int) const;

  /** Methods for interchanging between region-subdet-layer and the 
      corresponding element index. */

  inline static const ElementIndex elementIndex(const uint32_t region, const SubDet, const uint32_t layer);

  inline const ElementIndex elementIndex(const PositionIndex, const SubDet, const uint32_t layer) const;

  inline const ElementIndex elementIndex(const Position, const SubDet, const uint32_t layer) const;

  inline static const uint32_t layer(const ElementIndex);
  
  inline static const SubDet subdet(const ElementIndex);
  
  inline static const uint32_t region(const ElementIndex);
 
  /** Methods for extracting det-id information */

  static const SubDet subdetFromDetId(const uint32_t detid);

  static const uint32_t layerFromDetId(const uint32_t detid);

  static const uint32_t physicalLayerFromDetId(const uint32_t detid);

  static const uint32_t physicalLayer(const SubDet, const uint32_t layer);

  /** Methods for updating a SiStripRefGetter<T> container with elements 
      of interest  */
  
  template <class T>
    void updateSiStripRefGetter(edm::SiStripRefGetter<T>& refgetter, 
				const edm::Handle< edm::SiStripLazyGetter<T> >& lazygetter, 
				const ElementIndex) const;
  
  template <class T>
    void updateSiStripRefGetter(edm::SiStripRefGetter<T>& refgetter, 
				const edm::Handle< edm::SiStripLazyGetter<T> >& lazygetter,
				const Position position, 
				const double deltaeta, 
				const double deltaphi, 
				const SubDet subdet, 
				const uint32_t layer) const;
  
 private:
 
 SiStripRegionCabling() {;}
 
 /** Number of regions in eta,phi */
 int etadivisions_;
 int phidivisions_;
 
 /** Tracker extent in eta */
 double etamax_;
 
 /** Cabling */
 Cabling regioncabling_;
}; 

inline void SiStripRegionCabling::setRegionCabling(const Cabling& regioncabling) {
  regioncabling_ = regioncabling;
}

inline const SiStripRegionCabling::Cabling& SiStripRegionCabling::getRegionCabling() const {
  return regioncabling_;
}

inline const uint32_t SiStripRegionCabling::etadivisions() const {
  return static_cast<uint32_t>(etadivisions_);
}

inline const uint32_t SiStripRegionCabling::phidivisions() const {
  return static_cast<uint32_t>(phidivisions_);
}

inline const std::pair<double,double> SiStripRegionCabling::regionDimensions() const {
  return std::pair<double,double>((2.*etamax_)/etadivisions_,2.*M_PI/phidivisions_);
}

inline const SiStripRegionCabling::Position SiStripRegionCabling::position(const uint32_t region) const {
  PositionIndex index = positionIndex(region); 
  return position(index);
}

inline const SiStripRegionCabling::Position SiStripRegionCabling::position(const PositionIndex index) const {
  return Position(regionDimensions().first*(index.first+.5) - etamax_, regionDimensions().second*(index.second+.5)- M_PI);
}

inline const SiStripRegionCabling::PositionIndex SiStripRegionCabling::positionIndex(const uint32_t region) const {
  return PositionIndex(region/phidivisions_,region%phidivisions_);
}

inline const uint32_t SiStripRegionCabling::region(const PositionIndex index) const {
  return index.first*phidivisions_ + index.second;
}
  
inline const uint32_t SiStripRegionCabling::elementIndex(const uint32_t region, const SubDet subdet, const uint32_t layer) {
  return region*ALLSUBDETS*ALLLAYERS + subdet*ALLLAYERS + layer;
}

inline const uint32_t SiStripRegionCabling::elementIndex(const PositionIndex index, const SubDet subdet, const uint32_t layer) const {
  return elementIndex(region(index),subdet,layer);
}

inline const uint32_t SiStripRegionCabling::elementIndex(const Position position, const SubDet subdet, const uint32_t layer) const {
  return elementIndex(region(position),subdet,layer);
}

inline const uint32_t SiStripRegionCabling::layer(const uint32_t index) {
  return index%ALLLAYERS;
}
  
inline const SiStripRegionCabling::SubDet SiStripRegionCabling::subdet(const uint32_t index) {
  return static_cast<SiStripRegionCabling::SubDet>((index/ALLLAYERS)%ALLSUBDETS);
}
  
inline const uint32_t SiStripRegionCabling::region(const uint32_t index) {
  return index/(ALLSUBDETS*ALLLAYERS);
}

template <class T>
void SiStripRegionCabling::updateSiStripRefGetter(edm::SiStripRefGetter<T>& refgetter, const edm::Handle< edm::SiStripLazyGetter<T> >& lazygetter, const uint32_t index) const {
  refgetter.push_back(lazygetter,index);
}

template <class T>
void SiStripRegionCabling::updateSiStripRefGetter(edm::SiStripRefGetter<T>& refgetter, const edm::Handle< edm::SiStripLazyGetter<T> >& lazygetter, const SiStripRegionCabling::Position pos, const double deltaeta, const double deltaphi, const SubDet subdet, const uint32_t layer) const {

  PositionIndex index = positionIndex(pos);
  Position center = position(index);
  Position offset(pos.first-center.first,pos.second-center.second);
  double regionseta = deltaeta/regionDimensions().first;
  double regionsphi = deltaphi/regionDimensions().second; 
  uint32_t pluseta = static_cast<uint32_t>(offset.first+.5+regionseta);
  uint32_t plusphi = static_cast<uint32_t>(offset.second+.5+regionsphi);
  uint32_t minuseta = static_cast<uint32_t>(-offset.first+.5+regionseta);
  uint32_t minusphi = static_cast<uint32_t>(-offset.second+.5+regionsphi);
 
  for (uint32_t ieta = 0; ieta < minuseta+pluseta + 1; ieta++) {
    for (uint32_t iphi = 0; iphi < minusphi+plusphi + 1; iphi++) {
      updateSiStripRefGetter<T>(refgetter,lazygetter,elementIndex(increment(index,-minuseta+ieta,-minusphi+iphi),subdet,layer));
    }
  } 
}

#endif
