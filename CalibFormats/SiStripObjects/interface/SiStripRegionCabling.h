#ifndef CalibTracker_SiStripConnectivity_SiStripRegionCabling_H
#define CalibTracker_SiStripConnectivity_SiStripRegionCabling_H
#define _USE_MATH_DEFINES

#include <boost/cstdint.hpp>

//Cond Formats
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

//stl
#include <vector>
#include <map>
#include <cmath>

class SiStripRegionCabling {

 public:

  typedef std::map< uint32_t, std::vector<FedChannelConnection> > RegionMap;
  typedef std::vector< RegionMap > RegionCabling;
  typedef uint32_t Region;
  typedef std::vector<Region> Regions;
  typedef std::pair<double,double> Position;
  typedef std::pair<uint32_t,uint32_t> PositionIndex;

  ~SiStripRegionCabling() {;}
  SiStripRegionCabling(const uint32_t,const uint32_t, const double);

  inline void setRegionCabling(const RegionCabling&);

  inline const RegionCabling& getRegionCabling() const;

  inline const std::pair<double,double> regionDimensions() const;

  inline const Position position(Region) const;

  inline const Position position(PositionIndex) const;

  inline const PositionIndex positionIndex(Region) const;

  const PositionIndex positionIndex(Position) const;

  const Region region(Position) const;
  
  inline const Region region(PositionIndex) const;

  const Regions regions(Position, double dR) const;

 private:

  SiStripRegionCabling() {;}

  /** Number of regions in eta,phi */
  uint32_t etadivisions_;
  uint32_t phidivisions_;

  /** Tracker extent in eta */
  double etamax_;

  /** Cabling */
  RegionCabling regioncabling_;
}; 

void SiStripRegionCabling::setRegionCabling(const RegionCabling& regioncabling) {
  regioncabling_ = regioncabling;
}

const SiStripRegionCabling::RegionCabling& SiStripRegionCabling::getRegionCabling() const {
  return regioncabling_;
}

inline const std::pair<double,double> SiStripRegionCabling::regionDimensions() const {
  return std::pair<double,double>((2.*etamax_)/etadivisions_,2.*M_PI/phidivisions_);
}

inline const SiStripRegionCabling::Position SiStripRegionCabling::position(Region region) const {
  PositionIndex index = positionIndex(region); 
  return position(index);
}

inline const SiStripRegionCabling::Position SiStripRegionCabling::position(PositionIndex index) const {
  return Position(regionDimensions().first*index.first + regionDimensions().first/2.,
		  regionDimensions().second*index.second + regionDimensions().second/2.);
}

inline const SiStripRegionCabling::PositionIndex SiStripRegionCabling::positionIndex(Region region) const {
  return PositionIndex(region/phidivisions_,region%phidivisions_);
}

inline const SiStripRegionCabling::Region SiStripRegionCabling::region(PositionIndex index) const {
  return index.first*phidivisions_ + index.second;
}

#endif
