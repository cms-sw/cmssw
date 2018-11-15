#ifndef MTDMapDDDtoID_H
#define MTDMapDDDtoID_H

#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"

#include <map>
#include <vector>
#include <string>


class DDExpandedView;
class DDFilteredView;

class MTDMapDDDtoID {
 public:

  typedef GeometricTimingDet::nav_type nav_type;
  typedef std::map<nav_type,uint32_t> MapType;
  typedef std::map<uint32_t,nav_type> RevMapType;

  MTDMapDDDtoID(const GeometricTimingDet* iDet);
  ~MTDMapDDDtoID(){clear();}
  
  //! calculate the id of a given node
  unsigned int id(const nav_type &) const;

  nav_type const & navType(uint32_t) const;

  std::vector<nav_type> const & allNavTypes() const;
  void clear();
 private:
  void buildAll(const GeometricTimingDet*);
  void buildAllStep2(const GeometricTimingDet*);

  std::vector<nav_type> navVec;
  MapType path2id_;
  RevMapType revpath2id_;
};

//typedef Singleton<MTDMapDDDtoID> TkMapDDDtoID;

#endif
