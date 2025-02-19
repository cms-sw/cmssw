#ifndef TrackerMapDDDtoID_H
#define TrackerMapDDDtoID_H

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include <map>
#include <vector>
#include <string>


class DDExpandedView;
class DDFilteredView;

class TrackerMapDDDtoID {
 public:

  typedef GeometricDet::nav_type nav_type;
  typedef std::map<nav_type,uint32_t> MapType;
  typedef std::map<uint32_t,nav_type> RevMapType;

  TrackerMapDDDtoID(const GeometricDet* iDet);
  ~TrackerMapDDDtoID(){clear();}

  /*
  unsigned int id(const DDExpandedView &) const;
  //! calculate the id of a given node
  unsigned int id(const DDFilteredView &) const;
  */

  //! calculate the id of a given node
  unsigned int id(const nav_type &) const;

  nav_type const & navType(uint32_t) const;

  std::vector<nav_type> const & allNavTypes() const;
  void clear();
 private:
  void buildAll(const GeometricDet*);
  void buildAllStep2(const GeometricDet*);

  std::vector<nav_type> navVec;
  MapType path2id_;
  RevMapType revpath2id_;
};

//typedef Singleton<TrackerMapDDDtoID> TkMapDDDtoID;

#endif
