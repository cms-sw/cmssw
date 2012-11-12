#ifndef TOPOLOGY_CALOTOPOLOGY_CALOSUBDETECTORTOPOLOGY_H
#define TOPOLOGY_CALOTOPOLOGY_CALOSUBDETECTORTOPOLOGY_H 1


#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
/** \class CaloSubdetectorTopology
      
$Date: 2012/10/29 07:28:55 $
$Revision: 1.5 $
\author P.Meridiani INFN Roma1
\author J. Mans - Minnesota
*/

class CaloSubdetectorTopology {
public:
  /// standard constructor
  CaloSubdetectorTopology() {};
  /// virtual destructor
  virtual ~CaloSubdetectorTopology() { }
  /// is this detid present in the Topology?
  virtual bool valid(const DetId& /*id*/) const { return false; };
  /// return a linear packed id
  virtual unsigned int detId2denseId(const DetId& /*id*/) const { return 0; }
  /// return a linear packed id
  virtual DetId denseId2detId(unsigned int /*denseid*/) const { return DetId(0); }
  /// return a count of valid cells (for dense indexing use)
  virtual unsigned int ncells() const { return 1; }
  /// return a version which identifies the given topology
  virtual int topoVersion() const { return 0; }
  /// return whether this topology is consistent with the numbering in the given topology
  virtual bool denseIdConsistent(int topoVer) const { return topoVer==topoVersion(); }

  /** Get the neighbors of the given cell in east direction*/
  virtual std::vector<DetId> east(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in west direction*/
  virtual std::vector<DetId> west(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in north direction*/
  virtual std::vector<DetId> north(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in south direction*/
  virtual std::vector<DetId> south(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in up direction (outward)*/
  virtual std::vector<DetId> up(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in down direction (inward)*/
  virtual std::vector<DetId> down(const DetId& id) const = 0;
  /** Get the neighbors of the given cell given direction*/
  virtual std::vector<DetId> getNeighbours(const DetId& id, const CaloDirection& dir) const
    {
      std::vector<DetId> aNullVector;
      switch(dir)
	{
	case NONE:
	  return aNullVector;
	  break;
	case SOUTH:
	  return south(id);
	  break;
	case NORTH:
	  return north(id);
	  break;
	case EAST:
	  return east(id);
	  break;
	case WEST:
	  return west(id);
	  break;
        default:
	  throw cms::Exception("getNeighboursError") << "Unsopported direction";
	}
      return aNullVector;
    }

  /** Get the neighbors of the given cell in a window of given size*/
  virtual std::vector<DetId> getWindow(const DetId& id, const int& northSouthSize, const int& eastWestSize) const;

  /** Get all the neighbors of the given cell*/
  virtual std::vector<DetId> getAllNeighbours(const DetId& id) const
    {
      return getWindow(id,3,3);
    }

 protected:
  typedef std::pair<int,int> Coordinate;

  struct CellInfo
  {
    bool visited;
    
    DetId cell;
    
    CellInfo() : 
      visited(false)
    {
    }
    
    CellInfo(bool a_visited, const DetId &a_cell) : 
      visited(a_visited),
	 cell(a_cell)
    {
    }
  };

  inline Coordinate getNeighbourIndex(const Coordinate &coord, const CaloDirection& dir) const
    {
      switch (dir)
        {
        case NORTH: return Coordinate(coord.first,coord.second + 1);
        case SOUTH: return Coordinate(coord.first,coord.second - 1);
	  
        case EAST:  return Coordinate(coord.first + 1,coord.second);
        case WEST:  return Coordinate(coord.first - 1,coord.second);
	  
        default:
	  throw cms::Exception("getWindowError") << "Unsopported direction";
        }
    }
  
};


#endif
