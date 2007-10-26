#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include <cassert>

std::vector<DetId> CaloSubdetectorTopology::getWindow(const DetId& id, const int& northSouthSize, const int& eastWestSize) const
{
  
  std::vector<DetId> cellsInWindow;
  // check pivot
  if (id.null())
    return cellsInWindow;

  //
  DetId myTmpId(id);
  std::vector<std::pair<Coordinate,DetId> > fringe;
  fringe.push_back(std::pair<Coordinate,DetId>(Coordinate(0,0),myTmpId));
  
  int halfWestEast = eastWestSize/2 ;
  int halfNorthSouth = northSouthSize/2 ;

  std::vector<CellInfo> visited_cells;
  visited_cells.resize(northSouthSize * eastWestSize);
  
  while (fringe.size() > 0)
    {
      std::pair<Coordinate,DetId> cur = fringe.back();
      fringe.pop_back();
      
      // check all four neighbours
      const CaloDirection directions[4] = { NORTH, SOUTH, EAST, WEST };
      
      for (unsigned dirnum = 0; dirnum < 4; ++dirnum)
        {
          Coordinate neighbour = getNeighbourIndex(cur.first,directions[dirnum]);
	  //If outside the window range
	  if ( neighbour.first < -halfWestEast ||
	       neighbour.first > halfWestEast ||
	       neighbour.second < -halfNorthSouth ||
	       neighbour.second > halfNorthSouth )
	    continue;
	  
	  
	  //Found integer index in the matrix
	  unsigned int_index =  neighbour.first + halfWestEast +  
	    eastWestSize * (neighbour.second + halfNorthSouth );
	  assert(int_index < visited_cells.size());
	  
          // check whether we have seen this neighbour already
          if (visited_cells[int_index].visited)
            // we have seen this one already
            continue;
              
          // a new cell, get the DetId of the neighbour, mark it
          // as visited and add it to the fringe
          visited_cells[int_index].visited = true;
	  std::vector<DetId> neighbourCells = getNeighbours(cur.second,directions[dirnum]);

	  if ( neighbourCells.size() == 1 )
	    visited_cells[int_index].cell = neighbourCells[0];
	  else if ( neighbourCells.size() == 0 )
	    visited_cells[int_index].cell = DetId(0);
	  else
	    throw cms::Exception("getWindowError") << "Not supported subdetector for getWindow method";
	  
          if (!visited_cells[int_index].cell.null())
            fringe.push_back(std::pair<Coordinate,DetId>(neighbour,visited_cells[int_index].cell));
		  
	} // loop over all possible directions
    } // while some cells are left on the fringe
  
  
  for (unsigned int i=0; i<visited_cells.size(); i++)
    if (!visited_cells[i].cell.null())
      cellsInWindow.push_back(visited_cells[i].cell);
  
  return cellsInWindow;
}
