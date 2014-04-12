#include "FastSimulation/CaloGeometryTools/interface/CaloDirectionOperations.h"


CaloDirection CaloDirectionOperations::add2d(const CaloDirection& dir1, const CaloDirection & dir2)
{
  //  unsigned d1=Side(dir1);
  //  unsigned d2=Side(dir2);
  constexpr CaloDirection tab[4][4]={{NORTH,NORTHEAST,NONE,NORTHWEST},
    			             {NORTHEAST,EAST,SOUTHEAST,NONE},
    			             {NONE,SOUTHEAST,SOUTH,SOUTHWEST},
    			             {NORTHWEST,NONE,SOUTHWEST,WEST}};
  return tab[Side(dir1)][Side(dir2)];
}

CaloDirection CaloDirectionOperations::Side(unsigned i) 
{
  constexpr CaloDirection sides[6]={NORTH,EAST,SOUTH,WEST,UP,DOWN};
  //  if(i<0||i>5) return DOWN;
  return sides[i];
}

unsigned CaloDirectionOperations::neighbourDirection(const CaloDirection& side)
{
  unsigned result;
  switch (side)
    {
    case NORTH:
      result=0;
      break;
    case EAST:
      result=1;
      break;
    case SOUTH:
      result=2;
      break;
    case WEST:
      result=3;
      break;
    case NORTHEAST:
      result=4;
      break;
    case SOUTHEAST:
      result=5;
      break;
    case SOUTHWEST:
      result=6;
      break;
    case NORTHWEST:
      result=7;
      break;
    default:
      result=999;
    }
  return result;
}


// It should be merged with the previous one. But I am afraid to break something
CaloDirection CaloDirectionOperations::neighbourDirection(unsigned i)
{
  constexpr CaloDirection sides[8]={NORTH,EAST,SOUTH,WEST,NORTHEAST,SOUTHEAST,SOUTHWEST,NORTHWEST};
  //  if(i<0||i>7) return SOUTH;
  return sides[i];
}


unsigned CaloDirectionOperations::Side(const CaloDirection& side)
{
  unsigned result;
  switch (side)
    {
    case NORTH:
      result=0;
      break;
    case EAST:
      result=1;
      break;
    case SOUTH:
      result=2;
      break;
    case WEST:
      result=3;
      break;
    case UP:
      result=4;
      break;
    case DOWN:
      result=5;
      break;
    default:
      result=999;
    }
  return result;
}


CaloDirection CaloDirectionOperations::oppositeSide(const CaloDirection& side) 
{
  CaloDirection result;
  switch (side)
    {
    case UP:
      result=DOWN;
      break;
    case DOWN:
      result=UP;
      break;
    case EAST:
      result=WEST;
      break;
    case WEST:
      result=EAST;
      break;
    case NORTH:
      result=SOUTH;
      break;
    case SOUTH:
      result=NORTH;
      break;
    case NORTHEAST:
      result=SOUTHWEST;
      break;
    case NORTHWEST:
      result=SOUTHEAST;
      break;
    case SOUTHEAST:
      result=NORTHWEST;
      break;
    case SOUTHWEST:
      result=NORTHEAST;
      break;
      
    default:
      result=NONE;
    }
  return result;
}


unsigned CaloDirectionOperations::oppositeDirection(unsigned iside)
{
  constexpr unsigned od[8]={2,3,0,1,6,7,4,5};
  return od[iside];
  //  if(iside>=0&&iside<8) return od[iside];
  //  return 999;
}
