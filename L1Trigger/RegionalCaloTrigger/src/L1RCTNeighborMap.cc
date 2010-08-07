#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTNeighborMap.h"

#include <vector>
using std::vector;

vector<int> L1RCTNeighborMap::north(int crate,int card,int region){
  std::vector<int> north(3);
  int newregion(0),newcard(0),newcrate(0);
  if(card == 0 || card == 2 || card == 4){
    newcard = card+1;
    newregion = region;
    if(crate != 0 && crate != 9)
      newcrate = crate-1;
    else
      newcrate = crate+8;
  }
  else if(card == 1 || card == 3 || card == 5){
    newcard = card-1;
    newregion = region;
    newcrate = crate;
  }
  else if(card == 6){
    if(region == 0){
      newcard = card;
      newregion = region+1;
      if(crate != 0 && crate != 9)
	newcrate = crate-1;
      else
	newcrate = crate+8;
    }
    else {
      newcard = card;
      newregion = region-1;
      newcrate = crate;
    }
  }
  north.at(0) = newcrate;
  north.at(1) = newcard;
  north.at(2) = newregion;
  return north;
}

vector<int> L1RCTNeighborMap::south(int crate, int card, int region){
  int newcrate(0),newcard(0),newregion(0);
  std::vector<int> south(3);
  if(card == 0 || card == 2 || card == 4){
    newcard = card+1;
    newregion = region;
    newcrate = crate;
  }
  else if(card == 1 || card == 3 || card == 5){
    newcard = card-1;
    newregion = region;
    if(crate != 8 && crate != 17)
      newcrate = crate+1;
    else
      newcrate = crate-8;
  }
  else if(card == 6){
    if(region == 0){
      newcrate = crate;
      newcard = card;
      newregion = region+1;
    }
    else {
      newcard = card;
      newregion = region-1;
      if(crate != 8 && crate != 17)
	newcrate = crate+1;
      else
	newcrate = crate-8;
    }
  }
  south.at(0) = newcrate;
  south.at(1) = newcard;
  south.at(2) = newregion;
  return south;
}
  
vector<int> L1RCTNeighborMap::west(int crate,int card, int region){
  int newcrate(0),newcard(0),newregion(0);
  std::vector<int> west(3);
  if(crate < 9){
    newcrate = crate;
    if(card != 6){
      if(region == 0){
	newcard = card;
	newregion = region+1;
      }
      else{
	if(card != 4 && card != 5){
	  newcard = card+2;
	  newregion = region-1;
	}
	else if(card == 4){
	  newcard = 6;
	  newregion = 0;
	}
	else if(card == 5){
	  newcard = 6;
	  newregion = 1;
	}
      }
    }
    else{
      newcrate = -1;
      newcard = -1;
      newregion = -1;
    }
  }
  else{
    if(card == 0 || card == 1){
      if(region == 0){
	newcrate = crate-9;
	newregion = region;
	newcard = card;
      }
      else {
	newcrate = crate;
	newregion = region-1;
	newcard = card;
      }
    }
    else if(card != 6){
      newcrate = crate;
      if(region == 0){
	newcard = card-2;
	newregion = region+1;
      }
      else{
	newcard = card;
	newregion = region-1;
      }
    }
    else if(card == 6){
      newcrate = crate;
      newregion = 1;
      if(region == 0)
	newcard = 4;
      else
	newcard = 5;
    }
  }
  west.at(0) = newcrate;
  west.at(1) = newcard;
  west.at(2) = newregion;
  return west;
}

vector<int> L1RCTNeighborMap::east(int crate,int card, int region){
  int newcrate(0),newcard(0),newregion(0);
  std::vector<int> east(3);
  if(crate < 9){
    if(card != 0 && card != 1 && card != 6){
      newcrate = crate;
      if(region == 0){
	newregion = region+1;
	newcard = card-2;
      }
      else{
	newregion = region-1;
	newcard = card;
      }
    }
    else if(card == 0 || card == 1){
      newcard = card;
      if(region == 0){
	newcrate = crate+9;
	newregion = region;
      }
      else {
	newcrate = crate;
	newregion = region-1;
      }
    }
    else if(card == 6){
      newcrate = crate;
      newregion = 1;
      if(region == 0)
	newcard = 4;
      else
	newcard = 5;
    }
  }
  else {
    newcrate = crate;
    if(card != 6){
      if(region == 0){
	newcard = card;
	newregion = region+1;
      }
      else{
	if(card != 4 && card != 5){
	  newcard = card+2;
	  newregion = region-1;
	}
	else if(card == 4){
	  newcard = 6;
	  newregion = 0;
	}
	else if(card == 5){
	  newcard = 6;
	  newregion = 1;
	}
      }
    }
    else{
      newcrate = -1;
      newcard = -1;
      newregion = -1;
    }
  }
  east.at(0) = newcrate;
  east.at(1) = newcard;
  east.at(2) = newregion;
  return east;
}

vector<int> L1RCTNeighborMap::se(int crate,int card,int region){
  int newcrate(0),newcard(0),newregion(0);
  std::vector<int> se(3);
  if(crate < 9){
    if(card == 0){
      if(region == 0){
	newcrate = crate+9;
	newregion = region;
	newcard = 1;
      }
      else{
	newcrate = crate;
	newregion = 0;
	newcard = 1;
      }
    }
    else if(card == 1){
      if(region == 0){
	if(crate != 8)
	  newcrate = crate+10;
	else 
	  newcrate = 9;
	newregion = 0;
	newcard = 0;
      }
      else {
	if(crate != 8)
	  newcrate = crate+1;
	else
	  newcrate = 0;
	newregion = 0;
	newcard = 0;
      }
    }
    else if(card == 2 || card == 4){
      newcrate = crate;
      newregion = !region;
      newcard = card-1+2*region;
    }
    else if(card == 5 || card == 3){
      newregion = !region;
      newcard = card-3+2*region;
      if(crate != 8)
	newcrate = crate+1;
      else
	newcrate = 0;
    }
    else if(card == 6){
      if(region == 0){
	newcard = 5;
	newregion = 1;
	newcrate = crate;
      }
      else{
	newcard = 4;
	newregion = 1;
	if(crate != 8)
	  newcrate = crate+1;
	else
	  newcrate = 0;
      }
    }
  }
  else{
    if(card == 0 || card == 2){
      newregion = !region;
      newcrate = crate;
      newcard = card+2*region+1;
    }
    else if(card == 1 || card == 3){
      newregion = !region;
      newcard = card-1+2*region;
      if(crate != 17)
	newcrate = crate+1;
      else
	newcrate = 9;
    }
    else if(card == 4){
      newcrate = crate;
      if(region == 0){
	newregion = 1;
	newcard = 5;
      }
      else{
	newregion = 1;
	newcard = 6;
      }
    }
    else if(card == 5){
      if(region == 0){
	newcard = 4;
	newregion = 1;
      }
      else{
	newcard = 6;
	newregion = 0;
      }
      if(crate != 17)
	newcrate = crate+1;
      else 
	newcrate = 9;
    }
    else if(card == 6){
      newcard = -1;
      newregion = -1;
      newcrate = -1;
    }
  }

  se.at(0) = newcrate;
  se.at(1) = newcard;
  se.at(2) = newregion;
  return se;
}

vector<int> L1RCTNeighborMap::sw(int crate,int card,int region){
  int newcrate(0),newcard(0),newregion(0);
  std::vector<int> sw(3);
  if(crate > 8){
    if(card == 0){
      if(region == 0){
	newcrate = crate-9;
	newregion = region;
	newcard = 1;
      }
      else{
	newcrate = crate;
	newregion = 0;
	newcard = 1;
      }
    }
    else if(card == 1){
      if(region == 0){
	if(crate != 17)
	  newcrate = crate-8;
	else
	  newcrate = 0;
	newregion = 0;
	newcard = 0;
      }
      else {
	if(crate != 17)
	  newcrate = crate+1;
	else
	  newcrate = 9;
	newregion = 0;
	newcard = 0;
      }
    }
    else if(card == 2 || card == 4){
      newcrate = crate;
      newregion = !region;
      newcard = card-1+2*region;
    }
    else if(card == 5 || card == 3){
      newregion = !region;
      newcard = card-3+2*region;
      if(crate != 17)
	newcrate = crate+1;
      else
	newcrate = 9;
    }
    else if(card == 6){
      if(region == 0){
	newcard = 5;
	newregion = 1;
	newcrate = crate;
      }
      else{
	newcard = 4;
	newregion = 1;
	if(crate != 17)
	  newcrate = crate+1;
	else
	  newcrate = 9;
      }
    }
  }
  else{
    if(card == 0 || card == 2){
      newregion = !region;
      newcrate = crate;
      newcard = card+1+2*region;
    }
    else if(card == 1 || card == 3){
      newregion = !region;
      newcard = card-1+2*region;
      if(crate != 8)
	newcrate = crate+1;
      else
	newcrate = 0;
    }
    else if(card == 4){
      newcrate = crate;
      if(region == 0){
	newregion = 1;
	newcard = 5;
      }
      else{
	newregion = 1;
	newcard = 6;
      }
    }
    else if(card == 5){
      if(region == 0){
	newcard = 4;
	newregion = 1;
      }
      else{
	newcard = 6;
	newregion = 0;
      }
      if(crate != 8)
	newcrate = crate+1;
      else 
	newcrate = 0;
    }
    else if(card == 6){
      newcard = -1;
      newregion = -1;
      newcrate = -1;
    }
  }

  sw.at(0) = newcrate;
  sw.at(1) = newcard;
  sw.at(2) = newregion;
  return sw;
}

vector<int> L1RCTNeighborMap::ne(int crate,int card,int region){
  int newcrate(0),newcard(0),newregion(0);
  std::vector<int> ne(3);
  if(crate < 9){
    if(card == 0){
      newregion = 0;
      newcard = 1;
      if(region == 0){
	if(crate != 0)
	  newcrate = crate +8;
	else
	  newcrate = 17;
      }
      else{
	if(crate != 0)
	  newcrate = crate-1;
	else
	  newcrate = 8;
      }
    }
    else if(card == 1){
      newregion = 0;
      newcard = 0;
      if(region == 0)
	newcrate = crate+9;
      else
	newcrate = crate;
    }
    else if(card == 2 || card == 4){
      newregion = !region;
      newcard = card-1+2*region;
      if(crate != 0)
	newcrate = crate-1;
      else
	newcrate = 8;
    }
    else if(card == 5 || card == 3){
      newregion = !region;
      newcard = card-3+2*region;
      newcrate = crate;
    }
    else if(card == 6){
      newregion = 1;
      if(region==0){
	newcard = 5;
	if(crate != 0)
	  newcrate = crate-1;
	else
	  newcrate = 8;
      }
      else {
	newcard = 4;
	newcrate = crate;
      }
    }
  }
  else {
    if(card == 0 || card ==2){
      newregion = !region;
      newcard = card+1+2*region;
      if(crate!=9)
	newcrate = crate-1;
      else
	newcrate = 17;
    }
    else if(card == 1 || card == 3){
      newregion = !region;
      newcard = card-1+2*region;
      newcrate = crate;
    }
    else if(card == 4){
      newregion = 1;
      if(crate != 9)
	newcrate = crate-1;
      else
	newcrate = 17;
      if(region == 0)
	newcard = 5;
      else
	newcard = 6;
    }
    else if(card == 5){
      newregion = !region;
      newcrate = crate;
      if(region == 0)
	newcard = 4;
      else
	newcard = 6;
    }
    else if(card == 6){
      newcrate = -1;
      newcard = -1;
      newregion = -1;
    }
  }
  ne.at(0) = newcrate;
  ne.at(1) = newcard;
  ne.at(2) = newregion;
  return ne;
}

vector<int> L1RCTNeighborMap::nw(int crate,int card,int region){
  int newcrate(0),newcard(0),newregion(0);
  std::vector<int> nw(3);
  if(crate > 8){
    if(card == 0){
      newregion = 0;
      newcard = 1;
      if(region == 0){
	if(crate != 9)
	  newcrate = crate -10;
	else
	  newcrate = 8;
      }
      else{
	if(crate != 9)
	  newcrate = crate-1;
	else
	  newcrate = 17;
      }
    }
    else if(card == 1){
      newregion = 0;
      newcard = 0;
      if(region == 0)
	newcrate = crate-9;
      else
	newcrate = crate;
    }
    else if(card == 2 || card == 4){
      newregion = !region;
      newcard = card-1+2*region;
      if(crate != 9)
	newcrate = crate-1;
      else
	newcrate = 17;
    }
    else if(card == 5 || card == 3){
      newregion = !region;
      newcard = card-3+2*region;
      newcrate = crate;
    }
    else if(card == 6){
      newregion = 1;
      if(region==0){
	newcard = 5;
	if(crate != 9)
	  newcrate = crate-1;
	else
	  newcrate = 17;
      }
      else {
	newcard = 4;
	newcrate = crate;
      }
    }
  }
  else {
    if(card == 0 || card ==2){
      newregion = !region;
      newcard = card+1+2*region;
      if(crate!=0)
	newcrate = crate-1;
      else
	newcrate = 8;
    }
    else if(card == 1 || card == 3){
      newregion = !region;
      newcard = card-1+2*region;
      newcrate = crate;
    }
    else if(card == 4){
      newregion = 1;
      if(crate != 0)
	newcrate = crate-1;
      else
	newcrate = 8;
      if(region == 0)
	newcard = 5;
      else
	newcard = 6;
    }
    else if(card == 5){
      newregion = !region;
      newcrate = crate;
      if(region == 0)
	newcard = 4;
      else
	newcard = 6;
    }
    else if(card == 6){
      newcrate = -1;
      newcard = -1;
      newregion = -1;
    }
  }
  nw.at(0) = newcrate;
  nw.at(1) = newcard;
  nw.at(2) = newregion;
  return nw;
}
