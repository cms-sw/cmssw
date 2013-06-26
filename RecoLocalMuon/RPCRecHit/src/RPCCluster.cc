#include "RPCCluster.h"
#include <iostream>
#include <fstream>

using namespace std;

RPCCluster::RPCCluster() : fstrip(0), lstrip(0), bunchx(0)
{
}

RPCCluster::RPCCluster(int fs, int ls, int bx) : 
  fstrip(fs), lstrip(ls), bunchx(bx)
{
}

RPCCluster::~RPCCluster()
{
}

int
RPCCluster::firstStrip() const
{
  return fstrip;
}

int
RPCCluster::lastStrip() const
{
  return lstrip;
}

int
RPCCluster::clusterSize() const
{
  return -(fstrip-lstrip)+1;
}

int
RPCCluster::bx() const
{
  return bunchx;
}

bool RPCCluster::isAdjacent(const RPCCluster& cl) const{
  
    return ((cl.firstStrip() == this->firstStrip()-1) &&
	    (cl.bx() == this->bx()));
}

void RPCCluster::merge(const RPCCluster& cl){
  
   if(this->isAdjacent(cl))
     { 
       fstrip = cl.firstStrip();  
     }
}

bool RPCCluster::operator<(const RPCCluster& cl) const{
  
if(cl.bx() == this->bx())
 return cl.firstStrip()<this->firstStrip();
else 
 return cl.bx()<this->bx();
}

bool 
RPCCluster::operator==(const RPCCluster& cl) const {
  return ( (this->clusterSize() == cl.clusterSize()) &&
	   (this->bx()          == cl.bx())          && 
	   (this->firstStrip()  == cl.firstStrip()) );
}
