#include "GEMCluster.h"
#include <iostream>
#include <fstream>

using namespace std;

GEMCluster::GEMCluster() : fstrip(0), lstrip(0), bunchx(0)
{
}

GEMCluster::GEMCluster(int fs, int ls, int bx) : 
  fstrip(fs), lstrip(ls), bunchx(bx)
{
}

GEMCluster::~GEMCluster()
{
}

int
GEMCluster::firstStrip() const
{
  return fstrip;
}

int
GEMCluster::lastStrip() const
{
  return lstrip;
}

int
GEMCluster::clusterSize() const
{
  return -(fstrip-lstrip)+1;
}

int
GEMCluster::bx() const
{
  return bunchx;
}

bool GEMCluster::isAdjacent(const GEMCluster& cl) const{
  
    return ((cl.firstStrip() == this->firstStrip()-1) &&
	    (cl.bx() == this->bx()));
}

void GEMCluster::merge(const GEMCluster& cl){
  
   if(this->isAdjacent(cl))
     { 
       fstrip = cl.firstStrip();  
     }
}

bool GEMCluster::operator<(const GEMCluster& cl) const{
  
if(cl.bx() == this->bx())
 return cl.firstStrip()<this->firstStrip();
else 
 return cl.bx()<this->bx();
}

bool 
GEMCluster::operator==(const GEMCluster& cl) const {
  return ( (this->clusterSize() == cl.clusterSize()) &&
	   (this->bx()          == cl.bx())          && 
	   (this->firstStrip()  == cl.firstStrip()) );
}
