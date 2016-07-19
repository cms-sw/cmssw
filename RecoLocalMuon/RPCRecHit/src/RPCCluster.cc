#include "RPCCluster.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

RPCCluster::RPCCluster():
  fstrip(0), lstrip(0), bunchx(0),
  sumTime(0), sumTime2(0), nTime(0), sumY(0), sumY2(0), nY(0)
{
}

RPCCluster::RPCCluster(int fs, int ls, int bx) :
  fstrip(fs), lstrip(ls), bunchx(bx),
  sumTime(0), sumTime2(0), nTime(0), sumY(0), sumY2(0), nY(0)
{
}

RPCCluster::~RPCCluster() {}

int RPCCluster::firstStrip() const { return fstrip; }
int RPCCluster::lastStrip() const { return lstrip; }
int RPCCluster::clusterSize() const { return -(fstrip-lstrip)+1; }
int RPCCluster::bx() const { return bunchx; }

bool RPCCluster::hasTime() const { return nTime > 0; }
float RPCCluster::time() const { return hasTime() ? sumTime/nTime : 0; }
float RPCCluster::timeRMS2() const { return hasTime() ? (sumTime2*nTime - sumTime*sumTime)/nTime : 0; }
float RPCCluster::timeRMS() const { return sqrt(timeRMS2()); }

bool RPCCluster::hasY() const { return nY > 0; }
float RPCCluster::y() const { return hasY() ? sumY/nY : 0; }
float RPCCluster::yRMS2() const { return hasY() ? (sumY2*nY - sumY*sumY)/nY : 0; }
float RPCCluster::yRMS() const { return sqrt(yRMS2()); }

bool RPCCluster::isAdjacent(const RPCCluster& cl) const
{
  return ((cl.firstStrip() == this->firstStrip()-1) &&
	        (cl.bx() == this->bx()));
}

void RPCCluster::addTime(const float time)
{
  ++nTime;
  sumTime  += time;
  sumTime2 += time*time;
}

void RPCCluster::addY(const float y)
{
  ++nY;
  sumY  += y;
  sumY2 += y*y;
}

void RPCCluster::merge(const RPCCluster& cl)
{
  if ( !this->isAdjacent(cl) ) return;

  fstrip = cl.firstStrip();

  nTime    += cl.nTime;
  sumTime  += cl.sumTime;
  sumTime2 += cl.sumTime2;

  nY    += cl.nY;
  sumY  += cl.sumY;
  sumY2 += cl.sumY2;
}

bool RPCCluster::operator<(const RPCCluster& cl) const
{
  if(cl.bx() == this->bx()) return cl.firstStrip()<this->firstStrip();

  return cl.bx()<this->bx();
}

bool  RPCCluster::operator==(const RPCCluster& cl) const
{
  return ( (this->clusterSize() == cl.clusterSize()) &&
           (this->bx()          == cl.bx())          &&
           (this->firstStrip()  == cl.firstStrip()) );
}
