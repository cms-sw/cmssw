#include "CPPFCluster.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

CPPFCluster::CPPFCluster()
    : fstrip(0), lstrip(0), bunchx(0), sumTime(0), sumTime2(0), nTime(0), sumY(0), sumY2(0), nY(0) {}

CPPFCluster::CPPFCluster(int fs, int ls, int bx)
    : fstrip(fs), lstrip(ls), bunchx(bx), sumTime(0), sumTime2(0), nTime(0), sumY(0), sumY2(0), nY(0) {}

CPPFCluster::~CPPFCluster() {}

int CPPFCluster::firstStrip() const { return fstrip; }
int CPPFCluster::lastStrip() const { return lstrip; }
int CPPFCluster::clusterSize() const { return lstrip - fstrip + 1; }
int CPPFCluster::bx() const { return bunchx; }

bool CPPFCluster::hasTime() const { return nTime > 0; }
float CPPFCluster::time() const { return hasTime() ? sumTime / nTime : 0; }
float CPPFCluster::timeRMS() const {
  return hasTime() ? sqrt(max(0.F, sumTime2 * nTime - sumTime * sumTime)) / nTime : -1;
}

bool CPPFCluster::hasY() const { return nY > 0; }
float CPPFCluster::y() const { return hasY() ? sumY / nY : 0; }
float CPPFCluster::yRMS() const { return hasY() ? sqrt(max(0.F, sumY2 * nY - sumY * sumY)) / nY : -1; }

bool CPPFCluster::isAdjacent(const CPPFCluster& cl) const {
  return ((cl.firstStrip() == this->firstStrip() - 1) && (cl.bx() == this->bx()));
}

void CPPFCluster::addTime(const float time) {
  ++nTime;
  sumTime += time;
  sumTime2 += time * time;
}

void CPPFCluster::addY(const float y) {
  ++nY;
  sumY += y;
  sumY2 += y * y;
}

void CPPFCluster::merge(const CPPFCluster& cl) {
  if (!this->isAdjacent(cl))
    return;

  fstrip = cl.firstStrip();

  nTime += cl.nTime;
  sumTime += cl.sumTime;
  sumTime2 += cl.sumTime2;

  nY += cl.nY;
  sumY += cl.sumY;
  sumY2 += cl.sumY2;
}

bool CPPFCluster::operator<(const CPPFCluster& cl) const {
  if (cl.bx() == this->bx())
    return cl.firstStrip() < this->firstStrip();

  return cl.bx() < this->bx();
}

bool CPPFCluster::operator==(const CPPFCluster& cl) const {
  return ((this->clusterSize() == cl.clusterSize()) && (this->bx() == cl.bx()) &&
          (this->firstStrip() == cl.firstStrip()));
}
