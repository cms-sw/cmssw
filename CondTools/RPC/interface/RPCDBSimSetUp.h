#ifndef RPCDBSimSetUp_h
#define RPCDBSimSetUp_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include<cstring>
#include<string>
#include<stdlib.h>
#include <utility>

class RPCDigitizer;
class RPCGeometry;
class RPCDetId;

class RPCDBSimSetUp
{
public:

  explicit RPCDBSimSetUp(const edm::ParameterSet& ps);
  virtual ~RPCDBSimSetUp();

  std::vector<float> getNoise(uint32_t id);
  std::vector<float> getEff(uint32_t id);
  float getTime(uint32_t id);
  std::map< int, std::vector<double> > getClsMap();

  /// sets geometry
  void setGeometry(const RPCGeometry * geom) {theGeometry = geom;}
  const RPCGeometry * getGeometry() { return theGeometry; }

  const RPCGeometry * theGeometry;

  std::map<uint32_t, std::vector<float> > _mapDetIdNoise;
  std::map<uint32_t, std::vector<float> > _mapDetIdEff;
  std::map<RPCDetId, float> _bxmap;
  std::map< int, std::vector<double> > _clsMap;

  std::ifstream *_infile1;
  std::ifstream *_infile2;
  std::ifstream *_infile3;
  std::ifstream *_infile4;

};

#endif
