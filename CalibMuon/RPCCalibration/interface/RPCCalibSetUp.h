#ifndef RPCCalibSetUp_h
#define RPCCalibSetUp_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
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

class RPCCalibSetUp
{
public:

  explicit RPCCalibSetUp(const edm::ParameterSet& ps);
  virtual ~RPCCalibSetUp();

  std::vector<float> getNoise(uint32_t id);
  std::vector<float> getEff(uint32_t id);
  float getTime(uint32_t id);
  std::map< int, std::vector<double> > getClsMap();
  //  std::map<int, std::vector<double> > getClsMap(uint32_t id);
  std::vector<double> getCls(uint32_t id);

  /// sets geometry
  void setGeometry(const RPCGeometry * geom) {theGeometry = geom;}
  const RPCGeometry * getGeometry() { return theGeometry; }

  const RPCGeometry * theGeometry;

  std::map<uint32_t, std::vector<float> > _mapDetIdNoise;
  std::map<uint32_t, std::vector<float> > _mapDetIdEff;
  std::map<RPCDetId, float> _bxmap;
  std::map< int, std::vector<double> > _clsMap;
  std::map<uint32_t, std::vector<double> > _mapDetClsMap;

};

#endif
