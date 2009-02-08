#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibMuon/RPCCalibration/interface/RPCCalibSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include <cmath>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include<cstring>
#include<string>
#include<vector>
#include<stdlib.h>
#include <utility>
#include <map>

using namespace std;

RPCCalibSetUp::RPCCalibSetUp(const edm::ParameterSet& ps) {
  
  _mapDetIdNoise.clear();
  _mapDetIdEff.clear();
  _bxmap.clear();

    //------------------------ Noise Reading ----------------------------
    
  edm::FileInPath fp1 = ps.getParameter<edm::FileInPath>("noisemapfile");
  std::ifstream _infile1(fp1.fullPath().c_str(), std::ios::in);
  
  std::vector<float>  vnoise;

  int rpcdetid = 0;
  std::string buff;
  
  std::vector< std::string > words;

  int count = 0;
  while( getline(_infile1, buff, '\n') ){
    
    words.clear();
    vnoise.clear();
    
    stringstream ss;
    std::string chname;
    ss<<buff;
    ss>>chname>>rpcdetid;

    std::string::size_type pos = 0, prev_pos = 0;

    while ( (pos = buff.find("  ",pos)) != string::npos){
      
      words.push_back(buff.substr(prev_pos, pos - prev_pos));
      prev_pos = ++pos;
    }
    words.push_back(buff.substr(prev_pos, pos - prev_pos));
    
    for(unsigned int i = 2; i < words.size(); ++i){
      float value = atof( ((words)[i]).c_str() );
      vnoise.push_back(value);
    }
    
    _mapDetIdNoise.insert(make_pair(static_cast<uint32_t>(rpcdetid),vnoise));
    
    count++;
  }
   _infile1.close();

  //------------------------ Eff Reading ----------------------------
  
  edm::FileInPath fp2 = ps.getParameter<edm::FileInPath>("effmapfile");
  std::ifstream _infile2(fp2.fullPath().c_str(), std::ios::in);

  std::vector<float> veff ;
  rpcdetid = 0;
  
  while( getline(_infile2, buff, '\n') ){

    words.clear();
    veff.clear();
    
    stringstream ss;
    std::string chname;
    ss<<buff;
    ss>>chname>>rpcdetid;
    
    std::string::size_type pos = 0, prev_pos = 0;
    while ( (pos = buff.find("  ",pos)) != string::npos){
      
      words.push_back(buff.substr(prev_pos, pos - prev_pos));
      prev_pos = ++pos;
    }
    words.push_back(buff.substr(prev_pos, pos - prev_pos));
    
    for(unsigned int i = 2; i < words.size(); ++i){
      float value = atof(((words)[i]).c_str());
      veff.push_back(value);
    }
    _mapDetIdEff.insert(make_pair(static_cast<uint32_t>(rpcdetid),veff));
  }
  _infile2.close();

  //---------------------- Timing reading ------------------------------------

  edm::FileInPath fp3 = ps.getParameter<edm::FileInPath>("timingMap");
  std::ifstream _infile3(fp3.fullPath().c_str(), std::ios::in);

  uint32_t detUnit = 0;
  float timing = 0.;
  while(!_infile3.eof()){
    _infile3>>detUnit>>timing;
    _bxmap[RPCDetId(detUnit)] = timing;
  }
  _infile3.close();

  //---------------------- Cluster size --------------------------------------

  edm::FileInPath fp4 = ps.getParameter<edm::FileInPath>("clsmapfile");
  std::ifstream _infile4(fp4.fullPath().c_str(), ios::in);

  string buffer;
  double sum = 0;
  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<double> sum_clsize;

  while ( _infile4 >> buffer ) {
    const char *buffer1 = buffer.c_str();
    double dato = atof(buffer1);
    sum += dato;
    sum_clsize.push_back(sum);

    if(counter == row*20) {
      _clsMap[row] = sum_clsize;
      row++;
      sum = 0;
      sum_clsize.clear();
    }
    counter++;
  }
  _infile4.close();

}

std::vector<float> RPCCalibSetUp::getNoise(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdNoise.find(id);
  return (iter->second);
}

std::vector<float> RPCCalibSetUp::getEff(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdEff.find(id);
  return iter->second;
}

float RPCCalibSetUp::getTime(uint32_t id)
{
  RPCDetId rpcid(id);
  std::map<RPCDetId, float>::iterator iter = _bxmap.find(rpcid);
  return iter->second;
}

std::map< int, std::vector<double> > RPCCalibSetUp::getClsMap()
{
  return _clsMap;
}

RPCCalibSetUp::~RPCCalibSetUp(){}
