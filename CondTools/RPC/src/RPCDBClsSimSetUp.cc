#include "CondTools/RPC/interface/RPCDBClsSimSetUp.h"
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

RPCDBClsSimSetUp::RPCDBClsSimSetUp(const edm::ParameterSet& ps) {
  std::cout<<"RPCClusterSizePutDataFromFile::RPCClusterSizePutDataFromFile"<<std::endl;

  _mapDetClsMap.clear();

  edm::FileInPath fp1 =  ps.getParameter<edm::FileInPath>("clsidmapfile");
  std::ifstream inFile1(fp1.fullPath().c_str(), ios::in);
  
  std::vector<double> vClsDistrib ;
  int rpcdetid = 0;
  std::string buff;
  std::vector< std::string > words;

  while( getline(inFile1, buff, '\n') ){

    words.clear();
    vClsDistrib.clear();
    
    stringstream ss1;
    ss1<<buff;
    ss1>>rpcdetid;
    
    std::string::size_type pos = 0, prev_pos = 0;
    while ( (pos = buff.find("  ",pos)) != string::npos){
      
      words.push_back(buff.substr(prev_pos, pos - prev_pos));
      prev_pos = ++pos;
    }
    words.push_back(buff.substr(prev_pos, pos - prev_pos));
    
    //    std::cout<<"@@@ XXX rpcdetid\t"<<rpcdetid<<'\t'<<std::endl;;



    float clusterSizeSumData(0.);

    for(unsigned int i = 1; i < words.size(); ++i){
      float value = atof(((words)[i]).c_str());
      
      clusterSizeSumData+=value;
          vClsDistrib.push_back(clusterSizeSumData);
//       _mapDetClsMap.insert(make_pair(static_cast<uint32_t>(rpcdetid),clusterSizeSumData));
//       std::cout<<"_mapDetClsMap.size()\t"<<_mapDetClsMap.size()<<std::endl;
      //      std::cout<<i<<'\t'<<value<<'\t'<<clusterSizeSumData<<'\t'<<std::endl;
      if(!(i%20)){ 
        //      std::cout<<"H\t"<<std::endl; 
        clusterSizeSumData=0.;
      }
    }
    if(vClsDistrib.size()!=100){
      throw cms::Exception("DataCorrupt") 
        << "Exception comming from RPCCalibSetUp - cluster size - a wrong format "<< std::endl;
    }
    _mapDetClsMap.insert(make_pair(static_cast<uint32_t>(rpcdetid),vClsDistrib));
    std::cout<<"_mapDetClsMap.size()\t"<<_mapDetClsMap.size()<<std::endl;
    //     std::cout<<std::endl;
    //     std::cout<<"YYY"<<std::endl;

  }
}

// std::vector<float> RPCDBClsSimSetUp::getNoise(uint32_t id)
// {
//   map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdNoise.find(id);
//   return (iter->second);
// }

// std::vector<float> RPCDBClsSimSetUp::getEff(uint32_t id)
// {
//   map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdEff.find(id);
//   return iter->second;
// }

// float RPCDBClsSimSetUp::getTime(uint32_t id)
// {
//   RPCDetId rpcid(id);
//   std::map<RPCDetId, float>::iterator iter = _bxmap.find(rpcid);
//   return iter->second;
// }

// std::map< int, std::vector<double> > RPCDBClsSimSetUp::getClsMap()
// {
//   return _clsMap;
// }

RPCDBClsSimSetUp::~RPCDBClsSimSetUp(){
  delete _infile1;
  delete _infile2;
  delete _infile3;
  delete _infile4;
}

std::vector<double> RPCDBClsSimSetUp::getCls(uint32_t id){
  std::map<uint32_t,std::vector<double> >::iterator iter = _mapDetClsMap.find(id);
  if(iter == _mapDetClsMap.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCCalibSetUp - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 100){
    std::cout<<"(iter->second).size()\t"<<(iter->second).size()<<std::endl;
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCCalibSetUp - cluster size information in a wrong format for DetId\t"<<id<< std::endl;
  }
  return iter->second;

}
