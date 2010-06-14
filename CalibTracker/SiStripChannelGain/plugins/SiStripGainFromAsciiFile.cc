#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainFromAsciiFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "DataFormats/DetId/interface/DetId.h"


#include <iostream>
#include <fstream>
#include <sstream>


SiStripGainFromAsciiFile::SiStripGainFromAsciiFile(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>(iConfig){


  Asciifilename_=iConfig.getParameter<std::string>("InputFileName");
  referenceValue_ = iConfig.getParameter<double>("referenceValue");
  fp_ = iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
}               


SiStripGainFromAsciiFile::~SiStripGainFromAsciiFile(){
  edm::LogInfo("SiStripGainFromAsciiFile::~SiStripGainFromAsciiFile");
}

SiStripApvGain * SiStripGainFromAsciiFile::getNewObject(){

  SiStripApvGain* obj = new SiStripApvGain();

  uint32_t detid;
  FibersGain FG;

  std::stringstream ss;
  ss.str("");
  ss << "[SiStripGainFromAsciiFile::getNewObject]\n Reading Ascii File\n";
  std::ifstream infile;
  infile.open (Asciifilename_.c_str());
  if (infile.is_open()){
    while (infile.good()){
      infile >> detid >> FG.fiber[0] >> FG.fiber[1] >> FG.fiber[2];
      ss << detid << " " <<  FG.fiber[0] << " " <<  FG.fiber[1] << " " <<  FG.fiber[2] << std::endl;
      GainsMap.insert(std::pair<unsigned int,FibersGain>(detid,FG));
    }
    infile.close();
    edm::LogInfo("SiStripGainFromAsciiFile") << ss.str();
  } else  {
    edm::LogError("SiStripGainFromAsciiFile")<< " [SiStripGainFromAsciiFile::getNewObject] Error opening file " << Asciifilename_ << std::endl;
    assert(0);
  }
  
  

  SiStripDetInfoFileReader reader(fp_.fullPath());
  
  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  
  ss.str("");
  ss << "[SiStripGainFromAsciiFile::getNewObject]\n Filling SiStripApvGain object";
  short nApvPair;
  for(std::vector<uint32_t>::const_iterator it=DetIds.begin(); it!=DetIds.end(); it++){

    if (DetId(*it).det()!=DetId::Tracker)
      continue;

    nApvPair=reader.getNumberOfApvsAndStripLength(*it).first/2;
    
    ss << "Looking at detid " << *it << " nApvPairs  " << nApvPair << std::endl;

    __gnu_cxx::hash_map< unsigned int,FibersGain>::const_iterator iter=GainsMap.find(*it);
    
    if (iter!=GainsMap.end()){
      FG = iter->second;
      ss << " " <<  FG.fiber[0] << " " <<  FG.fiber[1] << " " <<  FG.fiber[2] << std::endl;
    }
    else {
      ss << "Hard reset for detid " << *it << std::endl;
      FG.hard_reset(referenceValue_);
    }
    
    std::vector<float> DetGainsVector;
    
    if (nApvPair==2){
      DetGainsVector.push_back(FG.fiber[0]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[0]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[2]/referenceValue_);        
      DetGainsVector.push_back(FG.fiber[2]/referenceValue_);  
    } else if (nApvPair==3){   		   		   		  
      DetGainsVector.push_back(FG.fiber[0]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[0]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[1]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[1]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[2]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[2]/referenceValue_);  
    } else {
      edm::LogError("SiStripGainFromAsciiFile") << " SiStripGainFromAsciiFile::getNewObject] ERROR for detid " << *it << " not expected number of APV pairs " << nApvPair <<std::endl;
    }
    
    SiStripApvGain::Range range(DetGainsVector.begin(),DetGainsVector.end());
    if ( ! obj->put(*it,range) ){
      edm::LogError("SiStripGainFromAsciiFile")<<" [SiStripGainFromAsciiFile::getNewObject] detid already exists"<<std::endl;
      ss <<" [SiStripGainFromAsciiFile::getNewObject] detid already exists"<<std::endl;
    }
  }
  edm::LogInfo("SiStripGainFromAsciiFile") << ss.str();

  return obj;
}




