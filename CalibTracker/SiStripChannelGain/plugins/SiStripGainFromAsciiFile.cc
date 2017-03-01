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


  std::stringstream ss;
  ss.str("");
  ss << "[SiStripGainFromAsciiFile::getNewObject]\n Reading Ascii File\n";
  FILE* infile = fopen (Asciifilename_.c_str(), "r");
  char line[4096];
  if (infile){
    while(fgets(line, 4096, infile)!=NULL){
       uint32_t detid;
       ModuleGain MG;
       MG.apv[0] = 0.0;  MG.apv[1] = 0.0; MG.apv[2] = 0.0; MG.apv[3] = 0.0; MG.apv[4] = 0.0; MG.apv[5] = 0.0;
       char* pch=strtok(line," "); int Arg=0;
       while (pch!=NULL){
            if(Arg==0){
               sscanf(pch, "%d", &detid);
            }else if(Arg<=6){
               sscanf(pch, "%f", &(MG.apv[Arg-1]));
            }else{
               //nothing to do here
            }       
            pch=strtok(NULL," ");Arg++;
      }
      ss << detid << " " <<  MG.apv[0] << " " <<  MG.apv[1] << " " <<  MG.apv[2] << " " <<  MG.apv[3] << " " <<  MG.apv[4] << " " <<  MG.apv[5] << std::endl;
      GainsMap.insert(std::pair<unsigned int,ModuleGain>(detid,MG));
    }
    fclose(infile);
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
    ModuleGain MG;
    if (DetId(*it).det()!=DetId::Tracker)
      continue;

    nApvPair=reader.getNumberOfApvsAndStripLength(*it).first/2;
    
    ss << "Looking at detid " << *it << " nApvPairs  " << nApvPair << std::endl;
    __gnu_cxx::hash_map< unsigned int,ModuleGain>::const_iterator iter=GainsMap.find(*it);    
    if (iter!=GainsMap.end()){
      MG = iter->second;
      ss << " " <<  MG.apv[0] << " " <<  MG.apv[1] << " " <<  MG.apv[2] <<  " " <<  MG.apv[3] << " " <<  MG.apv[4] << " " <<  MG.apv[5] << std::endl;
    }else {
      ss << "Hard reset for detid " << *it << std::endl;
      MG.hard_reset(referenceValue_);
    }
    
    std::vector<float> DetGainsVector;
    
    if (nApvPair==2){
      DetGainsVector.push_back(MG.apv[0]/referenceValue_);  
      DetGainsVector.push_back(MG.apv[1]/referenceValue_);  
      DetGainsVector.push_back(MG.apv[2]/referenceValue_);        
      DetGainsVector.push_back(MG.apv[3]/referenceValue_);  
    } else if (nApvPair==3){   		   		   		  
      DetGainsVector.push_back(MG.apv[0]/referenceValue_);  
      DetGainsVector.push_back(MG.apv[1]/referenceValue_);  
      DetGainsVector.push_back(MG.apv[2]/referenceValue_);  
      DetGainsVector.push_back(MG.apv[3]/referenceValue_);  
      DetGainsVector.push_back(MG.apv[4]/referenceValue_);  
      DetGainsVector.push_back(MG.apv[5]/referenceValue_);  
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




