// -*- C++ -*-
// Package:    SiStripCommon
// Class:      SiStripDetInfoFileReader
// Original Author:  G. Bruno
//         Created:  Mon May 20 10:04:31 CET 2007
// $Id: SiStripDetInfoFileReader.cc,v 1.6 2009/06/09 18:31:52 elmer Exp $

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

using namespace cms;
using namespace std;

SiStripDetInfoFileReader& SiStripDetInfoFileReader::operator=(const SiStripDetInfoFileReader &copy) {
  detData_=copy.detData_;
  detIds_=copy.detIds_;
  return *this;  
}

SiStripDetInfoFileReader::SiStripDetInfoFileReader(const edm::ParameterSet& pset, const edm::ActivityRegistry& ar){
  edm::FileInPath fp(pset.getUntrackedParameter<std::string>("filePath","CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
  reader(fp.fullPath());
}

SiStripDetInfoFileReader::SiStripDetInfoFileReader(const SiStripDetInfoFileReader& copy) {
  detData_=copy.detData_;
  detIds_=copy.detIds_;
}

SiStripDetInfoFileReader::SiStripDetInfoFileReader(std::string filePath) {
  reader(filePath);
}

void SiStripDetInfoFileReader::reader(std::string filePath) {

//   if(filePath==std::string("")){
//     filePath = edm::FileInPath(std::string("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat") ).fullPath();
//   }

  edm::LogInfo("SiStripDetInfoFileReader") << "filePath " << filePath << std::endl;


  detData_.clear();
  detIds_.clear();

  inputFile_.open(filePath.c_str());

  if (inputFile_.is_open()){

    for(;;) {

      uint32_t detid; 
      double stripLength;
      unsigned short numberOfAPVs;
      float thickness;

      inputFile_ >> detid >> numberOfAPVs  >> stripLength >> thickness;

      if (!(inputFile_.eof() || inputFile_.fail())){

	detIds_.push_back(detid);

	//	inputFile_ >> numberOfAPVs;
	//	inputFile_ >> stripLength;

	//     	edm::LogInfo("SiStripDetInfoFileReader::SiStripDetInfoFileReader") << detid <<" " <<numberOfAPVs <<" " <<stripLength << " "<< thickness<< endl;

	std::map<uint32_t, DetInfo >::const_iterator it = detData_.find(detid);

	if(it==detData_.end()){
	  detData_[detid]=DetInfo(numberOfAPVs, stripLength,thickness);
	}
	else{

	  edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader") <<"DetId " << detid << " already found on file. Ignoring new data"<<endl;
	
	  detIds_.pop_back();
	  continue;
	}
      }
      else if (inputFile_.eof()){

	edm::LogInfo("SiStripDetInfoFileReader::SiStripDetInfoFileReader - END of file reached")<<endl;
	break;

      }
      else if (inputFile_.fail()) {
      
	edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader - ERROR while reading file")<<endl;     
	break;
      }
    }

    inputFile_.close();

  }  
  else {

    edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader - Unable to open file")<<endl;
    return;
  
  }

//   int i=0;
//   for(std::map<uint32_t, std::pair<unsigned short, double> >::iterator it =detData_.begin(); it!=detData_.end(); it++ ) {
//     std::cout<< it->first << " " << (it->second).first << " " << (it->second).second<<endl;
//     i++;
//   }
//   std::cout<<i;


}


SiStripDetInfoFileReader::~SiStripDetInfoFileReader(){
}


const std::pair<unsigned short, double>  SiStripDetInfoFileReader::getNumberOfApvsAndStripLength(uint32_t detId) const{

  std::map<uint32_t, DetInfo >::const_iterator it = detData_.find(detId);

  if(it!=detData_.end()){

    return std::pair<unsigned short, double>(it->second.nApvs,it->second.stripLength); 

  }
  else{

    static std::pair<unsigned short, double> defaultValue(0,0);
    edm::LogWarning("SiStripDetInfoFileReader::getNumberOfApvsAndStripLength - Unable to find requested detid. Returning invalid data ")<<endl; 
    return defaultValue;

  }

}

const float & SiStripDetInfoFileReader::getThickness(uint32_t detId) const{

  std::map<uint32_t, DetInfo >::const_iterator it = detData_.find(detId);

  if(it!=detData_.end()){

    return it->second.thickness; 

  }
  else{

    static float defaultValue=0;
    edm::LogWarning("SiStripDetInfoFileReader::getThickness - Unable to find requested detid. Returning invalid data ")<<endl; 
    return defaultValue;

  }

}

