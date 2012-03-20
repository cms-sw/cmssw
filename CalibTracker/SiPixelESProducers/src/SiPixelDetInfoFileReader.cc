// -*- C++ -*-
// Package:    SiPixelESProducers
// Class:      SiPixelDetInfoFileReader
// Original Author:  V.Chiochia
//         Created:  Mon May 20 10:04:31 CET 2007
// $Id: SiPixelDetInfoFileReader.cc,v 1.1.2.1 2010/06/25 16:41:54 hcheung Exp $

#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//#include "FWCore/ParameterSet/interface/FileInPath.h"

using namespace cms;
using namespace std;


SiPixelDetInfoFileReader::SiPixelDetInfoFileReader(std::string filePath, bool readROCInfo) {

//   if(filePath==std::string("")){
//     filePath = edm::FileInPath(std::string("CalibTracker/SiPixelCommon/data/SiPixelDetInfo.dat") ).fullPath();
//   }

  detData_.clear();
  detIds_.clear();
  theROCInfo_.clear();
  inputFile_.open(filePath.c_str());

  if (inputFile_.is_open()){

    for(;;) {

      uint32_t detid;
      int ncols;
      int nrows;
      int rocrows;

      inputFile_ >> detid >> ncols  >> nrows ;
      if (readROCInfo) { inputFile_ >> rocrows;}

      if (!(inputFile_.eof() || inputFile_.fail())){

	detIds_.push_back(detid);

	//inputFile_ >> numberOfAPVs;
	//inputFile_ >> stripLength;
	
	//       edm::LogInfo("SiPixelDetInfoFileReader::SiPixelDetInfoFileReader") << detid <<" " <<numberOfAPVs <<" " <<stripLength << " "<< thickness<< endl;
	
	std::map<uint32_t, std::pair<int, int> >::const_iterator it = detData_.find(detid);
	
	if( it==detData_.end() ){
	  
	  detData_[detid]=pair<int, int>(ncols,nrows);
	  if (readROCInfo) { theROCInfo_[detid]=rocrows; }
	  else {theROCInfo_[detid]=80;} //Move hard-wired constant to here...
	}
	else{	  
	  edm::LogError("SiPixelDetInfoFileReader::SiPixelDetInfoFileReader") <<"DetId " << detid << " already found on file. Ignoring new data"<<endl;
	  detIds_.pop_back();
	  continue;
	}
      }
      else if (inputFile_.eof()){
	
	edm::LogInfo("SiPixelDetInfoFileReader::SiPixelDetInfoFileReader - END of file reached")<<endl;
	break;
	
      }
      else if (inputFile_.fail()) {
	
	edm::LogError("SiPixelDetInfoFileReader::SiPixelDetInfoFileReader - ERROR while reading file")<<endl;     
	break;
      }
    }
    
    inputFile_.close();
    
  }  
  else {
    
    edm::LogError("SiPixelDetInfoFileReader::SiPixelDetInfoFileReader - Unable to open file")<<endl;
    return;
    
  }
  

//   int i=0;
//   for(std::map<uint32_t, std::pair<unsigned short, double> >::iterator it =detData_.begin(); it!=detData_.end(); it++ ) {
//     std::cout<< it->first << " " << (it->second).first << " " << (it->second).second<<endl;
//     i++;
//   }
//   std::cout<<i;


}
//
// Destructor
//
SiPixelDetInfoFileReader::~SiPixelDetInfoFileReader(){

   edm::LogInfo("SiPixelDetInfoFileReader::~SiPixelDetInfoFileReader");
}
//
// get DetId's
//
const std::vector<uint32_t> & SiPixelDetInfoFileReader::getAllDetIds() const{

  return detIds_;

}
//
// get method
//
const std::pair<int, int> & SiPixelDetInfoFileReader::getDetUnitDimensions(uint32_t detId) const{

  std::map<uint32_t, std::pair<int, int> >::const_iterator it = detData_.find(detId);

  if(it!=detData_.end()){
    
    return (*it).second; 

  }
  else{

    static std::pair< int, int> defaultValue(0,0);
    edm::LogWarning("SiPixelDetInfoFileReader::getDetUnitDimensions - Unable to find requested detid. Returning invalid data ")<<endl; 
    return defaultValue;

  }

}

const int & SiPixelDetInfoFileReader::getROCRows(uint32_t detId) const{

  std::map<uint32_t, int >::const_iterator it = theROCInfo_.find(detId);

  if(it!=theROCInfo_.end()){
    
    return (*it).second; 

  }
  else{

    static int defaultValue(80);
    edm::LogWarning("SiPixelDetInfoFileReader::getROCRows - Unable to find requested detid. Returning default data ")<<endl; 
    return defaultValue;

  }

}

