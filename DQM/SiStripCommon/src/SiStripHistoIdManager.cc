// -*- C++ -*-
//
// Package:     SiStripCommon
// Class  :     SiStripHistoIdManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  dkcira
//         Created:  Sat Feb  4 16:21:12 CET 2006
// $Id$
//

// system include files
#include<boost/cstdint.hpp>
#include<iostream>
#include<sstream>

// user include files
#include "DQM/SiStripCommon/interface/SiStripHistoIdManager.h"

using namespace std;

SiStripHistoIdManager::SiStripHistoIdManager()
{
 separator="_#_";
}


SiStripHistoIdManager::~SiStripHistoIdManager()
{
}

std::string SiStripHistoIdManager::createHistoId(std::string description, uint32_t component_id){
  unsigned int pos = description.find( separator, 0 ); // check if string 'description' contains by mistake the 'separator'
  if ( pos == std::string::npos ){ // ok, not found
    ostringstream compid; compid<<component_id; // use ostringstream for casting integer to string
    string local_histo_id = description + separator + compid.str();
    return local_histo_id;
  }else{
    cout<<"SiStripHistoIdManager::createHistoId - histogram description cannot contain: "<<separator
        <<" histogram description = "<<description
        <<" . Throwing exception"<<endl;
    throw string("Exception thrown");
  }
}


uint32_t SiStripHistoIdManager::getComponentId(std::string histoid){
  unsigned int startpos = histoid.find( separator, 0 ); // find first position of separator
  uint32_t local_component_id = 0;
  if ( startpos == std::string::npos ){ // no separator found, cannot be histoid created from SiStripHistoIdManager
    cout<<"SiStripHistoIdManager::getComponentId - no separator "<<separator<<" found in: "<<histoid<<endl;
    cout<<"  This histogram id is not created from SiStripHistoIdManager and cannot be used by this class."<<endl;
    cout<<"  Returning a 0 component id."<<endl;
  }else{
    unsigned int length_separator = separator.size();
    string compid = histoid.substr(startpos+length_separator);     // substring from end of separator + 1 to string end
    std::istringstream input(compid); input >> local_component_id; // use istringstream for casting from string to uint32_t
  }
  return local_component_id;
}

