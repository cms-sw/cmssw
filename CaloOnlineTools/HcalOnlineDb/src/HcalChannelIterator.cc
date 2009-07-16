// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     HcalChannelIterator
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev
//         Created:  Mon Jul 13 12:15:33 CEST 2009
// $Id$
//

#include <fstream>
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelIterator.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

HcalChannelIterator::HcalChannelIterator()
{
}


HcalChannelIterator::~HcalChannelIterator()
{
}



int HcalChannelIterator::size(void){
  return channel_list.size();
}


int HcalChannelIterator::clearChannelList(void){
  channel_list.clear();
  return 0;
}


int HcalChannelIterator::addListFromLmapAscii(std::string filename){
  RooGKCounter lines;
  int _current_size = size();
  string _row;
  ifstream inFile( filename . c_str(), ios::in );
  if (!inFile){
    cout << "Unable to open file with the electronic map: " << filename << endl;
  }
  else{
    cout << "File with the logical map opened successfully: " << filename << endl;
  }
  while ( getline( inFile, _row ) > 0 ){
    //#   side    eta    phi   dphi  depth    det
    int _num, _side, _eta, _phi, _dphi, _depth;
    char subdetbuf[32];
    
    int _read;
    const char * _format = "%d %d %d %d %d %d %s";
    _read = sscanf( _row . c_str(), _format,
		    &_num, &_side, &_eta, &_phi, &_dphi, &_depth,
		    subdetbuf );
    if ( _read == 7 ){
      lines . count();
      
      std::string subdet(subdetbuf);

      HcalSubdetector _det;
      if ( subdet.find("HB")!=string::npos ) _det = HcalBarrel;
      else if ( subdet.find("HE")!=string::npos ) _det = HcalEndcap;
      else if ( subdet.find("HO")!=string::npos ) _det = HcalOuter;
      else if ( subdet.find("HF")!=string::npos ) _det = HcalForward;
      else _det = HcalOther;

      HcalDetId _detid(_det, _side*_eta, _phi, _depth);
      
      if (_det == HcalBarrel || _det == HcalEndcap || _det == HcalOuter || _det == HcalForward){
	channel_list . push_back( _detid );
      }
    }  
  }
  inFile.close();
  cout << "Logical map file: " << lines . getCount() << " lines read" << endl;
  cout << "Logical map file: " << size() - _current_size << " lines added to the list" << endl;
  //
  return 0;
}


int HcalChannelIterator::begin(void){
  const_iterator = channel_list.begin();
  return 0;
}


int HcalChannelIterator::next(void){
  const_iterator++;
}


bool HcalChannelIterator::end(void){
  if (const_iterator==channel_list.end()){
    return true;
  }
  else{
    return false;
  }
}


HcalGenericDetId HcalChannelIterator::getHcalGenericDetId(void){
  if (const_iterator!=channel_list.end()){
    return *const_iterator;
  }
  else{
    return 0;
  }
}


HcalSubdetector HcalChannelIterator::getHcalSubdetector(void){
  if (const_iterator!=channel_list.end()){
    HcalDetId _id(*const_iterator);
    return _id.subdet();
  }
  else return HcalOther;
}
 

int HcalChannelIterator::getIeta(void){
  if (const_iterator!=channel_list.end()){
    HcalDetId _id(*const_iterator);
    return _id.ieta();
  }
  else return -1000;
}


int HcalChannelIterator::getIphi(void){
  if (const_iterator!=channel_list.end()){
    HcalDetId _id(*const_iterator);
    return _id.iphi();
  }
  else return -1000;
}


int HcalChannelIterator::getDepth(void){
  if (const_iterator!=channel_list.end()){
    HcalDetId _id(*const_iterator);
    return _id.depth();
  }
  else return -1000;
}


int HcalChannelIterator::initHBEFListFromLmapAscii(void){
  clearChannelList();
  addListFromLmapAscii("HCALmapHBEF_Jan.27.2009.txt");
  addListFromLmapAscii("HCALmapHO_Jan.27.2009.txt");
}
