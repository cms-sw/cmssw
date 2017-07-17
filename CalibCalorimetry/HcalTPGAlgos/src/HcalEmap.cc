// -*- C++ -*-
//
// Package:     CalibCalorimetry/HcalTPGAlgos
// Class  :     HcalEmap
// 
// Implementation:
//     structure and functionality for HCAL electronic map
//     NOTE!
//     Keep xdaq and Oracle dependencies out of here!
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 14 14:30:20 CDT 2009
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcalEmap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

using namespace std;




int HcalEmap::read_map( std::string filename )
{
  int lines=0;

  std::string _row;
  ifstream inFile( filename . c_str(), std::ios::in );
  if (!inFile){
    std::cout << "Unable to open file with the electronic map: " << filename << std::endl;
  }
  else{
    std::cout << "File with the electronic map opened successfully: " << filename << std::endl;
  }
  while (getline( inFile, _row )) {
    HcalEmapRow aRow;
    char fpga[32];
    char subdet[32];
    
    int _read;
    const char * _format = "%d %d %d %s %d %d %d %d %s %d %d %d";
    _read = sscanf( _row . c_str(), _format,
		    &(aRow.rawId),
		    &(aRow.crate), &(aRow.slot),
		      fpga,
		    &(aRow.dcc),
		    &(aRow.spigot),&(aRow.fiber),&(aRow.fiberchan),
		    subdet,
		    &(aRow.ieta), &(aRow.iphi), &(aRow.idepth) );
    if ( _read >= 12 ){
      lines++;
      
      aRow . subdet .append( subdet );
      aRow . topbottom .append( fpga );
      
      map . push_back( aRow );
    }  
  }
  inFile.close();
  std::cout << "HcalEmap: " << lines << " lines read" << std::endl;

  return 0;
}
  


std::vector<HcalEmap::HcalEmapRow> & HcalEmap::get_map( void )
{
  return map;
}


bool HcalEmap::HcalEmapRow::operator<( const HcalEmap::HcalEmapRow & other) const{
  return rawId < other.rawId;
}



//
// _____ test procedures for the HcalEmap class _____________________________
//
int HcalEmap_test::test_read_map( std::string filename )
{
  HcalEmap map( filename );
  return 0;
}

