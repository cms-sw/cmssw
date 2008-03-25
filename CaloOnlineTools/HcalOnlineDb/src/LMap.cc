// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     LMap
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
// $Id: LMap.cc,v 1.1 2008/02/12 17:02:01 kukartse Exp $
//

// system include files
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>


// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"

using namespace std;



LMap::LMap()
{
  //read( "HCALmapHBEF_10.31.2007.txt" );
}



LMap::~LMap()
{
}



int LMap::read( string map_file, string type )
{

  RooGKCounter lines;

  string _row;
  ifstream inFile( map_file . c_str(), ios::in );
  if (!inFile)
    {
      cout << "Unable to open file with the logical map: " << map_file << endl;
    }
  else
    {
      cout << "File with the logical map opened successfully: " << map_file << endl;
    }
  while ( getline( inFile, _row ) > 0 )
    {
      LMapRow aRow;
      char det[32];
      char rbx[32];
      char fpga[32];
      char slbin[32];
      char slbin2[32];
      char slnam[32];
      char rctnam[32];

      char * let_code = "Z";

      int _read;
      if ( type == "HBEF" )
	{
	  const char * _format = " %d %d %d %d %d %s %s %d %d %d %d %d %d %d %d %d %s %d %d %d %d %d %s %s %s %d %d %d %s %d";
	  _read = sscanf( _row . c_str(), _format,
			  &(aRow.side),
			  &(aRow.eta), &(aRow.phi), &(aRow.dphi), &(aRow.depth),
			  det,
			  rbx,
			  &(aRow.wedge), &(aRow.rm), &(aRow.pixel), &(aRow.qie), &(aRow.adc), &(aRow.rm_fi), &(aRow.fi_ch),
			  &(aRow.crate), &(aRow.htr),
			  fpga,
			  &(aRow.htr_fi),
			  &(aRow.dcc_sl), &(aRow.spigo), &(aRow.dcc), &(aRow.slb),
			  slbin, slbin2, slnam,
			  &(aRow.rctcra), &(aRow.rctcar), &(aRow.rctcon),
			  rctnam,
			  &(aRow.fedid) );
	}
      else if ( type == "HO" )
	{
	  const char * _format = " %d %d %d %d %d %s %s %d %d %d %d %d %d %d %s %d %d %s %d %d %d %d %d %s %s %s %d %d %d %s %d";
	  _read = sscanf( _row . c_str(), _format,
			  &(aRow.side),
			  &(aRow.eta), &(aRow.phi), &(aRow.dphi), &(aRow.depth),
			  det,
			  rbx,
			  &(aRow.wedge), &(aRow.rm), &(aRow.pixel), &(aRow.qie), &(aRow.adc), &(aRow.rm_fi), &(aRow.fi_ch),
			  &let_code,
			  &(aRow.crate), &(aRow.htr),
			  fpga,
			  &(aRow.htr_fi),
			  &(aRow.dcc_sl), &(aRow.spigo), &(aRow.dcc), &(aRow.slb),
			  slbin, slbin2, slnam,
			  &(aRow.rctcra), &(aRow.rctcar), &(aRow.rctcon),
			  rctnam,
			  &(aRow.fedid) );
	}
	  if ( _read >= 30 )
	    {
	      lines . count();
	      
	      aRow . det .append( det );
	      aRow . rbx .append( rbx );
	      aRow . fpga .append( fpga );
	      aRow . slbin .append( slbin );
	      aRow . slbin2 .append( slbin2 );
	      aRow . slnam .append( slnam );
	      aRow . rctnam .append( rctnam );
	      aRow . let_code .append( let_code );
	      
	      _table . push_back( aRow );
	      
	      //cout << aRow.side << "	" << aRow.eta << "	" << aRow.rctnam << endl;
	}
    }
  inFile.close();
  cout << "LMap: " << lines . getCount() << " lines read" << endl;

  return 0;
}



hcal::ConfigurationDatabase::LUTId LMap::getLUTId( LMapDetId _etaphi ){

  hcal::ConfigurationDatabase::LUTId result;

  return result;
}






