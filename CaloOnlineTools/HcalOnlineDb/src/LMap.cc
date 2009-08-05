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
// $Id: LMap.cc,v 1.6 2009/08/04 22:25:18 kukartse Exp $
//

// system include files
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalAssistant.h"

using namespace std;



class LMap::impl {
public:
  impl(){ }
  ~impl(){ }

  int read( string accessor, string type );
  std::map<int,LMapRow> & get_map( void ){ return _lmap; };
  
private:
  std::vector<LMapRow> _table;
  std::map<int,LMapRow> _lmap;

};



LMap::LMap() : p_impl( new impl ) { }

LMap::~LMap() { }



int LMap::read( string accessor, string type )
{
  return p_impl -> read( accessor, type );
}

std::map<int,LMapRow> & LMap::get_map( void )
{
  return p_impl -> get_map();
}

int LMap::impl::read( string map_file, string type )
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
      cout << "Type: " << type << endl;
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
	  const char * _format = " %d %d %d %d %d %s %s %d %d %d %d %d %d %d %s %d %d %s %d %d %d %d %d";
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
			  &(aRow.dcc_sl), &(aRow.spigo), &(aRow.dcc), &(aRow.slb) );
	  //slbin, slbin2, slnam,
	  //&(aRow.rctcra), &(aRow.rctcar), &(aRow.rctcon),
	  //rctnam,
	  //&(aRow.fedid) );
	}
      if ( _read >= 23 )
	{
	  lines . count();
	  
	  string _det(det);
	  if      ( _det.find("HB") != string::npos ) aRow . det = HcalBarrel;
	  else if ( _det.find("HE") != string::npos ) aRow . det = HcalEndcap;
	  else if ( _det.find("HF") != string::npos ) aRow . det = HcalForward;
	  else if ( _det.find("HO") != string::npos ) aRow . det = HcalOuter;
	  else                    aRow . det = HcalOther;

	  aRow . rbx .append( rbx );
	  aRow . fpga .append( fpga );
	  aRow . slbin .append( slbin );
	  aRow . slbin2 .append( slbin2 );
	  aRow . slnam .append( slnam );
	  aRow . rctnam .append( rctnam );
	  aRow . let_code .append( let_code );
	  
	  _table . push_back( aRow );

	  HcalDetId _hdid(aRow.det, aRow.side*aRow.eta, aRow.phi, aRow.depth);

	  _lmap[_hdid.rawId()] = aRow;

	}
    }
  inFile.close();
  cout << "LMap: " << lines . getCount() << " lines read" << endl;

  return 0;
}

//_______________________________________________________________________
//
//_____ EMAP stuff
//
//_______________________________________________________________________

EMap::EMap( const HcalElectronicsMap * emap ){
  if (emap){
    HcalAssistant _ass;
    //
    //_____ precision channels __________________________________________
    //
    std::vector <HcalElectronicsId> v_eId = emap->allElectronicsIdPrecision();
    for (std::vector <HcalElectronicsId>::const_iterator eId=v_eId.begin();
	 eId!=v_eId.end();
	 eId++){
      EMapRow row;
      row.rawId     = eId->rawId();
      row.crate     = eId->readoutVMECrateId();
      row.slot      = eId->htrSlot();
      row.dcc       = eId->dccid();
      row.spigot    = eId->spigot();
      row.fiber     = eId->fiberIndex();
      row.fiberchan = eId->fiberChanId();
      if (eId->htrTopBottom()==1) row.topbottom = "t";
      else row.topbottom = "b";
      //
      HcalGenericDetId _gid( emap->lookup(*eId) );
      if ( !(_gid.null()) &&
	   (_gid.genericSubdet()==HcalGenericDetId::HcalGenBarrel ||
	    _gid.genericSubdet()==HcalGenericDetId::HcalGenEndcap ||
	    _gid.genericSubdet()==HcalGenericDetId::HcalGenForward ||
	    _gid.genericSubdet()==HcalGenericDetId::HcalGenOuter
	    )
	   ){
	HcalDetId _id( emap->lookup(*eId) );
	row.ieta      = _id.ieta();
	row.iphi      = _id.iphi();
	row.idepth    = _id.depth();
	row.subdet    = _ass.getSubdetectorString(_id.subdet());
	// fill the map
	map.push_back(row);
      }
      else if ( !(_gid.null()) &&
	   _gid.genericSubdet()==HcalGenericDetId::HcalGenZDC
	   ){
	HcalZDCDetId _id( emap->lookup(*eId) );
	row.zdc_channel      = _id.channel();
	row.zdc_section      = _ass.getZDCSectionString(_id.section());
	row.idepth           = _id.depth();
	row.zdc_zside        = _id.zside();
	// fill the map
	map.push_back(row);
      }
    }
    //
    //_____ trigger channels __________________________________________
    //
    v_eId = emap->allElectronicsIdTrigger();
    for (std::vector <HcalElectronicsId>::const_iterator eId=v_eId.begin();
	 eId!=v_eId.end();
	 eId++){
      EMapRow row;
      row.rawId     = eId->rawId();
      row.crate     = eId->readoutVMECrateId();
      row.slot      = eId->htrSlot();
      row.dcc       = eId->dccid();
      row.spigot    = eId->spigot();
      if (eId->htrTopBottom()==1) row.topbottom = "t";
      else row.topbottom = "b";
      //
      HcalTrigTowerDetId _id( emap->lookupTrigger(*eId) );
      if ( !(_id.null()) ){
	row.ieta      = _id.ieta();
	row.iphi      = _id.iphi();
	row.idepth    = _id.depth();
	row.subdet    = _ass.getSubdetectorString(_id.subdet());
	// fill the map
	map.push_back(row);
      }
    }
  }
  else{
    cerr << "Pointer to HcalElectronicsMap is 0!!!" << endl;
  }
}


int EMap::read_map( std::string filename )
{
  RooGKCounter lines;

  string _row;
  ifstream inFile( filename . c_str(), ios::in );
  if (!inFile){
    cout << "Unable to open file with the electronic map: " << filename << endl;
  }
  else{
    cout << "File with the electronic map opened successfully: " << filename << endl;
  }
  while ( getline( inFile, _row ) > 0 ){
    EMapRow aRow;
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
      lines . count();
      
      aRow . subdet .append( subdet );
      aRow . topbottom .append( fpga );
      
      map . push_back( aRow );
      //cout << "DEBUG: " << _row << endl;
      //cout << "DEBUG: " << aRow.ieta << endl;
    }  
  }
  inFile.close();
  cout << "EMap: " << lines . getCount() << " lines read" << endl;

  return 0;
}
  


std::vector<EMap::EMapRow> & EMap::get_map( void )
{
  return map;
}


bool EMap::EMapRow::operator<( const EMap::EMapRow & other) const{
  return rawId < other.rawId;
}




// ===> test procedures for the EMap class
int EMap_test::test_read_map( std::string filename )
{
  EMap map( filename );
  return 0;
}

// ===> test procedures for the LMap class
LMap_test::LMap_test() :_lmap(new LMap){ }

int LMap_test::test_read(string accessor, string type)
{
  _lmap -> read(accessor,type);
  return 0;
}
