
//
// Gena Kukartsev (Brown), Feb 1, 2008
// $Id:
//

#include <iostream>
#include <string.h>

#include "xgi/Utils.h"
#include "toolbox/string.h"

#include "occi.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplOracle.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplXMLFile.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

using namespace std;
using namespace oracle::occi;
using namespace hcal;

HCALConfigDB::HCALConfigDB( void )
{    
  database = NULL;
  database2 = NULL;
}

HCALConfigDB::HCALConfigDB( string _accessor )
{    
  database = NULL;
  database2 = NULL;
  accessor = _accessor;
}


HCALConfigDB::~HCALConfigDB( void )
{    
  if ( database != NULL ) delete database;
  if ( database2 != NULL ) delete database2;
}


void HCALConfigDB::setAccessor( string _accessor )
{      
  accessor = _accessor;
}

void HCALConfigDB::connect( string _accessor )
{

  accessor = _accessor;

  std::string::size_type i = accessor . find( "occi://" );
  if ( i!=std::string::npos )
    {
      database = new ConfigurationDatabaseImplOracle();
      database -> connect( accessor );
    }
  else
    {
      database = new ConfigurationDatabaseImplXMLFile();
      database -> connect( accessor );
    }
}



void HCALConfigDB::connect( string _accessor1, string _accessor2 )
{

  connect (_accessor1 );

  accessor2 = _accessor2;

  std::string::size_type i = accessor2 . find( "occi://" );
  if ( i!=std::string::npos )
    {
      database2 = new ConfigurationDatabaseImplOracle();
      database2 -> connect( accessor2 );
    }
  else
    {
      database2 = new ConfigurationDatabaseImplXMLFile();
      database2 -> connect( accessor2 );
    }
}




void HCALConfigDB::disconnect( void )
{
  if ( database != NULL ) database -> disconnect();
  if ( database2 != NULL ) database2 -> disconnect();
}



std::vector<unsigned int> HCALConfigDB::getOnlineLUT( string tag, int crate, int slot, int topbottom, int fiber, int channel, int luttype )
{

  //connect( accessor );

  std::vector<unsigned int> result;

  hcal::ConfigurationDatabase::FPGASelection _fpga;
  if ( topbottom == 0 ) _fpga = hcal::ConfigurationDatabase::Bottom;
  else if ( topbottom == 1 ) _fpga = hcal::ConfigurationDatabase::Top;
  else
    {
      cout << "topbottom out of range" << endl;
      exit(-1);
    }

  hcal::ConfigurationDatabase::LUTType _lt;
  if ( luttype == 1 ) _lt = hcal::ConfigurationDatabase::LinearizerLUT;
  else if ( luttype == 2 ) _lt = hcal::ConfigurationDatabase::CompressionLUT;
  else
    {
      cout << "LUT type out of range" << endl;
      exit(-1);
    }

  hcal::ConfigurationDatabase::LUTId _lutid( crate, slot, _fpga, fiber, channel, _lt );
  std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT> testLUTs;

  XMLProcessor * theProcessor = XMLProcessor::getInstance();

  try {
    database -> getLUTs(tag, crate, slot, testLUTs);
  } catch (hcal::exception::ConfigurationItemNotFoundException& e) {
    cout << "Found nothing!" << endl;
  } catch (hcal::exception::Exception& e2) {
    cout << "Exception: " << e2.what() << endl;
  }

  result = testLUTs[_lutid];

  //database -> disconnect();

  return result;
}

std::vector<unsigned int> HCALConfigDB::getOnlineLUT( string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt )
{
  HcalDetId _id( _rawid );

  double _condition_data_set_id;
  unsigned int _crate, _slot, _fiber, _channel;
  hcal::ConfigurationDatabase::FPGASelection _fpga;

  int side   = _id . zside();
  int etaAbs = _id . ietaAbs();
  int phi    = _id . iphi();
  int depth  = _id . depth();
  string subdetector;
  if ( _id . subdet() == HcalBarrel) subdetector = "HB";
  else if ( _id . subdet() == HcalEndcap) subdetector = "HE";
  else if ( _id . subdet() == HcalOuter) subdetector = "HO";
  else if ( _id . subdet() == HcalForward) subdetector = "HF";

  oracle::occi::Connection * _connection = database -> getConnection();

  try {
    Statement* stmt = _connection -> createStatement();
    std::string query = ("SELECT RECORD_ID, CRATE, HTR_SLOT, HTR_FPGA, HTR_FIBER, FIBER_CHANNEL ");
    query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V3 ";
    query += toolbox::toString(" WHERE SIDE=%d AND ETA=%d AND PHI=%d AND DEPTH=%d AND SUBDETECTOR='%s'", side, etaAbs, phi, depth, subdetector . c_str() );
    
    //SELECT
    ResultSet *rs = stmt->executeQuery(query.c_str());

    _condition_data_set_id = 0.0;

    while (rs->next()) {
      double _cdsi = rs -> getDouble(1);
      if ( _condition_data_set_id < _cdsi )
	{
	  _condition_data_set_id = _cdsi;
	  _crate    = rs -> getInt(2);
	  _slot     = rs -> getInt(3);
	  std::string fpga_ = rs -> getString(4);
	  if ( fpga_ == "top" ) _fpga = hcal::ConfigurationDatabase::Top;
	  else _fpga  = hcal::ConfigurationDatabase::Bottom;
	  _fiber    = rs -> getInt(5);
	  _channel  = rs -> getInt(6);
	  
	  //cout << _cdsi << "   " << _crate << "   " << _slot << "   " << _fiber << "   " << _channel << endl;
	}
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }

  int topbottom, luttype;
  if ( _fpga == hcal::ConfigurationDatabase::Top ) topbottom = 1;
  else topbottom = 0;
  if ( _lt == hcal::ConfigurationDatabase::LinearizerLUT ) luttype = 1;
  else luttype = 2;
  
  std::vector<unsigned int> result = getOnlineLUT( tag, _crate, _slot, topbottom, _fiber, _channel, luttype );
  
  return result;
}


std::vector<unsigned int> HCALConfigDB::getOnlineLUTFromXML( string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt ){

  std::vector<unsigned int> result;

  if ( database && database2 ){
    
    HcalDetId _id( _rawid );
    
    double _condition_data_set_id;
    unsigned int _crate, _slot, _fiber, _channel;
    hcal::ConfigurationDatabase::FPGASelection _fpga;
    
    int side   = _id . zside();
    int etaAbs = _id . ietaAbs();
    int phi    = _id . iphi();
    int depth  = _id . depth();
    string subdetector;
    if ( _id . subdet() == HcalBarrel) subdetector = "HB";
    else if ( _id . subdet() == HcalEndcap) subdetector = "HE";
    else if ( _id . subdet() == HcalOuter) subdetector = "HO";
    else if ( _id . subdet() == HcalForward) subdetector = "HF";
    
    oracle::occi::Connection * _connection = database2 -> getConnection();
    
    try {
      Statement* stmt = _connection -> createStatement();
      std::string query = ("SELECT RECORD_ID, CRATE, HTR_SLOT, HTR_FPGA, HTR_FIBER, FIBER_CHANNEL ");
      query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V3 ";
      query += toolbox::toString(" WHERE SIDE=%d AND ETA=%d AND PHI=%d AND DEPTH=%d AND SUBDETECTOR='%s'", side, etaAbs, phi, depth, subdetector . c_str() );
      
      //SELECT
      ResultSet *rs = stmt->executeQuery(query.c_str());
      
      _condition_data_set_id = 0.0;
      
      while (rs->next()) {
	double _cdsi = rs -> getDouble(1);
	if ( _condition_data_set_id < _cdsi )
	  {
	    _condition_data_set_id = _cdsi;
	    _crate    = rs -> getInt(2);
	    _slot     = rs -> getInt(3);
	    std::string fpga_ = rs -> getString(4);
	    if ( fpga_ == "top" ) _fpga = hcal::ConfigurationDatabase::Top;
	    else _fpga  = hcal::ConfigurationDatabase::Bottom;
	    _fiber    = rs -> getInt(5);
	    _channel  = rs -> getInt(6);
	    
	    //cout << _cdsi << "   " << _crate << "   " << _slot << "   " << _fiber << "   " << _channel << endl;
	  }
      }
      //Always terminate statement
      _connection -> terminateStatement(stmt);
    } catch (SQLException& e) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
    }
    
    int topbottom, luttype;
    if ( _fpga == hcal::ConfigurationDatabase::Top ) topbottom = 1;
    else topbottom = 0;
    if ( _lt == hcal::ConfigurationDatabase::LinearizerLUT ) luttype = 1;
    else luttype = 2;
    
    result = getOnlineLUT( tag, _crate, _slot, topbottom, _fiber, _channel, luttype );
    
  }
  else{
    cout << "Either the XML file with LUTs or the database with LMap are not defined" << endl;
  }

  return result;
}


oracle::occi::Connection * HCALConfigDB::getConnection( void ){
  return database -> getConnection();
}

oracle::occi::Environment * HCALConfigDB::getEnvironment( void ){
  return database -> getEnvironment();
}





