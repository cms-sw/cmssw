
//
// Gena Kukartsev (Brown), Feb 1, 2008
// $Id:
//

#include <iostream>
#include <string.h>

#include "xgi/Utils.h"
#include "toolbox/string.h"

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
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
  database = 0;
  database2 = 0;
}

HCALConfigDB::HCALConfigDB( std::string _accessor )
{    
  database = 0;
  database2 = 0;
  accessor = _accessor;
}


HCALConfigDB::~HCALConfigDB( void )
{    
  delete database;
  delete database2;
}


void HCALConfigDB::setAccessor( std::string _accessor )
{      
  accessor = _accessor;
}

void HCALConfigDB::connect( std::string _accessor )
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



void HCALConfigDB::connect( std::string _accessor1, std::string _accessor2 )
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



std::vector<unsigned int> HCALConfigDB::getOnlineLUT( std::string tag, int crate, int slot, int topbottom, int fiber, int channel, int luttype )
{

  //connect( accessor );

  std::vector<unsigned int> result;

  hcal::ConfigurationDatabase::FPGASelection _fpga;
  if ( topbottom == 0 ) _fpga = hcal::ConfigurationDatabase::Bottom;
  else if ( topbottom == 1 ) _fpga = hcal::ConfigurationDatabase::Top;
  else
    {
      std::cout << "topbottom out of range" << std::endl;
      exit(-1);
    }

  hcal::ConfigurationDatabase::LUTType _lt;
  if ( luttype == 1 ) _lt = hcal::ConfigurationDatabase::LinearizerLUT;
  else if ( luttype == 2 ) _lt = hcal::ConfigurationDatabase::CompressionLUT;
  else
    {
      std::cout << "LUT type out of range" << std::endl;
      exit(-1);
    }

  hcal::ConfigurationDatabase::LUTId _lutid( crate, slot, _fpga, fiber, channel, _lt );
  std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT> testLUTs;

  XMLProcessor::getInstance();

  try {
    database -> getLUTs(tag, crate, slot, testLUTs);
  } catch (hcal::exception::ConfigurationItemNotFoundException& e) {
    std::cout << "Found nothing!" << std::endl;
  } catch (hcal::exception::Exception& e2) {
    std::cout << "Exception: " << e2.what() << std::endl;
  }

  result = testLUTs[_lutid];

  //database -> disconnect();

  return result;
}

std::vector<unsigned int> HCALConfigDB::getOnlineLUT( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt )
{
  std::vector<unsigned int> result;
  HcalDetId _id( _rawid );

  double _condition_data_set_id;
  unsigned int _crate, _slot, _fiber, _channel;
  hcal::ConfigurationDatabase::FPGASelection _fpga;

  int side   = _id . zside();
  int etaAbs = _id . ietaAbs();
  int phi    = _id . iphi();
  int depth  = _id . depth();
  std::string subdetector;
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
	  
	  int topbottom, luttype;
	  if ( _fpga == hcal::ConfigurationDatabase::Top ) topbottom = 1;
	  else topbottom = 0;
	  if ( _lt == hcal::ConfigurationDatabase::LinearizerLUT ) luttype = 1;
	  else luttype = 2;
	  
	  result = getOnlineLUT( tag, _crate, _slot, topbottom, _fiber, _channel, luttype );
	}
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  return result;
}


std::vector<unsigned int> HCALConfigDB::getOnlineLUTFromXML( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt ){

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
    std::string subdetector;
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
	    
	    //
	    int topbottom, luttype;
	    if ( _fpga == hcal::ConfigurationDatabase::Top ) topbottom = 1;
	    else topbottom = 0;
	    if ( _lt == hcal::ConfigurationDatabase::LinearizerLUT ) luttype = 1;
	    else luttype = 2;
	    result = getOnlineLUT( tag, _crate, _slot, topbottom, _fiber, _channel, luttype );
	  }
      }
      //Always terminate statement
      _connection -> terminateStatement(stmt);

    } catch (SQLException& e) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
    }
  }
  else{
    std::cout << "Either the XML file with LUTs or the database with LMap are not defined" << std::endl;
  }

  return result;
}


oracle::occi::Connection * HCALConfigDB::getConnection( void ){
  return database -> getConnection();
}

oracle::occi::Environment * HCALConfigDB::getEnvironment( void ){
  return database -> getEnvironment();
}



//Utility function that cnverts oracle::occi::Clob to std::string
string HCALConfigDB::clobToString(const oracle::occi::Clob& _clob){
		oracle::occi::Clob clob = _clob;
                Stream *instream = clob.getStream (1,0);
		unsigned int size = clob.length();
                char *cbuffer = new char[size];
                memset (cbuffer, 0, size);
                instream->readBuffer (cbuffer, size);
                std::string str(cbuffer,size);
		return str;
}
