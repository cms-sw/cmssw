
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

HCALConfigDB::HCALConfigDB( string _accessor )
{    
  accessor = _accessor;
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



void HCALConfigDB::disconnect( void )
{
  database -> disconnect();
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

  long long int _condition_data_set_id;
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
    std::string query = ("SELECT CONDITION_DATA_SET_ID, CRATE, HTR_SLOT, HTR_FPGA, HTR_FIBER, FIBER_CHANNEL ");
    query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V3 ";
    query += toolbox::toString(" WHERE SIDE=%d AND ETA=%d AND PHI=%d AND DEPTH=%d AND SUBDETECTOR='%s'", side, etaAbs, phi, depth, subdetector . c_str() );
    
    //SELECT
    ResultSet *rs = stmt->executeQuery(query.c_str());

    while (rs->next()) {
      long long int _cdsi = rs -> getInt(1);
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
	}
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  
  std::vector<unsigned int> result = getOnlineLUT( tag, _crate, _slot, _fpga, _fiber, _channel, _lt );
  
  return result;
}

/*
hcal::ConfigurationDatabase::LUTId HCALConfigDB::getLUTId( uint32_t _rawid )
{


  hcal::ConfigurationDatabase::LUTId result();

  return result;
}
*/
