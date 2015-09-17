//
// Gena Kukartsev (Brown), Feb 23, 2008
// $Id:

#include <fstream>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalQIEManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"

#ifdef HAVE_XDAQ
#include "toolbox/string.h"
#else
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Replaces toolbox::toString
#endif

using namespace std;
using namespace oracle::occi;
using namespace hcal;

HcalQIEManager::HcalQIEManager( void )
{    

}



HcalQIEManager::~HcalQIEManager( void )
{    

}



bool HcalChannelId::operator<( const HcalChannelId & other) const
{
  long long int _res_this, _res_other;
  int _sub_this, _sub_other;

  if (this->subdetector == "HE") _sub_this=1;
  else if (this->subdetector == "HF") _sub_this=2;
  else if (this->subdetector == "HO") _sub_this=3;
  else  _sub_this=4;
  
  if (other.subdetector == "HE") _sub_other=1;
  else if (other.subdetector == "HF") _sub_other=2;
  else if (other.subdetector == "HO") _sub_other=3;
  else  _sub_other=4;
  

  _res_this = 100+eta + (phi+100)*1000 + (depth+10)*1000000 + _sub_this*1000000000;
  _res_other = 100+other.eta + (other.phi+100)*1000 + (other.depth+10)*1000000 + _sub_other*1000000000;

  return _res_this < _res_other;
}

std::map<HcalChannelId,HcalQIECaps> & HcalQIEManager::getQIETableFromFile( std::string _filename )
{
  std::map<HcalChannelId,HcalQIECaps> * result_sup = new std::map<HcalChannelId,HcalQIECaps>;
  std::map<HcalChannelId,HcalQIECaps> & result = (*result_sup);

  ifstream infile( _filename . c_str() );
  std::string buf;

  if ( infile . is_open() ){
    std::cout << "File is open" << std::endl;
    while (getline( infile, buf )) {
      std::vector<std::string> _line = splitString( buf );

      HcalChannelId _id;
      sscanf(_line[0].c_str(), "%d", &_id . eta);
      sscanf(_line[1].c_str(), "%d", &_id . phi);
      sscanf(_line[2].c_str(), "%d", &_id . depth);
      _id . subdetector = _line[3];

      HcalQIECaps _adc;
      int _columns = _line . size();
      for(int i = 4; i != _columns; i++){
	sscanf(_line[i].c_str(), "%lf", &_adc . caps[i-4]);
      }

      /* DEBUG: double entries
      if(result.find(_id) != result.end()){
	cout << "TABLE DEBUG: " << _filename << "	" << _id.eta << "	" << _id.phi << "	" << _id.depth << "	" << _id.subdetector << std::endl;
      }
      */

      //result[_id]=_adc;
      result.insert( std::pair<HcalChannelId,HcalQIECaps>(_id, _adc ) );

      //std::cout << result.size() << std::endl;

      //std::cout << _id.eta << "	" << _id . subdetector << "	" << _adc.caps[7] << std::endl;
    }
  }
  return result;
}



// courtesy of Fedor Ratnikov
std::vector <std::string> HcalQIEManager::splitString (const std::string& fLine) {
  std::vector <std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ' ' || fLine [i] == '\n' || fLine [i] == '	' || i == fLine.size ()) {
      if (!empty) {
        std::string item (fLine, start, i-start);
        result.push_back (item);
        empty = true;
      }
      start = i+1;
    }
    else {
      if (empty) empty = false;
    }
  }
  return result;
}



void HcalQIEManager::getTableFromDb( std::string query_file, std::string output_file)
{
  std::cout << "Creating the output file: " << output_file << "... ";
  ofstream out_file;
  out_file . open( output_file.c_str() );
  std::cout << " done" << std::endl;

  HCALConfigDB * db = new HCALConfigDB();
  const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );

  oracle::occi::Connection * _connection = db -> getConnection();  

  std::cout << "Preparing to request the QIE table from the database..." << std::endl;

  //loop over RM slots and QIEs (to save time, otherwise the query runs forever)
  for (int _rm=1; _rm!=5; _rm++){
    for (int _qie=1; _qie!=4; _qie++){
      try {
	cout << "Preparing the query..." << std::endl;
	Statement* stmt = _connection -> createStatement();
	std::string query, buf;
	ifstream inFile( query_file . c_str(), std::ios::in );
	if (!inFile){
	  std::cout << " Unable to open file with query!" << std::endl;
	}
	else{
	  std::cout << "Query file opened successfully: " << query_file << std::endl;
	}
	while (getline( inFile, buf )) {
	  query . append(buf);
	  query . append("\n");
	}
	
	char query_fixed[50000];
	sprintf(query_fixed,query.c_str(),_rm,_rm,_qie,_qie);
	inFile.close();
	cout << "Preparing the query... done" << std::endl;
	
	//SELECT
	cout << "Executing the query..." << std::endl;
	//std::cout << query_fixed << std::endl;
	ResultSet *rs = stmt->executeQuery(query_fixed);
	cout << "Executing the query... done" << std::endl;
	
	cout << "Processing the query results..." << std::endl;
	RooGKCounter _lines(1,100);
	//int count;
	while (rs->next()) {
	  _lines . count();
	  HcalChannelId _id;
	  HcalQIECaps _caps;
	  //count = rs->getInt(1);
	  _id.eta = rs->getInt(1);
	  _id.phi = rs->getInt(2);
	  _id.depth = rs->getInt(3);
	  _id.subdetector  = rs -> getString(4);
	  for (int j=0; j!=32; j++){
	    _caps.caps[j] = rs -> getDouble(j+5);
	  }

	  //==> output QIE table line
	  char buffer[1024];
	  sprintf(buffer, "%15d %15d %15d %15s", _id.eta, _id.phi, _id.depth, _id.subdetector.c_str());
	  //std::cout << buffer;
	  out_file << buffer;
	  for (int j = 0; j != 32; j++){
	    double _x = _caps . caps[j];
	    sprintf(buffer, " %8.5f", _x);
	    //std::cout << buffer;      
	    out_file << buffer;      
	  }
	  //std::cout << std::endl;
	  out_file << std::endl;
	  //===

	}
	//Always terminate statement
	_connection -> terminateStatement(stmt);
	
	//std::cout << "Query count: " << count << std::endl;
	cout << "Query line count: " << _lines.getCount() << std::endl;
      } catch (SQLException& e) {
	XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
      }
    }
  }
  
  db -> disconnect();
  out_file.close();
}



// This function generates proper QIE table ASCII file based on the result
// of the database query from db_file.
// Missing channels are filled from the old table old_file.
// The result is placed in output_file.
int HcalQIEManager::generateQieTable( std::string db_file, std::string old_file, std::string output_file )
{
  std::cout << "Creating the output file: " << output_file << "... ";
  ofstream out_file;
  out_file . open( output_file.c_str() );
  std::cout << " done" << std::endl;

  std::string badchan_file = output_file + ".badchannels";
  std::cout << "Creating the output file for bad channels: " << badchan_file << "... ";
  ofstream bad_file;
  bad_file . open( badchan_file.c_str() );
  std::cout << " done" << std::endl;

  std::map<HcalChannelId,HcalQIECaps> & _old = getQIETableFromFile( old_file . c_str() );
  std::map<HcalChannelId,HcalQIECaps> & _new = getQIETableFromFile( db_file . c_str() );
  //std::map<HcalChannelId,HcalQIECaps> & _old = _manager . getQIETableFromFile( "qie_normalmode_v3.txt" );
  //std::map<HcalChannelId,HcalQIECaps> & _new = _manager . getQIETableFromFile( "qie_adc_table_after.txt" );

  int goodChannels = 0;
  int badChannels = 0;
  std::cout << "old size: " << _old.size() << std::endl;
  std::cout << "new size: " << _new.size() << std::endl;
  for (std::map<HcalChannelId,HcalQIECaps>::const_iterator line=_old.begin(); line!=_old.end(); line++ ){
    HcalQIECaps * the_caps;
    HcalChannelId theId = line -> first;
    bool badchannel = false;
    if (_new.find(theId)==_new.end()){
      badchannel=true;
      badChannels++;
      the_caps = &_old[theId];
    }
    else{
      goodChannels++;
      the_caps = &_new[theId];
    }
    char buffer[1024];
    int eta = theId.eta;
    int phi = theId.phi;
    int depth = theId.depth;
    sprintf(buffer, "%15d %15d %15d %15s", eta, phi, depth, theId.subdetector.c_str());
    out_file << buffer;
    if (badchannel) bad_file << buffer;

    for (int j = 0; j != 32; j++){
      double _x = the_caps->caps[j];
      sprintf(buffer, " %8.5f", _x);
      out_file << buffer;      
      if (badchannel) bad_file << buffer;
    }
    out_file << std::endl;
    if (badchannel) bad_file << std::endl;
  }

  std::cout<< goodChannels<< "   " << badChannels << "   " << goodChannels+badChannels << std::endl;

  out_file . close();
  bad_file . close();

  return 0;
}


// get QIE barcodes per RBX from ascii file based on Tina Vernon's spreadsheet,
// get the LMAP channels based on LMAP from the database (OMDS, validation)
// associate QIEs with channels, then append respective ADC slopes and offsets
// based on QIE normmode table info from the database
// finally, generate the QIE table ascii file for HF
int HcalQIEManager::getHfQieTable( std::string input_file, std::string output_file )
{
  std::cout << "Creating the output file: " << output_file << "... ";
  std::ofstream out_file;
  out_file . open( output_file.c_str() );
  std::cout << " done" << std::endl;

  // process the file with QIE-per-RBX info
  std::map<std::string,std::vector<int> > _qie; // 12 QIE barcodes per RBX: _qie["HFP11"]=...
  std::ifstream infile( input_file . c_str() );
  std::string buf;
  if ( infile . is_open() ){
    std::cout << "File is open" << std::endl;
    getline( infile, buf );
    std::cout << "Table legend: " << std::endl << buf << std::endl;
    while (getline( infile, buf )) {
      std::vector<std::string> _line = splitString( buf );
      if ( _line . size() != 17){
	cout << "Table line is malformed, not clear what to do... exiting." << std::endl;
	return -1;
      }
      std::string _rbx = _line[0];
      std::vector<int> _barcodes;
      for (int i=0; i!=12; i++){
	int _code;
	sscanf(_line[i+5].c_str(),"%d", &_code);
	_barcodes . push_back(_code);
      }
      _qie . insert( std::pair<std::string,std::vector<int> >(_rbx,_barcodes) );
    }
  }

  // database stuff
  HCALConfigDB * db = new HCALConfigDB();
  const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );
  oracle::occi::Connection * _connection = db -> getConnection();  
  std::cout << "Preparing to request the HF channels from LMAP in the database..." << std::endl;
  try {
    std::cout << "Preparing the query..." << std::endl;
    Statement* stmt = _connection -> createStatement();
    
    std::cout << "Preparing the query... done" << std::endl;
    std::string query;
    query . append("select\n");
    query . append("lmap.side*lmap.eta, lmap.phi, lmap.depth, lmap.subdetector,\n");
    query . append("lmap.rbx, lmap.rm_slot, lmap.qie_slot, lmap.adc\n");
    query . append("from cms_hcl_hcal_condition_owner.hcal_hardware_logical_maps_v3 lmap\n");
    query . append("inner join cms_hcl_core_condition_owner.cond_data_sets cds\n");
    query . append("on cds.condition_data_set_id=lmap.condition_data_set_id\n");
    query . append("where cds.version='30'\n");
    query . append("and lmap.subdetector='HF'\n");
    query . append("order by lmap.rbx, lmap.rm_slot, lmap.qie_slot, lmap.adc\n");
    
    //SELECT
    std::cout << "Executing the query..." << std::endl;
    //std::cout << query << std::endl;
    ResultSet *rs = stmt->executeQuery(query.c_str());
    std::cout << "Executing the query... done" << std::endl;
    
    std::cout << "Processing the query results..." << std::endl;
    RooGKCounter _lines(1,100);
    while (rs->next()) {
      _lines . count();
      HcalChannelId _id;
      std::string rbx;
      int rm_slot, qie_slot, adc, qie_barcode;
      //count = rs->getInt(1);
      _id.eta = rs->getInt(1);
      _id.phi = rs->getInt(2);
      _id.depth = rs->getInt(3);
      _id.subdetector  = rs -> getString(4);
      rbx = rs->getString(5);
      rm_slot = rs->getInt(6);
      qie_slot = rs->getInt(7);
      adc = rs->getInt(8);
      qie_barcode = _qie[rbx][(rm_slot-1)*4+qie_slot-1];

      //==>another DB query to get the ADC caps' slopes and offsets
      // NOTE! In HF slope range and capID seem to be exchanged
      try {
	//std::cout << "Preparing the query..." << std::endl;
	Statement* stmt2 = _connection -> createStatement();
	std::string query2;
	query2 . append("select\n");
	query2 . append("vadcs.cap0_range0_offset,vadcs.cap0_range1_offset,vadcs.cap0_range2_offset,vadcs.cap0_range3_offset,\n");
	query2 . append("vadcs.cap1_range0_offset,vadcs.cap1_range1_offset,vadcs.cap1_range2_offset,vadcs.cap1_range3_offset,\n");
	query2 . append("vadcs.cap2_range0_offset,vadcs.cap2_range1_offset,vadcs.cap2_range2_offset,vadcs.cap2_range3_offset,\n");
	query2 . append("vadcs.cap3_range0_offset,vadcs.cap3_range1_offset,vadcs.cap3_range2_offset,vadcs.cap3_range3_offset,\n");
	query2 . append("vadcs.cap0_range0_slope,vadcs.cap0_range1_slope,vadcs.cap0_range2_slope,vadcs.cap0_range3_slope,\n");
	query2 . append("vadcs.cap1_range0_slope,vadcs.cap1_range1_slope,vadcs.cap1_range2_slope,vadcs.cap1_range3_slope,\n");
	query2 . append("vadcs.cap2_range0_slope,vadcs.cap2_range1_slope,vadcs.cap2_range2_slope,vadcs.cap2_range3_slope,\n");
	query2 . append("vadcs.cap3_range0_slope,vadcs.cap3_range1_slope,vadcs.cap3_range2_slope,vadcs.cap3_range3_slope\n");
	query2 . append("from CMS_HCL_HCAL_CONDITION_OWNER.V_QIECARD_ADC_NORMMODE vadcs\n");
	query2 . append("where substr(vadcs.name_label,14,6)='%d'\n");
	query2 . append("and substr(vadcs.name_label,21,1)='%d'\n");
	query2 . append("order by version desc,record_id desc, condition_data_set_id desc\n");
	char query2_fixed[5000];
	sprintf(query2_fixed,query2.c_str(),qie_barcode,adc);
    	//std::cout << "Preparing the query... done" << std::endl;
	//std::cout << query2_fixed << std::endl;

	//SELECT
	//std::cout << "Executing the query..." << std::endl;
	//std::cout << query2 << std::endl;
	ResultSet *rs2 = stmt2->executeQuery(query2_fixed);
	//std::cout << "Executing the query... done" << std::endl;
    
	//std::cout << "Processing the query results..." << std::endl;
	// take only the first line - sorted by version descending, latest on top
	if (rs2->next()) {
	  HcalQIECaps _caps;
	  for (int j=0; j!=32; j++){
	    _caps.caps[j] = rs2 -> getDouble(j+1);
	  }	  
	  //==> output QIE table line
	  char buffer[1024];
	  sprintf(buffer, "%15d %15d %15d %15s", _id.eta, _id.phi, _id.depth, _id.subdetector.c_str());
	  //std::cout << buffer;
	  out_file << buffer;
	  for (int j = 0; j != 32; j++){
	    double _x = _caps . caps[j];
	    sprintf(buffer, " %8.5f", _x);
	    //std::cout << buffer;      
	    out_file << buffer;      
	  }
	  //std::cout << std::endl;
	  out_file << std::endl;
	  //===
	}
	//Always terminate statement
	_connection -> terminateStatement(stmt2);
	
      } catch (SQLException& e) {
	XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
      }
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
    
    //std::cout << "Query count: " << count << std::endl;
    std::cout << "Query line count: " << _lines.getCount() << std::endl;
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  
  db -> disconnect();
  out_file.close();
  
  return 0;
}
