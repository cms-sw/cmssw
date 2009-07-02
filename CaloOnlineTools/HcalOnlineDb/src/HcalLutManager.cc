#include <fstream>
#include <sstream>
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalQIEManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"

using namespace std;

/**

   \class HcalLutManager
   \brief Various manipulations with trigger Lookup Tables
   \author Gena Kukartsev, Brown University, March 14, 2008

*/

HcalLutManager::HcalLutManager( void )
{    
  init();
}



void HcalLutManager::init( void )
{    
  lut_xml = NULL;
}



HcalLutManager::~HcalLutManager( void )
{    
  if (lut_xml) delete lut_xml;
}



std::string & HcalLutManager::getLutXml( std::vector<unsigned int> & _lut )
{

  if (lut_xml) delete lut_xml;

  lut_xml = new LutXml();

  LutXml::Config _config;
  _config.lut = _lut;
  lut_xml -> addLut( _config );
  lut_xml -> addLut( _config );
  lut_xml -> addLut( _config );

  //return lut_xml->getString();
  return lut_xml->getCurrentBrick();

}


int HcalLutManager::getInt( string number )
{
  int result;
  sscanf(number.c_str(), "%d", &result);
  return result;
}

HcalSubdetector HcalLutManager::get_subdetector( string _det )
{
  HcalSubdetector result;
  if      ( _det.find("HB") != string::npos ) result = HcalBarrel;
  else if ( _det.find("HE") != string::npos ) result = HcalEndcap;
  else if ( _det.find("HF") != string::npos ) result = HcalForward;
  else if ( _det.find("HO") != string::npos ) result = HcalOuter;
  else                                        result = HcalOther;  

  return result;
}


int HcalLutManager_test::getLutSetFromFile_test( string _filename )
{
  HcalLutManager _manager;
  HcalLutSet _set = _manager . getLutSetFromFile( _filename );
  cout << "===> Test of HcalLutSet HcalLutManager::getLutSetFromFile( string _filename )" << endl << endl;
  cout << _set . label << endl;
  for (unsigned int i = 0; i != _set.subdet.size(); i++) cout << _set.subdet[i] << "	";
  cout << endl;
  for (unsigned int i = 0; i != _set.eta_min.size(); i++) cout << _set.eta_min[i] << "	";
  cout << endl;
  for (unsigned int i = 0; i != _set.eta_max.size(); i++) cout << _set.eta_max[i] << "	";
  cout << endl;
  for (unsigned int i = 0; i != _set.phi_min.size(); i++) cout << _set.phi_min[i] << "	";
  cout << endl;
  for (unsigned int i = 0; i != _set.phi_max.size(); i++) cout << _set.phi_max[i] << "	";
  cout << endl;
  for (unsigned int i = 0; i != _set.depth_min.size(); i++) cout << _set.depth_min[i] << "	";
  cout << endl;
  for (unsigned int i = 0; i != _set.depth_max.size(); i++) cout << _set.depth_max[i] << "	";
  cout << endl;
  for (unsigned int j = 0; j != _set.lut[0].size(); j++){
    for (unsigned int i = 0; i != _set.lut.size(); i++){
      cout << _set.lut[i][j] << "	";
    }
    cout << "---> " << j << endl;
  }
}


HcalLutSet HcalLutManager::getLutSetFromFile( string _filename )
{
  HcalLutSet _lutset;

  ifstream infile( _filename . c_str() );
  string buf;

  if ( infile . is_open() ){
    cout << "File " << _filename << " is open..." << endl;
    cout << "Reading LUTs and their eta/phi/depth/subdet ranges...";

    // get label
    getline( infile, _lutset . label );

    //get subdetectors
    getline( infile, buf );
    _lutset . subdet = HcalQIEManager::splitString( buf );

    //get min etas
    vector<string> buf_vec;
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (vector<string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.eta_min.push_back(HcalLutManager::getInt(*iter));
    }

    //get max etas
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (vector<string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.eta_max.push_back(HcalLutManager::getInt(*iter));
    }

    //get min phis
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (vector<string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.phi_min.push_back(HcalLutManager::getInt(*iter));
    }

    //get max phis
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (vector<string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.phi_max.push_back(HcalLutManager::getInt(*iter));
    }

    //get min depths
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (vector<string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.depth_min.push_back(HcalLutManager::getInt(*iter));
    }

    //get max depths
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (vector<string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.depth_max.push_back(HcalLutManager::getInt(*iter));
    }

    bool first_lut_entry = true;
    while ( getline( infile, buf ) > 0 ){
      buf_vec = HcalQIEManager::splitString( buf );
      for (unsigned int i = 0; i < buf_vec.size(); i++){
	if (first_lut_entry){
	  vector<unsigned int> _l;
	  _lutset.lut.push_back(_l);
	}
	_lutset . lut[i] . push_back(HcalLutManager::getInt(buf_vec[i]));
      }
      first_lut_entry = false;
    }
  }

  cout << "done." << endl;

  return _lutset;
}

std::string HcalLutManager::getLutXmlFromAsciiMaster( string _filename, string _tag, int _crate, bool split_by_crate )
{
  //shared_ptr<LutXml> _xml( new LutXml() );
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  LMap _lmap;
  _lmap . read( "HCALmapHBEF.txt", "HBEF" );
  _lmap . read( "HCALmapHO.txt", "HO" );
  map<int,LMapRow> & _map = _lmap.get_map();
  cout << "LMap contains " << _map . size() << " channels" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename );
  int lut_set_size = _set.lut.size(); // number of different luts

  //loop over all HCAL channels
  for( map<int,LMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;

    // search for the correct LUT for a given channel,
    // higher LUT numbers have priority in case of overlapping
    int lut_index=-1;
    for ( int i=0; i<lut_set_size; i++ ){
      if ( (row->second.crate == _crate || _crate == -1) && // -1 stands for all crates
	   _set.eta_min[i] <= row->second.side*row->second.eta &&
	   _set.eta_max[i] >= row->second.side*row->second.eta &&
	   _set.phi_min[i] <= row->second.phi &&
	   _set.phi_max[i] >= row->second.phi &&
	   _set.depth_min[i] <= row->second.depth &&
	   _set.depth_max[i] >= row->second.depth &&
	   get_subdetector(_set.subdet[i]) == row->second.det ){
	lut_index=i;
      }
    }
    if ( lut_index >= 0 ){
      if ( _xml.count(row->second.crate) == 0 && split_by_crate ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(row->second.crate,shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(0,shared_ptr<LutXml>(new LutXml())) );
      }
      _cfg.ieta = row->second.side*row->second.eta;
      _cfg.iphi = row->second.phi;
      _cfg.depth = row->second.depth;
      _cfg.crate = row->second.crate;
      _cfg.slot = row->second.htr;
      if (row->second.fpga . find("top") != string::npos) _cfg.topbottom = 1;
      else if (row->second.fpga . find("bot") != string::npos) _cfg.topbottom = 0;
      else cout << "Warning! fpga out of range..." << endl;
      _cfg.fiber = row->second.rm_fi;
      _cfg.fiberchan = row->second.fi_ch;
      if (_set.lut[lut_index].size() == 128) _cfg.lut_type = 1;
      else _cfg.lut_type = 2;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp( time(0) );
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1"; //???
      _cfg.generalizedindex =
	_cfg.iphi*10000+_cfg.depth*1000+
	(row->second.side>0)*100+row->second.eta;
      _cfg.lut = _set.lut[lut_index];
      if (split_by_crate ){
	_xml[row->second.crate]->addLut( _cfg );  
      }
      else{
	_xml[0]->addLut( _cfg );  
      }
    }
  }

  /*
  string result = _xml->getString();
  stringstream output_file_name;
  if (_crate<0) output_file_name << _tag << "_all" << ".xml";
  else output_file_name << _tag << "_" << _crate << ".xml";
  //_xml -> write("LUT_GR0T_v1.0.xml");
  _xml -> write( output_file_name.str().c_str() );
  */

  string result;
  for (map<int,shared_ptr<LutXml> >::const_iterator cr = _xml.begin(); cr != _xml.end(); cr++){
    result = cr->second->getString();
    stringstream output_file_name;
    if ( split_by_crate ){
      output_file_name << _tag << "_" << cr->first << ".xml";
    }
    else{
      output_file_name << _tag << ".xml";
    }
    cr->second->write( output_file_name.str().c_str() );
  }

  return result;
}



string HcalLutManager::get_time_stamp( time_t _time )
{
  char timebuf[50];
  //strftime( timebuf, 50, "%c", gmtime( &_time ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S", gmtime( &_time ) );
  string creationstamp = timebuf;

  return creationstamp;
}
