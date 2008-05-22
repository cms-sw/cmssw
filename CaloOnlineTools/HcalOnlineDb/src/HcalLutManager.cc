#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalQIEManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
//#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"

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


HcalLutSet HcalLutManager::getLutSetFromFile( string _filename, int _type )
{
  HcalLutSet _lutset;

  ifstream infile( _filename . c_str() );
  string buf;

  if ( infile . is_open() ){
    cout << "File " << _filename << " is open..." << endl;
    cout << "Reading LUTs and their eta/phi/depth/subdet ranges...";

    // get label
    getline( infile, _lutset . label );

    if ( _type == 1 ){ // for linearization LUTs get subdetectors (default)
      //get subdetectors
      getline( infile, buf );
      _lutset . subdet = HcalQIEManager::splitString( buf );
    }

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

    if ( _type == 1 ){ // for linearization LUTs get depth range (default)
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

std::map<int, shared_ptr<LutXml> > HcalLutManager::getLutXmlFromAsciiMaster( string _filename, string _tag, int _crate, bool split_by_crate )
{
  cout << "Generating linearization (input) LUTs from ascii master file..." << endl;
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
  cout << "Generating linearization (input) LUTs from ascii master file...DONE" << endl;
  return _xml;
}


int HcalLutManager::writeLutXmlFiles( std::map<int, shared_ptr<LutXml> > & _xml, string _tag, bool split_by_crate )
{
  for (map<int,shared_ptr<LutXml> >::const_iterator cr = _xml.begin(); cr != _xml.end(); cr++){
    stringstream output_file_name;
    if ( split_by_crate ){
      output_file_name << _tag << "_" << cr->first << ".xml";
    }
    else{
      output_file_name << _tag << ".xml";
    }
    cr->second->write( output_file_name.str().c_str() );
  }
  return 0;
}

int HcalLutManager::createAllLutXmlFiles( string _tag, string _lin_file, string _comp_file, bool split_by_crate )
{
  std::map<int, shared_ptr<LutXml> > xml;
  if ( _lin_file.size() != 0 ){
    addLutMap( xml, getLutXmlFromAsciiMaster( _lin_file, _tag ) );
  }
  if ( _comp_file.size() != 0 ){
    cout << "DEBUG1!!!!" << endl;
    addLutMap( xml, getCompressionLutXmlFromAsciiMaster( _comp_file, _tag ) );
    cout << "DEBUG2!!!!" << endl;
  }
  writeLutXmlFiles( xml, _tag, split_by_crate );
  return 0;
}



void HcalLutManager::addLutMap(std::map<int, shared_ptr<LutXml> > & result, const std::map<int, shared_ptr<LutXml> > & other)
{
  for ( std::map<int, shared_ptr<LutXml> >::const_iterator lut=other.begin(); lut!=other.end(); lut++ ){
    if ( result.count(lut->first)==0 ){
      result . insert( *lut );
    }
    else{
      //LutXml a;
      //(*(result[lut->first])) . write("stdout");
      //LutXml b;
      //(*(lut->second)) . write("stdout");
      //a.write("stdout");
      //a+=b;
      *(result[lut->first]) += *(lut->second);
    }
  }
}


std::map<int, shared_ptr<LutXml> > HcalLutManager::getCompressionLutXmlFromAsciiMaster( string _filename, string _tag, int _crate, bool split_by_crate )
{
  cout << "Generating compression (output) LUTs from ascii master file..." << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v5_080208.txt");
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  cout << "EMap contains " << _map . size() << " channels" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename, 2 );
  int lut_set_size = _set.lut.size(); // number of different luts

  //loop over all EMap channels
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;

    // search for the correct LUT for a given channel,
    // higher LUT numbers have priority in case of overlapping
    int lut_index=-1;
    for ( int i=0; i<lut_set_size; i++ ){
      if ( row->subdet . find("HT") != string::npos &&
	   (row->crate == _crate || _crate == -1) && // -1 stands for all crates
	   _set.eta_min[i] <= row->ieta &&
	   _set.eta_max[i] >= row->ieta &&
	   _set.phi_min[i] <= row->iphi &&
	   _set.phi_max[i] >= row->iphi ){
	lut_index=i;
      }
    }
    if ( lut_index >= 0 ){
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(row->crate,shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(0,shared_ptr<LutXml>(new LutXml())) );
      }
      _cfg.ieta = row->ieta;
      _cfg.iphi = row->iphi;
      _cfg.depth = row->idepth;
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom . find("t") != string::npos) _cfg.topbottom = 1;
      else if (row->topbottom . find("b") != string::npos) _cfg.topbottom = 0;
      else cout << "Warning! fpga out of range..." << endl;
      _cfg.fiber = row->fiber;
      _cfg.fiberchan = row->fiberchan;
      if (_set.lut[lut_index].size() == 128) _cfg.lut_type = 1;
      else _cfg.lut_type = 2;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp( time(0) );
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1"; //???
      _cfg.generalizedindex =
	_cfg.iphi*10000+
	(row->ieta>0)*100+abs(row->ieta);
      _cfg.lut = _set.lut[lut_index];
      if (split_by_crate ){
	_xml[row->crate]->addLut( _cfg );  
      }
      else{
	_xml[0]->addLut( _cfg );  
      }
    }
  }
  cout << "Generating compression (output) LUTs from ascii master file...DONE" << endl;
  return _xml;
}



string HcalLutManager::get_time_stamp( time_t _time )
{
  char timebuf[50];
  //strftime( timebuf, 50, "%c", gmtime( &_time ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S", gmtime( &_time ) );
  string creationstamp = timebuf;

  return creationstamp;
}




int HcalLutManager::test_xml_access( string _tag, string _filename )
{

  HCALConfigDB * db = new HCALConfigDB();
  db -> connect( _filename, "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22" );

  //vector<unsigned int> _lut = db -> getOnlineLUTFromXML( "emap_hcal_emulator_test_luts", 17, 2, 1, 1, 0, 1 );
  //vector<unsigned int> _lut = db -> getOnlineLUTFromXML( "GREN_170_realped", 17, 2, 1, 1, 0, 1 );

  struct timeval _t;
  gettimeofday( &_t, NULL );
  cout << "before getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;
  vector<unsigned int> _lut = db -> getOnlineLUT( _tag, 17, 2, 1, 1, 0, 1 );
  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;
  _lut = db -> getOnlineLUT( _tag, 15, 2, 1, 1, 0, 1 );
  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;
  _lut = db -> getOnlineLUT( _tag, 17, 2, 1, 1, 0, 1 );
  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;
  _lut = db -> getOnlineLUT( _tag, 9, 2, 1, 1, 0, 1 );
  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;
  _lut = db -> getOnlineLUT( _tag, 0, 2, 1, 1, 0, 1 );
  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;

  /*
  HcalDetId _hcaldetid( HcalBarrel, -11, 12, 1 );

  struct timeval _t;
  gettimeofday( &_t, NULL );
  cout << "before getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;

  vector<unsigned int> _lut = db -> getOnlineLUTFromXML( _tag, _hcaldetid . rawId() );

  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;

  HcalDetId _hcaldetid2( HcalBarrel, -11, 13, 1 );
  _lut = db -> getOnlineLUTFromXML( _tag, _hcaldetid2 . rawId() );

  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;

  _lut = db -> getOnlineLUTFromXML( _tag, _hcaldetid . rawId() );

  gettimeofday( &_t, NULL );
  cout << "after getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;

  */


  cout << "LUT length = " << _lut . size() << endl;
  for ( vector<unsigned int>::const_iterator i = _lut . end() - 1; i != _lut . begin()-1; i-- )
    {
      cout << (i-_lut.begin()) << "     " << _lut[(i-_lut.begin())] << endl;
      break;
    }


  db -> disconnect();
  

  return 0;
}
