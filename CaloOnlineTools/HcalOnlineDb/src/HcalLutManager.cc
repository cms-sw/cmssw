#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>  // For srand() and rand()

#include "xgi/Utils.h"
#include "toolbox/string.h"
#include "occi.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalQIEManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLLUTLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"


using namespace std;
using namespace oracle::occi;
using namespace hcal;

/**

   \class HcalLutManager
   \brief Various manipulations with trigger Lookup Tables
   \author Gena Kukartsev, Brown University, March 14, 2008

*/

HcalLutManager::HcalLutManager( void )
{    
  init();
}


HcalLutManager::HcalLutManager(std::vector<HcalGenericDetId> & map)
{
  init();
  _iter . init(map);
}


HcalLutManager::HcalLutManager(const HcalElectronicsMap * _emap)
{
  init();
  emap = _emap;
}


void HcalLutManager::init( void )
{    
  lut_xml = 0;
  lut_checksums_xml = 0;
  db = 0;
  lmap = 0;
  emap = 0;
}



HcalLutManager::~HcalLutManager( void )
{    
  delete lut_xml;
  delete lut_checksums_xml;
  delete db;
  delete lmap;
}


int HcalLutManager::initChannelIterator(std::vector<HcalGenericDetId> & map)
{
  _iter . init(map);
  return _iter.size();
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
  return 0;
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
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  LMap _lmap;
  _lmap . read( "./backup/HCALmapHBEF.txt", "HBEF" );
  _lmap . read( "./backup/HCALmapHO.txt", "HO" );
  map<int,LMapRow> & _map = _lmap.get_map();
  cout << "LMap contains " << _map . size() << " channels" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename );
  int lut_set_size = _set.lut.size(); // number of different luts

  RooGKCounter _counter;
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
      else if ( _xml.count(0) == 0 && !split_by_crate ){
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
      // FIXME: probably fixed. fiber==htr_fi, not rm_fi in LMAP notation.
      //_cfg.fiber = row->second.rm_fi;
      _cfg.fiber = row->second.htr_fi;
      _cfg.fiberchan = row->second.fi_ch;
      if (_set.lut[lut_index].size() == 128) _cfg.lut_type = 1;
      else _cfg.lut_type = 2;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp( time(0) );
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1"; //???
      // "original" definition of GENERALIZEDINDEX from Mike Weinberger
      //    int generalizedIndex=id.ietaAbs()+1000*id.depth()+10000*id.iphi()+
      //        ((id.ieta()<0)?(0):(100))+((id.subdet()==HcalForward && id.ietaAbs()==29)?(4*10000):(0));
      _cfg.generalizedindex =
	_cfg.iphi*10000 + _cfg.depth*1000 +
	(row->second.side>0)*100 + row->second.eta +
	((row->second.det==HcalForward && row->second.eta==29)?(4*10000):(0));
      _cfg.lut = _set.lut[lut_index];
      if (split_by_crate ){
	_xml[row->second.crate]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
      else{
	_xml[0]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
    }
  }
  cout << "LUTs generated: " << _counter.getCount() << endl;
  cout << "Generating linearization (input) LUTs from ascii master file...DONE" << endl;
  return _xml;
}


std::map<int, shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromAsciiMasterEmap( string _filename, string _tag, int _crate, bool split_by_crate )
{
  cout << "Generating linearization (input) LUTs from ascii master file..." << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  cout << "EMap contains " << _map . size() << " entries" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename );
  int lut_set_size = _set.lut.size(); // number of different luts
  cout << "  ==> " << lut_set_size << " sets of different LUTs read from the master file" << endl;

  RooGKCounter _counter;
  //loop over all EMap channels
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    if( (row->subdet.find("HB")!=string::npos ||
	 row->subdet.find("HE")!=string::npos ||
	 row->subdet.find("HO")!=string::npos ||
	 row->subdet.find("HF")!=string::npos ) &&
	row->subdet.size()==2
	){
      LutXml::Config _cfg;
      
      // search for the correct LUT for a given channel,
      // higher LUT numbers have priority in case of overlapping
      int lut_index=-1;
      for ( int i=0; i<lut_set_size; i++ ){
	if ( (row->crate == _crate || _crate == -1) && // -1 stands for all crates
	     _set.eta_min[i] <= row->ieta &&
	     _set.eta_max[i] >= row->ieta &&
	     _set.phi_min[i] <= row->iphi &&
	     _set.phi_max[i] >= row->iphi &&
	     _set.depth_min[i] <= row->idepth &&
	     _set.depth_max[i] >= row->idepth &&
	     _set.subdet[i].find(row->subdet)!=string::npos ){
	  lut_index=i;
	}
      }
      if ( lut_index >= 0 ){
	if ( _xml.count(row->crate) == 0 && split_by_crate ){
	  _xml.insert( pair<int,shared_ptr<LutXml> >(row->crate,shared_ptr<LutXml>(new LutXml())) );
	}
	else if ( _xml.count(0) == 0 && !split_by_crate ){
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
	_cfg.lut_type = 1;
	_cfg.creationtag = _tag;
	_cfg.creationstamp = get_time_stamp( time(0) );
	_cfg.targetfirmware = "1.0.0";
	_cfg.formatrevision = "1"; //???
	// "original" definition of GENERALIZEDINDEX from Mike Weinberger
	//    int generalizedIndex=id.ietaAbs()+1000*id.depth()+10000*id.iphi()+
	//        ((id.ieta()<0)?(0):(100))+((id.subdet()==HcalForward && id.ietaAbs()==29)?(4*10000):(0));
	_cfg.generalizedindex =
	  _cfg.iphi*10000 + _cfg.depth*1000 +
	  (row->ieta>0)*100 + abs(row->ieta) +
	  (((row->subdet.find("HF")!=string::npos) && abs(row->ieta)==29)?(4*10000):(0));
	_cfg.lut = _set.lut[lut_index];
	if (split_by_crate ){
	  _xml[row->crate]->addLut( _cfg, lut_checksums_xml );  
	  _counter.count();
	}
	else{
	  _xml[0]->addLut( _cfg, lut_checksums_xml );  
	  _counter.count();
	}
      }
    }
  }
  cout << "LUTs generated: " << _counter.getCount() << endl;
  cout << "Generating linearization (input) LUTs from ascii master file...DONE" << endl;
  return _xml;
}


std::map<int, shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromAsciiMasterEmap_new( string _filename, string _tag, int _crate, bool split_by_crate )
{
  cout << "Generating linearization (input) LUTs from ascii master file..." << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename );
  int lut_set_size = _set.lut.size(); // number of different luts
  cout << "  ==> " << lut_set_size << " sets of different LUTs read from the master file" << endl;

  RooGKCounter _counter;
  //loop over all EMap channels
  for( _iter.begin(); !_iter.end(); _iter.next() ){
    HcalSubdetector _subdet = _iter.getHcalSubdetector();
    if( (_subdet == HcalBarrel ||
	 _subdet == HcalEndcap ||
	 _subdet == HcalForward ||
	 _subdet == HcalOuter )
	){
      int _ieta  = _iter.getIeta();
      int _iphi  = _iter.getIphi();
      int _depth = _iter.getDepth();

      HcalElectronicsId _eId(_iter.getHcalGenericDetId().rawId());
      int aCrate      = _eId . readoutVMECrateId();
      int aSlot       = _eId . htrSlot();
      int aTopBottom  = _eId . htrTopBottom();
      int aFiber      = _eId . fiberIndex();
      int aFiberChan  = _eId . fiberChanId();

      LutXml::Config _cfg;
      
      // search for the correct LUT for a given channel,
      // higher LUT numbers have priority in case of overlapping
      int lut_index=-1;
      for ( int i=0; i<lut_set_size; i++ ){
	if ( (aCrate == _crate || _crate == -1) && // -1 stands for all crates
	     _set.eta_min[i] <= _ieta &&
	     _set.eta_max[i] >= _ieta &&
	     _set.phi_min[i] <= _iphi &&
	     _set.phi_max[i] >= _iphi &&
	     _set.depth_min[i] <= _depth &&
	     _set.depth_max[i] >= _depth &&
	     _set.subdet[i].find(_ass.getSubdetectorString(_subdet))!=string::npos ){
	  lut_index=i;
	}
      }
      if ( lut_index >= 0 ){
	if ( _xml.count(aCrate) == 0 && split_by_crate ){
	  _xml.insert( pair<int,shared_ptr<LutXml> >(aCrate,shared_ptr<LutXml>(new LutXml())) );
	}
	else if ( _xml.count(0) == 0 && !split_by_crate ){
	  _xml.insert( pair<int,shared_ptr<LutXml> >(0,shared_ptr<LutXml>(new LutXml())) );
	}
	_cfg.ieta = _ieta;
	_cfg.iphi = _iphi;
	_cfg.depth = _depth;
	_cfg.crate = aCrate;
	_cfg.slot = aSlot;
	_cfg.topbottom = aTopBottom;
	_cfg.fiber = aFiber;
	_cfg.fiberchan = aFiberChan;
	_cfg.lut_type = 1;
	_cfg.creationtag = _tag;
	_cfg.creationstamp = get_time_stamp( time(0) );
	_cfg.targetfirmware = "1.0.0";
	_cfg.formatrevision = "1"; //???
	// "original" definition of GENERALIZEDINDEX from Mike Weinberger
	//    int generalizedIndex=id.ietaAbs()+1000*id.depth()+10000*id.iphi()+
	//        ((id.ieta()<0)?(0):(100))+((id.subdet()==HcalForward && id.ietaAbs()==29)?(4*10000):(0));
	_cfg.generalizedindex =
	  _cfg.iphi*10000 + _cfg.depth*1000 +
	  (_ieta>0)*100 + abs(_ieta) +
	  (((_subdet==HcalForward) && abs(_ieta)==29)?(4*10000):(0));
	_cfg.lut = _set.lut[lut_index];
	if (split_by_crate ){
	  _xml[aCrate]->addLut( _cfg, lut_checksums_xml );  
	  _counter.count();
	}
	else{
	  _xml[0]->addLut( _cfg, lut_checksums_xml );  
	  _counter.count();
	}
      }
    }
  }
  cout << "LUTs generated: " << _counter.getCount() << endl;
  cout << "Generating linearization (input) LUTs from ascii master file...DONE" << endl;
  return _xml;
}


std::map<int, shared_ptr<LutXml> > HcalLutManager::getCompressionLutXmlFromAsciiMaster( string _filename, string _tag, int _crate, bool split_by_crate )
{
  cout << "Generating compression (output) LUTs from ascii master file..." << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  cout << "instantiating CaloTPGTranscoderULUT in order to check the validity of (ieta,iphi)..." << endl;
  CaloTPGTranscoderULUT _coder;

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  cout << "EMap contains " << _map . size() << " channels" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename, 2 );
  int lut_set_size = _set.lut.size(); // number of different luts
  cout << "  ==> " << lut_set_size << " sets of different LUTs read from the master file" << endl;

  //loop over all EMap channels
  RooGKCounter _counter;
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
	   _set.phi_max[i] >= row->iphi &&
	   _coder.HTvalid(row->ieta, row->iphi) ){
	lut_index=i;
      }
    }
    if ( lut_index >= 0 ){
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(row->crate,shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
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
      // "original" definition of GENERALIZEDINDEX from Mike Weinberger
      //   int generalizedIndex=id.ietaAbs()+10000*id.iphi()+
      //       ((id.ieta()<0)?(0):(100));
      _cfg.generalizedindex =
	_cfg.iphi*10000+
	(row->ieta>0)*100+abs(row->ieta);
      _cfg.lut = _set.lut[lut_index];
      if (split_by_crate ){
	_xml[row->crate]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
      else{
	_xml[0]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
    }
  }
  cout << "LUTs generated: " << _counter.getCount() << endl;
  cout << "Generating compression (output) LUTs from ascii master file...DONE" << endl;
  return _xml;
}


std::map<int, shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromCoder( const HcalTPGCoder & _coder, string _tag, bool split_by_crate )
{
  cout << "Generating linearization (input) LUTs from HcaluLUTTPGCoder..." << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //std::vector<EMap::EMapRow> & _map = _emap.get_map();
  //cout << "EMap contains " << _map . size() << " entries" << endl;

  LMap _lmap;
  _lmap . read( "backup/HCALmapHBEF.txt", "HBEF" );
  // HO is not part of trigger, so TPGCoder cannot generate LUTs for it
  //_lmap . read( "backup/HCALmapHO.txt", "HO" );
  map<int,LMapRow> & _map = _lmap.get_map();
  cout << "LMap contains " << _map . size() << " channels" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  //HcalLutSet _set = getLinearizationLutSetFromCoder();
  //int lut_set_size = _set.lut.size(); // number of different luts

  //loop over all HCAL channels
  RooGKCounter _counter;
  for( map<int,LMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;
    
    if ( _xml.count(row->second.crate) == 0 && split_by_crate ){
      _xml.insert( pair<int,shared_ptr<LutXml> >(row->second.crate,shared_ptr<LutXml>(new LutXml())) );
    }
    else if ( _xml.count(0) == 0 && !split_by_crate ){
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
    // FIXME: probably fixed. fiber==htr_fi, not rm_fi in LMAP notation.
    //_cfg.fiber = row->second.rm_fi;
    _cfg.fiber = row->second.htr_fi;
    _cfg.fiberchan = row->second.fi_ch;
    _cfg.lut_type = 1;
    _cfg.creationtag = _tag;
    _cfg.creationstamp = get_time_stamp( time(0) );
    _cfg.targetfirmware = "1.0.0";
    _cfg.formatrevision = "1"; //???
    // "original" definition of GENERALIZEDINDEX from Mike Weinberger
    //    int generalizedIndex=id.ietaAbs()+1000*id.depth()+10000*id.iphi()+
    //        ((id.ieta()<0)?(0):(100))+((id.subdet()==HcalForward && id.ietaAbs()==29)?(4*10000):(0));
    _cfg.generalizedindex =
      _cfg.iphi*10000 + _cfg.depth*1000 +
      (row->second.side>0)*100 + row->second.eta +
      ((row->second.det==HcalForward && row->second.eta==29)?(4*10000):(0));

    //HcalDetId _detid(row->first);
    HcalDetId _detid(row->second.det, row->second.side*row->second.eta, row->second.phi, row->second.depth);
    //cout << "### DEBUG: rawid = " << _detid.rawId() << endl;    

    //cout << "### DEBUG: subdetector = " << row->second.det << endl;    
    std::vector<unsigned short>  coder_lut = _coder . getLinearizationLUT(_detid);
    for (std::vector<unsigned short>::const_iterator _i=coder_lut.begin(); _i!=coder_lut.end();_i++){
      unsigned int _temp = (unsigned int)(*_i);
      //if (_temp!=0) cout << "DEBUG non-zero LUT!!!!!!!!!!!!!!!" << (*_i) << "     " << _temp << endl;
      //unsigned int _temp = 0;
      _cfg.lut.push_back(_temp);
    }
    if (split_by_crate ){
      _xml[row->second.crate]->addLut( _cfg, lut_checksums_xml );  
      _counter.count();
    }
    else{
      _xml[0]->addLut( _cfg, lut_checksums_xml );  
      _counter.count();
    }
  }
  cout << "Generated LUTs: " << _counter.getCount() << endl;
  cout << "Generating linearization (input) LUTs from HcaluLUTTPGCoder...DONE" << endl;
  return _xml;
}




std::map<int, shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromCoderEmap( const HcalTPGCoder & _coder, string _tag, bool split_by_crate )
{
  cout << "Generating linearization (input) LUTs from HcaluLUTTPGCoder..." << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  cout << "EMap contains " << _map . size() << " entries" << endl;

  RooGKCounter _counter;
  //loop over all EMap channels
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    if( (row->subdet.find("HB")!=string::npos ||
	 row->subdet.find("HE")!=string::npos ||
	 row->subdet.find("HF")!=string::npos ) &&
	row->subdet.size()==2
       ){
      LutXml::Config _cfg;
      
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(row->crate,shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
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
      _cfg.lut_type = 1;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp( time(0) );
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1"; //???
      // "original" definition of GENERALIZEDINDEX from Mike Weinberger
      //    int generalizedIndex=id.ietaAbs()+1000*id.depth()+10000*id.iphi()+
      //        ((id.ieta()<0)?(0):(100))+((id.subdet()==HcalForward && id.ietaAbs()==29)?(4*10000):(0));
      _cfg.generalizedindex =
	_cfg.iphi*10000 + _cfg.depth*1000 +
	(row->ieta>0)*100 + abs(row->ieta) +
	(((row->subdet.find("HF")!=string::npos) && abs(row->ieta)==29)?(4*10000):(0));
      HcalSubdetector _subdet;
      if ( row->subdet.find("HB")!=string::npos ) _subdet = HcalBarrel;
      else if ( row->subdet.find("HE")!=string::npos ) _subdet = HcalEndcap;
      else if ( row->subdet.find("HO")!=string::npos ) _subdet = HcalOuter;
      else if ( row->subdet.find("HF")!=string::npos ) _subdet = HcalForward;
      else _subdet = HcalOther;
      HcalDetId _detid(_subdet, row->ieta, row->iphi, row->idepth);
      //cout << "### DEBUG: rawid = " << _detid.rawId() << ", " << _subdet << endl;    
      //cout << "### DEBUG: subdetector = " << row->subdet << endl;    
      std::vector<unsigned short>  coder_lut = _coder . getLinearizationLUT(_detid);
      for (std::vector<unsigned short>::const_iterator _i=coder_lut.begin(); _i!=coder_lut.end();_i++){
	unsigned int _temp = (unsigned int)(*_i);
	//if (_temp!=0) cout << "DEBUG non-zero LUT!!!!!!!!!!!!!!!" << (*_i) << "     " << _temp << endl;
	//unsigned int _temp = 0;
	_cfg.lut.push_back(_temp);
      }
      if (split_by_crate ){
	_xml[row->crate]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
      else{
	_xml[0]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
    }
  }
  cout << "Generated LUTs: " << _counter.getCount() << endl;
  cout << "Generating linearization (input) LUTs from HcaluLUTTPGCoder...DONE" << endl;
  return _xml;
}



std::map<int, shared_ptr<LutXml> > HcalLutManager::getCompressionLutXmlFromCoder( const CaloTPGTranscoderULUT & _coder, string _tag, bool split_by_crate )
{
  cout << "Generating compression (output) LUTs from CaloTPGTranscoderULUT," << endl;
  cout << "initialized from Event Setup" << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);

  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  cout << "EMap contains " << _map . size() << " channels" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  //HcalLutSet _set = getLutSetFromFile( _filename, 2 );
  //int lut_set_size = _set.lut.size(); // number of different luts

  //loop over all EMap channels
  RooGKCounter _counter;
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;

    // only trigger tower channels
    // and valid (ieta,iphi)
    if ( row->subdet . find("HT") != string::npos && _coder.HTvalid(row->ieta, row->iphi) ){
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(row->crate,shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
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
      _cfg.lut_type = 2;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp( time(0) );
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1"; //???
      // "original" definition of GENERALIZEDINDEX from Mike Weinberger
      //   int generalizedIndex=id.ietaAbs()+10000*id.iphi()+
      //       ((id.ieta()<0)?(0):(100));
      _cfg.generalizedindex =
	_cfg.iphi*10000+
	(row->ieta>0)*100+abs(row->ieta);
      
      // FIXME: work around bug in emap v6: rawId wasn't filled
      //HcalTrigTowerDetId _detid(row->rawId);
      HcalTrigTowerDetId _detid(row->ieta, row->iphi);
      
      std::vector<unsigned char> coder_lut = _coder.getCompressionLUT(_detid);
      for (std::vector<unsigned char>::const_iterator _i=coder_lut.begin(); _i!=coder_lut.end();_i++){
	unsigned int _temp = (unsigned int)(*_i);
	//if (_temp!=0) cout << "DEBUG non-zero LUT!!!!!!!!!!!!!!!" << (*_i) << "     " << _temp << endl;
	//unsigned int _temp = 0;
	_cfg.lut.push_back(_temp);
      }
      //_cfg.lut = _set.lut[lut_index];
      
      if (split_by_crate ){
	_xml[row->crate]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
      else{
	_xml[0]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
    }
  }
  cout << "LUTs generated: " << _counter.getCount() << endl;
  cout << "Generating compression (output) LUTs from CaloTPGTranscoderULUT...DONE" << endl;
  return _xml;
}



std::map<int, shared_ptr<LutXml> > HcalLutManager::getCompressionLutXmlFromCoder( string _tag, bool split_by_crate )
{
  cout << "Generating compression (output) LUTs from CaloTPGTranscoderULUT" << endl;
  map<int, shared_ptr<LutXml> > _xml; // index - crate number

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v5_080208.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);

  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  cout << "EMap contains " << _map . size() << " channels" << endl;

  // read LUTs and their eta/phi/depth/subdet ranges
  //HcalLutSet _set = getLutSetFromFile( _filename, 2 );
  //int lut_set_size = _set.lut.size(); // number of different luts

  CaloTPGTranscoderULUT _coder;

  //loop over all EMap channels
  RooGKCounter _counter;
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;

    // only trigger tower channels
    // and valid (ieta,iphi)
    if ( row->subdet . find("HT") != string::npos && _coder.HTvalid(row->ieta, row->iphi) ){
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( pair<int,shared_ptr<LutXml> >(row->crate,shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
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
      _cfg.lut_type = 2;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp( time(0) );
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1"; //???
      // "original" definition of GENERALIZEDINDEX from Mike Weinberger
      //   int generalizedIndex=id.ietaAbs()+10000*id.iphi()+
      //       ((id.ieta()<0)?(0):(100));
      _cfg.generalizedindex =
	_cfg.iphi*10000+
	(row->ieta>0)*100+abs(row->ieta);
      
      // FIXME: work around bug in emap v6: rawId wasn't filled
      //HcalTrigTowerDetId _detid(row->rawId);
      HcalTrigTowerDetId _detid(row->ieta, row->iphi);
      
      std::vector<unsigned char> coder_lut = _coder.getCompressionLUT(_detid);
      for (std::vector<unsigned char>::const_iterator _i=coder_lut.begin(); _i!=coder_lut.end();_i++){
	unsigned int _temp = (unsigned int)(*_i);
	//if (_temp!=0) cout << "DEBUG non-zero LUT!!!!!!!!!!!!!!!" << (*_i) << "     " << _temp << endl;
	//unsigned int _temp = 0;
	_cfg.lut.push_back(_temp);
      }
      //_cfg.lut = _set.lut[lut_index];
      
      if (split_by_crate ){
	_xml[row->crate]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
      else{
	_xml[0]->addLut( _cfg, lut_checksums_xml );  
	_counter.count();
      }
    }
  }
  cout << "LUTs generated: " << _counter.getCount() << endl;
  cout << "Generating compression (output) LUTs from CaloTPGTranscoderULUT...DONE" << endl;
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

int HcalLutManager::createLinLutXmlFiles( string _tag, string _lin_file, bool split_by_crate )
{
  //cout << "DEBUG1: split_by_crate = " << split_by_crate << endl;
  std::map<int, shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    addLutMap( xml, getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate ) );
  }
  writeLutXmlFiles( xml, _tag, split_by_crate );

  string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

int HcalLutManager::createAllLutXmlFiles( string _tag, string _lin_file, string _comp_file, bool split_by_crate )
{
  //cout << "DEBUG1: split_by_crate = " << split_by_crate << endl;
  std::map<int, shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    //addLutMap( xml, getLutXmlFromAsciiMaster( _lin_file, _tag, -1, split_by_crate ) );
    addLutMap( xml, getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate ) );
  }
  if ( _comp_file.size() != 0 ){
    //cout << "DEBUG1!!!!" << endl;
    addLutMap( xml, getCompressionLutXmlFromAsciiMaster( _comp_file, _tag, -1, split_by_crate ) );
    //cout << "DEBUG2!!!!" << endl;
  }
  writeLutXmlFiles( xml, _tag, split_by_crate );

  string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

int HcalLutManager::createCompLutXmlFilesFromCoder( string _tag, bool split_by_crate )
{
  //cout << "DEBUG1: split_by_crate = " << split_by_crate << endl;
  std::map<int, shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  addLutMap( xml, getCompressionLutXmlFromCoder( _tag, split_by_crate ) );

  writeLutXmlFiles( xml, _tag, split_by_crate );

  string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

int HcalLutManager::createAllLutXmlFilesFromCoder( const HcalTPGCoder & _coder, string _tag, bool split_by_crate )
{
  //cout << "DEBUG1: split_by_crate = " << split_by_crate << endl;
  std::map<int, shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  //addLutMap( xml, getLinearizationLutXmlFromCoder( _coder, _tag, split_by_crate ) );
  addLutMap( xml, getLinearizationLutXmlFromCoderEmap( _coder, _tag, split_by_crate ) );
  addLutMap( xml, getCompressionLutXmlFromCoder( _tag, split_by_crate ) );

  writeLutXmlFiles( xml, _tag, split_by_crate );

  string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

//
//_____ use this for creating a full set of LUTs ________________________
//
int HcalLutManager::createLutXmlFiles_HBEFFromCoder_HOFromAscii( string _tag, const HcalTPGCoder & _coder, const CaloTPGTranscoderULUT & _transcoder, string _lin_file, bool split_by_crate )
{
  std::map<int, shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    const std::map<int, shared_ptr<LutXml> > _lin_lut_ascii_xml = getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate );
    addLutMap( xml, _lin_lut_ascii_xml );
  }
  const std::map<int, shared_ptr<LutXml> > _lin_lut_xml = getLinearizationLutXmlFromCoderEmap( _coder, _tag, split_by_crate );
  addLutMap( xml, _lin_lut_xml );
  //
  const std::map<int, shared_ptr<LutXml> > _comp_lut_xml = getCompressionLutXmlFromCoder( _transcoder, _tag, split_by_crate );
  addLutMap( xml, _comp_lut_xml );
  
  writeLutXmlFiles( xml, _tag, split_by_crate );
  
  string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );
  
  return 0;
}


int HcalLutManager::createLutXmlFiles_HBEFFromCoder_HOFromAscii( string _tag, const HcalTPGCoder & _coder, string _lin_file, bool split_by_crate )
{
  std::map<int, shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    const std::map<int, shared_ptr<LutXml> > _lin_lut_ascii_xml = getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate );
    addLutMap( xml, _lin_lut_ascii_xml );
  }
  const std::map<int, shared_ptr<LutXml> > _lin_lut_xml = getLinearizationLutXmlFromCoderEmap( _coder, _tag, split_by_crate );
  addLutMap( xml, _lin_lut_xml );
  //
  const std::map<int, shared_ptr<LutXml> > _comp_lut_xml = getCompressionLutXmlFromCoder( _tag, split_by_crate );
  addLutMap( xml, _comp_lut_xml );
  
  writeLutXmlFiles( xml, _tag, split_by_crate );
  
  string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );
  
  return 0;
}


// use this to create HBEF only from coders (physics LUTs)
int HcalLutManager::createAllLutXmlFilesLinAsciiCompCoder( string _tag, string _lin_file, bool split_by_crate )
{
  //cout << "DEBUG1: split_by_crate = " << split_by_crate << endl;
  std::map<int, shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    addLutMap( xml, getLutXmlFromAsciiMaster( _lin_file, _tag, -1, split_by_crate ) );
  }
  addLutMap( xml, getCompressionLutXmlFromCoder( _tag, split_by_crate ) );
  writeLutXmlFiles( xml, _tag, split_by_crate );

  string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}



void HcalLutManager::addLutMap(std::map<int, shared_ptr<LutXml> > & result, const std::map<int, shared_ptr<LutXml> > & other)
{
  for ( std::map<int, shared_ptr<LutXml> >::const_iterator lut=other.begin(); lut!=other.end(); lut++ ){
    cout << "Added LUTs for crate " << lut->first << endl;
    if ( result.count(lut->first)==0 ){
      result . insert( *lut );
    }
    else{
      *(result[lut->first]) += *(lut->second);
    }
  }
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
  local_connect( _filename, "backup/HCALmapHBEF.txt", "backup/HCALmapHO.txt" );

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  int map_size = _map . size();
  cout << "EMap contains " << map_size << " channels" << endl;

  // make sure that all init is done
  vector<unsigned int> _lut;
  _lut = getLutFromXml( _tag, 1107313727, hcal::ConfigurationDatabase::LinearizerLUT );


  cout << endl << "Testing direct parsing of the LUT XML" << endl;
  struct timeval _t;
  gettimeofday( &_t, NULL );
  double _time =(double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0;
  test_direct_xml_parsing(_filename);
  gettimeofday( &_t, NULL );
  cout << "parsing took that much time: " << (double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0 - _time << endl;


  gettimeofday( &_t, NULL );
  _time =(double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0;
  cout << "before loop over random LUTs: " << _time << endl;
  int _raw_id;

  // loop over random LUTs
  for (int _iter=0; _iter<100; _iter++){
    gettimeofday( &_t, NULL );
    //cout << "before getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << endl;

    // select valid random emap channel
    while(1){
      int _key = (rand() % map_size);
      //_key = 3356;
      if( (_map[_key].subdet.find("HB")!=string::npos ||
	   _map[_key].subdet.find("HE")!=string::npos ||
	   _map[_key].subdet.find("HO")!=string::npos ||
	   _map[_key].subdet.find("HF")!=string::npos ) &&
	  _map[_key].subdet.size()==2
	  ){
	HcalSubdetector _subdet;
	if ( _map[_key].subdet.find("HB")!=string::npos ) _subdet = HcalBarrel;
	else if ( _map[_key].subdet.find("HE")!=string::npos ) _subdet = HcalEndcap;
	else if ( _map[_key].subdet.find("HO")!=string::npos ) _subdet = HcalOuter;
	else if ( _map[_key].subdet.find("HF")!=string::npos ) _subdet = HcalForward;
	else _subdet = HcalOther;
	HcalDetId _detid(_subdet, _map[_key].ieta, _map[_key].iphi, _map[_key].idepth);
	_raw_id = _detid.rawId();
	break;
      }
    }
    _lut = getLutFromXml( _tag, _raw_id, hcal::ConfigurationDatabase::LinearizerLUT );
    
    gettimeofday( &_t, NULL );
  }
  double d_time = _t.tv_sec+_t.tv_usec/1000000.0 - _time;
  cout << "after the loop over random LUTs: " << _time+d_time << endl;  
  cout << "total time: " << d_time << endl;  
  
  cout << "LUT length = " << _lut . size() << endl;
  for ( vector<unsigned int>::const_iterator i = _lut . end() - 1; i != _lut . begin()-1; i-- )
    {
      cout << (i-_lut.begin()) << "     " << _lut[(i-_lut.begin())] << endl;
      break;
    }
  
  db -> disconnect();
  
  delete db;
  db = 0;
  
  return 0;
}



int HcalLutManager::read_lmap( string lmap_hbef_file, string lmap_ho_file )
{
  delete lmap;
  lmap = new LMap();
  lmap -> read( lmap_hbef_file, "HBEF" );
  lmap -> read( lmap_ho_file, "HO" );
  cout << "LMap contains " << lmap -> get_map() . size() << " channels (compare to 9072 of all HCAL channels)" << endl;
  return 0;
}



int HcalLutManager::read_luts( string lut_xml_file )
{
  delete db;
  db = new HCALConfigDB();
  db -> connect( lut_xml_file );
  return 0;
}





int HcalLutManager::local_connect( string lut_xml_file, string lmap_hbef_file, string lmap_ho_file )
{
  read_lmap( lmap_hbef_file, lmap_ho_file );
  read_luts( lut_xml_file );
  return 0;
}




std::vector<unsigned int> HcalLutManager::getLutFromXml( string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt )
{
  cout << "getLutFromXml (new version) is not implemented. Use getLutFromXml_old() for now" << endl;

  std::vector<unsigned int> result;



  return result;
}


// obsolete, use getLutFromXml() instead
std::vector<unsigned int> HcalLutManager::getLutFromXml_old( string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt )
{
  if ( !lmap ){
    cout << "HcalLutManager: cannot find LUT without LMAP, exiting..." << endl;
    exit(-1);
  }
  if ( !db ){
    cout << "HcalLutManager: cannot find LUT, no source (local XML file), exiting..." << endl;
    exit(-1);
  }

  std::vector<unsigned int> result;

  map<int,LMapRow> & _map = lmap -> get_map();
  //cout << "HcalLutManager: LMap contains " << _map . size() << " channels (out of 9072 total)" << endl;

  HcalDetId _id( _rawid );
    
  unsigned int _crate, _slot, _fiber, _channel;
  string _fpga;
  int topbottom, luttype;

  // FIXME: check validity of _rawid
  if ( _map . find(_rawid) != _map.end() ){
    _crate   = _map[_rawid] . crate;
    _slot    = _map[_rawid] . htr;
    _fiber   = _map[_rawid] . htr_fi;
    _channel = _map[_rawid] . fi_ch;
    _fpga    = _map[_rawid] . fpga;
    
    if ( _fpga . find("top") != string::npos ) topbottom = 1;
    else if ( _fpga . find("bot") != string::npos ) topbottom = 0;
    else{
      cout << "HcalLutManager: irregular LMAP fpga value... do not know what to do - exiting" << endl;
      exit(-1);
    }
    if ( _lt == hcal::ConfigurationDatabase::LinearizerLUT ) luttype = 1;
    else luttype = 2;
    
    result = db -> getOnlineLUT( tag, _crate, _slot, topbottom, _fiber, _channel, luttype );
  }
  
  return result;
}



int HcalLutManager::get_xml_files_from_db( std::string tag, const std::string db_accessor, bool split_by_crate )
{
  std::map<int, shared_ptr<LutXml> > lut_map = get_brickSet_from_oracle( tag, db_accessor );
  if (split_by_crate){
    writeLutXmlFiles( lut_map, tag, split_by_crate );
  }      
  else{
    LutXml result;
    for( std::map<int, shared_ptr<LutXml> >::const_iterator xml = lut_map.begin(); xml != lut_map.end(); xml++ ){
      result += *(xml->second);
    }
    stringstream out_file;
    out_file << tag << ".xml";
    result . write(out_file.str());    
  }

  return 0;
}

std::map<int, shared_ptr<LutXml> > HcalLutManager::get_brickSet_from_oracle( std::string tag, const std::string _accessor )
{
  HCALConfigDB * db = new HCALConfigDB();
  XMLProcessor::getInstance(); // initialize xerces-c engine
  //const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );
  oracle::occi::Connection * _connection = db -> getConnection();  

  cout << "Preparing to request the LUT CLOBs from the database..." << endl;

  //int crate = 0;
  
  //
  // _____ query is different for the old validation DB _________________
  //
  //std::string query = ("SELECT TRIG_PRIM_LOOKUPTBL_DATA_CLOB, CRATE FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_TRIG_LOOKUP_TABLES");
  std::string query = ("SELECT TRIG_PRIM_LOOKUPTBL_DATA_CLOB, CRATE FROM CMS_HCL_HCAL_COND.V_HCAL_TRIG_LOOKUP_TABLES");
  //query+=toolbox::toString(" WHERE TAG_NAME='%s' AND CRATE=%d", tag.c_str(), crate);
  query+=toolbox::toString(" WHERE TAG_NAME='%s'", tag.c_str() );

  std::string brick_set;

  std::map<int, shared_ptr<LutXml> > lut_map;

  try {
    //SELECT
    cout << "Executing the query..." << endl;
    Statement* stmt = _connection -> createStatement();
    ResultSet *rs = stmt->executeQuery(query.c_str());
    cout << "Executing the query... done" << endl;
    
    cout << "Processing the query results..." << endl;
    //RooGKCounter _lines;
    while (rs->next()) {
      //_lines.count();
      oracle::occi::Clob clob = rs->getClob (1);
      int crate = rs->getInt(2);
      if ( crate != -1 ){ // not a brick with checksums
	cout << "Getting LUTs for crate #" << crate << " out of the database...";
	brick_set = db -> clobToString(clob);
	/*
	// FIXME: DEBUG lut xml files from simple strings
	stringstream file_name;
	ofstream out_file;
	file_name << tag << "_" << crate << "_debug" << ".xml";
	out_file . open( file_name.str().c_str() );
	out_file << brick_set;
	out_file . close();
	*/
	const char * bs = brick_set . c_str();
	MemBufInputSource * lut_clob = new MemBufInputSource( (const XMLByte *)bs, strlen( bs ), "lut_clob", false );
	shared_ptr<LutXml> lut_xml = shared_ptr<LutXml>( new LutXml( *lut_clob ) );
	lut_map[crate] = lut_xml;
        cout << " done" << endl;
      }
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
    //cout << "Query line count: " << _lines.getCount() << endl;
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }

  //cout << lut_map.size() << endl;

  db -> disconnect();
  //delete db;
  return lut_map;
}


int HcalLutManager::create_lut_loader( string file_list, string _prefix, string tag_name, string comment, string version, int subversion )
{
  cout << "Generating XML loader for LUTs..." << endl;
  //cout << _prefix << "..." << tag_name << endl;

  XMLLUTLoader::loaderBaseConfig baseConf;
  XMLLUTLoader::lutDBConfig conf;
  XMLLUTLoader::checksumsDBConfig CSconf;

  baseConf . tag_name = tag_name;
  //baseConf . comment_description = tag_name;
  baseConf . comment_description = comment;
  baseConf . iov_begin = "1";
  baseConf . iov_end = "-1";

  conf . version = version;

  stringstream _subversion;
  _subversion << subversion;
  conf . subversion = _subversion.str();

  CSconf . version = conf . version;
  CSconf . subversion = conf . subversion;
  CSconf . trig_prim_lookuptbl_data_file = _prefix + "_checksums.xml.dat";
  CSconf . comment_description = tag_name;

  XMLLUTLoader doc( &baseConf );

  vector<int> crate_number;
  vector<string> file_name = HcalQIEManager::splitString(file_list);
  for (std::vector<string>::const_iterator _f = file_name.begin(); _f != file_name.end(); _f++){
    int crate_begin = _f->rfind("_");
    int crate_end = _f->rfind(".xml.dat");
    crate_number . push_back(getInt(_f->substr(crate_begin+1,crate_end-crate_begin-1)));
  }
  for ( vector<string>::const_iterator _file = file_name . begin(); _file != file_name . end(); _file++ )
    {
      conf . trig_prim_lookuptbl_data_file = *_file;
      //conf . trig_prim_lookuptbl_data_file += ".dat";
      conf . crate = crate_number[ _file - file_name . begin() ];
      
      char _buf[128];
      sprintf( _buf, "CRATE%.2d", conf . crate );
      string _namelabel;
      _namelabel . append( _buf );
      conf . name_label = _namelabel;
      doc . addLUT( &conf );
    }
  
  doc . addChecksums( &CSconf );
  //doc . write( _prefix + "_Loader.xml" );
  doc . write( tag_name + "_Loader.xml" );

  cout << "Generating XML loader for LUTs... done." << endl;

  return 0;
}

void HcalLutManager::test_emap( void ){
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v5_080208.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  cout << "EMap contains " << _map . size() << " channels" << endl;
  
  //loop over all EMap channels
  //RooGKCounter _c;
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    
    // only trigger tower channels
    if ( row->subdet . find("HT") != string::npos ){
      cout << " -----> Subdet = " << row->subdet << endl;
      
      if (abs(row->ieta)>28){
	//if (row->iphi == 71){
	cout << " ==> (ieta,iphi) = " << row->ieta << ",	" << row->iphi << endl;
      }
    }
  }
}





int HcalLutManager::test_direct_xml_parsing( string _filename ){
  /*
  XMLDOMBlock _xml(_filename);
  //DOMElement * data_set_elem = (DOMElement *)(document -> getElementsByTagName( XMLProcessor::_toXMLCh( "DATA_SET" ) ) -> item(0));  
  DOMNodeList * brick_list = _xml . getDocument() ->  getElementsByTagName( XMLProcessor::_toXMLCh( "CFGBrick" ));  

  double n_bricks = brick_list->getLength();
  cout << "amount of LUT bricks: " << n_bricks << endl;

  for (int iter=0; iter!=n_bricks; iter++){
    DOMElement * _brick = (DOMElement *)(brick_list->item(iter));
    
    DOMElement * _param = 0;
    // loop over brick parameters
    int par_iter = 0;
    while(1){
      _param = (DOMElement *)(_brick->getElementsByTagName(XMLProcessor::_toXMLCh("Parameter")));
      string _name = _param->getAttribute( XMLProcessor::_toXMLCh( "name" ) );
      if (_name.find("IETA")==string::npos) break;

      string _tag = "Parameter";
      cout << "### Parameter IETA = " << _xml.getTagValue( _tag, 0, _brick);
      par_iter++;
    }
  }
  */
  return 0;
}
