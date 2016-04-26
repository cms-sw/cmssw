#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>  // For srand() and rand()

#ifdef HAVE_XDAQ
#include <toolbox/string.h>
#else
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Replaces toolbox::toString
#endif

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ZdcLut.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalQIEManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLLUTLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
XERCES_CPP_NAMESPACE_USE 
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


HcalLutManager::HcalLutManager(const HcalElectronicsMap * _emap,
			       const HcalChannelQuality * _cq,
			       uint32_t _status_word_to_mask)
{
  init();
  emap = _emap;
  cq   = _cq;
  status_word_to_mask = _status_word_to_mask;
}


void HcalLutManager::init( void )
{    
  lut_xml = 0;
  lut_checksums_xml = 0;
  db = 0;
  lmap = 0;
  emap = 0;
  cq   = 0;
  status_word_to_mask = 0x0000;
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


int HcalLutManager::getInt( std::string number )
{
  int result;
  sscanf(number.c_str(), "%d", &result);
  return result;
}

HcalSubdetector HcalLutManager::get_subdetector( std::string _det )
{
  HcalSubdetector result;
  if      ( _det.find("HB") != std::string::npos ) result = HcalBarrel;
  else if ( _det.find("HE") != std::string::npos ) result = HcalEndcap;
  else if ( _det.find("HF") != std::string::npos ) result = HcalForward;
  else if ( _det.find("HO") != std::string::npos ) result = HcalOuter;
  else                                        result = HcalOther;  

  return result;
}


int HcalLutManager_test::getLutSetFromFile_test( std::string _filename )
{
  HcalLutManager _manager;
  HcalLutSet _set = _manager . getLutSetFromFile( _filename );
  std::stringstream s;
  s << "===> Test of HcalLutSet HcalLutManager::getLutSetFromFile( std::string _filename )" << std::endl << std::endl;
  s << _set . label << std::endl;
  for (unsigned int i = 0; i != _set.subdet.size(); i++) s << _set.subdet[i] << "	";
  s << std::endl;
  for (unsigned int i = 0; i != _set.eta_min.size(); i++) s << _set.eta_min[i] << "	";
  s << std::endl;
  for (unsigned int i = 0; i != _set.eta_max.size(); i++) s << _set.eta_max[i] << "	";
  s << std::endl;
  for (unsigned int i = 0; i != _set.phi_min.size(); i++) s << _set.phi_min[i] << "	";
  s << std::endl;
  for (unsigned int i = 0; i != _set.phi_max.size(); i++) s << _set.phi_max[i] << "	";
  s << std::endl;
  for (unsigned int i = 0; i != _set.depth_min.size(); i++) s << _set.depth_min[i] << "	";
  s << std::endl;
  for (unsigned int i = 0; i != _set.depth_max.size(); i++) s << _set.depth_max[i] << "	";
  s << std::endl;
  for (unsigned int j = 0; j != _set.lut[0].size(); j++){
    for (unsigned int i = 0; i != _set.lut.size(); i++){
      s << _set.lut[i][j] << "	";
    }
    s << "---> " << j << std::endl;
  }
  edm::LogInfo("HcalLutManager") << s.str();
  return 0;
}


HcalLutSet HcalLutManager::getLutSetFromFile( std::string _filename, int _type )
{
  HcalLutSet _lutset;

  ifstream infile( _filename . c_str() );
  std::string buf;

  if ( infile . is_open() ){
    edm::LogInfo("HcalLutManager") << "File " << _filename << " is open..." << std::endl
      << "Reading LUTs and their eta/phi/depth/subdet ranges...";

    // get label
    getline( infile, _lutset . label );

    if ( _type == 1 ){ // for linearization LUTs get subdetectors (default)
      //get subdetectors
      getline( infile, buf );
      _lutset . subdet = HcalQIEManager::splitString( buf );
    }

    //get min etas
    std::vector<std::string> buf_vec;
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.eta_min.push_back(HcalLutManager::getInt(*iter));
    }

    //get max etas
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.eta_max.push_back(HcalLutManager::getInt(*iter));
    }

    //get min phis
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.phi_min.push_back(HcalLutManager::getInt(*iter));
    }

    //get max phis
    getline( infile, buf );
    buf_vec = HcalQIEManager::splitString( buf );
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
      _lutset.phi_max.push_back(HcalLutManager::getInt(*iter));
    }

    if ( _type == 1 ){ // for linearization LUTs get depth range (default)
      //get min depths
      getline( infile, buf );
      buf_vec = HcalQIEManager::splitString( buf );
      for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
	_lutset.depth_min.push_back(HcalLutManager::getInt(*iter));
      }
      
      //get max depths
      getline( infile, buf );
      buf_vec = HcalQIEManager::splitString( buf );
      for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++){
	_lutset.depth_max.push_back(HcalLutManager::getInt(*iter));
      }
    }

    bool first_lut_entry = true;
    while (getline( infile, buf )) {
      buf_vec = HcalQIEManager::splitString( buf );
      for (unsigned int i = 0; i < buf_vec.size(); i++){
	if (first_lut_entry){
	  std::vector<unsigned int> _l;
	  _lutset.lut.push_back(_l);
	}
	_lutset . lut[i] . push_back(HcalLutManager::getInt(buf_vec[i]));
      }
      first_lut_entry = false;
    }
  }

  edm::LogInfo("HcalLutManager") << "done.";

  return _lutset;
}



std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getLutXmlFromAsciiMaster( std::string _filename, std::string _tag, int _crate, bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating linearization (input) LUTs from ascii master file...";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  LMap _lmap;
  _lmap . read( "./backup/HCALmapHBEF.txt", "HBEF" );
  _lmap . read( "./backup/HCALmapHO.txt", "HO" );
  std::map<int,LMapRow> & _map = _lmap.get_map();
  edm::LogInfo("HcalLutManager") << "LMap contains " << _map . size() << " channels";

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename );
  int lut_set_size = _set.lut.size(); // number of different luts

  RooGKCounter _counter;
  //loop over all HCAL channels
  for( std::map<int,LMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
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
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->second.crate,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      _cfg.ieta = row->second.side*row->second.eta;
      _cfg.iphi = row->second.phi;
      _cfg.depth = row->second.depth;
      _cfg.crate = row->second.crate;
      _cfg.slot = row->second.htr;
      if (row->second.fpga . find("top") != std::string::npos) _cfg.topbottom = 1;
      else if (row->second.fpga . find("bot") != std::string::npos) _cfg.topbottom = 0;
      else edm::LogWarning("HcalLutManager") << "fpga out of range...";
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
  edm::LogInfo("HcalLutManager") << "LUTs generated: " << _counter.getCount() << std::endl
    << "Generating linearization (input) LUTs from ascii master file...DONE" << std::endl;
  return _xml;
}

//
//_____ get HO from ASCII master here ___________________________________
//
std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromAsciiMasterEmap( std::string _filename, std::string _tag, int _crate, bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating linearization (input) LUTs from ascii master file...";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map . size() << " entries";

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename );
  int lut_set_size = _set.lut.size(); // number of different luts
  edm::LogInfo("HcalLutManager") << "  ==> " << lut_set_size << " sets of different LUTs read from the master file";

  // setup "zero" LUT for channel masking
  std::vector<unsigned int> zeroLut;
  for (size_t adc = 0; adc < 128; adc++) zeroLut.push_back(0);

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
	  _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->crate,boost::shared_ptr<LutXml>(new LutXml())) );
	}
	else if ( _xml.count(0) == 0 && !split_by_crate ){
	  _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
	}
	_cfg.ieta = row->ieta;
	_cfg.iphi = row->iphi;
	_cfg.depth = row->idepth;
	_cfg.crate = row->crate;
	_cfg.slot = row->slot;
	if (row->topbottom . find("t") != std::string::npos) _cfg.topbottom = 1;
	else if (row->topbottom . find("b") != std::string::npos) _cfg.topbottom = 0;
	else edm::LogWarning("HcalLutManager") << "fpga out of range...";
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
	//
	// consider channel status here
	DetId _detId(row->rawId);
	uint32_t status_word = cq->getValues(_detId)->getValue();
	if ((status_word & status_word_to_mask) > 0){
	  _cfg.lut = zeroLut;
	}
	else{
	  _cfg.lut = _set.lut[lut_index];
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
  }
  edm::LogInfo("HcalLutManager") << "LUTs generated: " << _counter.getCount() << std::endl
    << "Generating linearization (input) LUTs from ascii master file...DONE" << std::endl;
  return _xml;
}


std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromAsciiMasterEmap_new( std::string _filename, std::string _tag, int _crate, bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating linearization (input) LUTs from ascii master file...";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename );
  int lut_set_size = _set.lut.size(); // number of different luts
  edm::LogInfo("HcalLutManager") << "  ==> " << lut_set_size << " sets of different LUTs read from the master file";

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

      // FIXME: this is probably wrong, raw ids are different
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
	  _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(aCrate,boost::shared_ptr<LutXml>(new LutXml())) );
	}
	else if ( _xml.count(0) == 0 && !split_by_crate ){
	  _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
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
  edm::LogInfo("HcalLutManager") << "LUTs generated: " << _counter.getCount() << std::endl
    << "Generating linearization (input) LUTs from ascii master file...DONE" << std::endl;
  return _xml;
}


std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getCompressionLutXmlFromAsciiMaster( std::string _filename, std::string _tag, int _crate, bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating compression (output) LUTs from ascii master file...";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  edm::LogInfo("HcalLutManager") << "instantiating CaloTPGTranscoderULUT in order to check the validity of (ieta,iphi)...";
  CaloTPGTranscoderULUT _coder;

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map . size() << " channels";

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile( _filename, 2 );
  int lut_set_size = _set.lut.size(); // number of different luts
  edm::LogInfo("HcalLutManager") << "  ==> " << lut_set_size << " sets of different LUTs read from the master file";

  //loop over all EMap channels
  RooGKCounter _counter;
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;

    // search for the correct LUT for a given channel,
    // higher LUT numbers have priority in case of overlapping
    int lut_index=-1;
    for ( int i=0; i<lut_set_size; i++ ){
      if ( row->subdet . find("HT") != std::string::npos &&
	   (row->crate == _crate || _crate == -1) && // -1 stands for all crates
	   _set.eta_min[i] <= row->ieta &&
	   _set.eta_max[i] >= row->ieta &&
	   _set.phi_min[i] <= row->iphi &&
	   _set.phi_max[i] >= row->iphi &&
	   _coder.HTvalid(row->ieta, row->iphi, row->idepth / 10) ){
	lut_index=i;
      }
    }
    if ( lut_index >= 0 ){
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->crate,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      _cfg.ieta = row->ieta;
      _cfg.iphi = row->iphi;
      _cfg.depth = row->idepth;
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom . find("t") != std::string::npos) _cfg.topbottom = 1;
      else if (row->topbottom . find("b") != std::string::npos) _cfg.topbottom = 0;
      else edm::LogWarning("HcalLutManager") << "fpga out of range...";
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
  edm::LogInfo("HcalLutManager") << "LUTs generated: " << _counter.getCount() << std::endl
    << "Generating compression (output) LUTs from ascii master file...DONE" << std::endl;
  return _xml;
}


std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromCoder( const HcalTPGCoder & _coder, std::string _tag, bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating linearization (input) LUTs from HcaluLUTTPGCoder...";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //std::vector<EMap::EMapRow> & _map = _emap.get_map();
  //std::cout << "EMap contains " << _map . size() << " entries" << std::endl;

  LMap _lmap;
  _lmap . read( "backup/HCALmapHBEF.txt", "HBEF" );
  // HO is not part of trigger, so TPGCoder cannot generate LUTs for it
  //_lmap . read( "backup/HCALmapHO.txt", "HO" );
  std::map<int,LMapRow> & _map = _lmap.get_map();
  edm::LogInfo("HcalLutManager") << "LMap contains " << _map . size() << " channels";

  // read LUTs and their eta/phi/depth/subdet ranges
  //HcalLutSet _set = getLinearizationLutSetFromCoder();
  //int lut_set_size = _set.lut.size(); // number of different luts

  //loop over all HCAL channels
  RooGKCounter _counter;
  for( std::map<int,LMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;
    
    if ( _xml.count(row->second.crate) == 0 && split_by_crate ){
      _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->second.crate,boost::shared_ptr<LutXml>(new LutXml())) );
    }
    else if ( _xml.count(0) == 0 && !split_by_crate ){
      _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
    }
    _cfg.ieta = row->second.side*row->second.eta;
    _cfg.iphi = row->second.phi;
    _cfg.depth = row->second.depth;
    _cfg.crate = row->second.crate;
    _cfg.slot = row->second.htr;
    if (row->second.fpga . find("top") != std::string::npos) _cfg.topbottom = 1;
    else if (row->second.fpga . find("bot") != std::string::npos) _cfg.topbottom = 0;
    else edm::LogWarning("HcalLutManager") << "fpga out of range...";
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
    //std::cout << "### DEBUG: rawid = " << _detid.rawId() << std::endl;    

    //std::cout << "### DEBUG: subdetector = " << row->second.det << std::endl;    
    std::vector<unsigned short>  coder_lut = _coder . getLinearizationLUT(_detid);
    for (std::vector<unsigned short>::const_iterator _i=coder_lut.begin(); _i!=coder_lut.end();_i++){
      unsigned int _temp = (unsigned int)(*_i);
      //if (_temp!=0) std::cout << "DEBUG non-zero LUT!!!!!!!!!!!!!!!" << (*_i) << "     " << _temp << std::endl;
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
  edm::LogInfo("HcalLutManager") << "Generated LUTs: " << _counter.getCount() << std::endl
    << "Generating linearization (input) LUTs from HcaluLUTTPGCoder...DONE" << std::endl;
  return _xml;
}




std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getLinearizationLutXmlFromCoderEmap( const HcalTPGCoder & _coder, std::string _tag, bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating linearization (input) LUTs from HcaluLUTTPGCoder...";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map . size() << " entries";

  std::vector<unsigned int> zeroLut;
  for (size_t adc = 0; adc < 128; adc++) zeroLut.push_back(0);

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
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->crate,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      _cfg.ieta = row->ieta;
      _cfg.iphi = row->iphi;
      _cfg.depth = row->idepth;
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom . find("t") != std::string::npos) _cfg.topbottom = 1;
      else if (row->topbottom . find("b") != std::string::npos) _cfg.topbottom = 0;
      else edm::LogWarning("HcalLutManager") << "fpga out of range...";
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
      //
      // consider channel status here
      uint32_t status_word = cq->getValues(_detid)->getValue();
      if ((status_word & status_word_to_mask) > 0){
	_cfg.lut = zeroLut;
      }
      else{
	std::vector<unsigned short>  coder_lut = _coder . getLinearizationLUT(_detid);
	for (std::vector<unsigned short>::const_iterator _i=coder_lut.begin(); _i!=coder_lut.end();_i++){
	  unsigned int _temp = (unsigned int)(*_i);
	  _cfg.lut.push_back(_temp);
	}
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
  edm::LogInfo("HcalLutManager") << "Generated LUTs: " << _counter.getCount() << std::endl
    << "Generating linearization (input) LUTs from HcaluLUTTPGCoder...DONE" << std::endl;
  return _xml;
}



std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getCompressionLutXmlFromCoder( const CaloTPGTranscoderULUT & _coder, std::string _tag, bool split_by_crate )
{
    edm::LogInfo("HcalLutManager") << "Generating compression (output) LUTs from CaloTPGTranscoderULUT," << std::endl
        << "initialized from Event Setup" << std::endl;
    std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

    EMap _emap(emap);

    std::vector<EMap::EMapRow> & _map = _emap.get_map();
    edm::LogInfo("HcalLutManager") << "EMap contains " << _map . size() << " channels";

    RooGKCounter _counter;
    for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
	LutXml::Config _cfg;

	if ( row->subdet.find("HT") == std::string::npos) continue;

	HcalTrigTowerDetId _detid(row->rawId);

	if(!cq->topo()->validHT(_detid)) continue;


	if ( _xml.count(row->crate) == 0 && split_by_crate ){
	    _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->crate,boost::shared_ptr<LutXml>(new LutXml())) );
	}
	else if ( _xml.count(0) == 0 && !split_by_crate ){
	    _xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
	}

	_cfg.ieta = row->ieta;
	_cfg.iphi = row->iphi;
	_cfg.depth = row->idepth;
	_cfg.crate = row->crate;
	_cfg.slot = row->slot;
	if (row->topbottom . find("t") != std::string::npos) _cfg.topbottom = 1;
	else if (row->topbottom . find("b") != std::string::npos) _cfg.topbottom = 0;
	else edm::LogWarning("HcalLutManager") << "fpga out of range...";
	_cfg.fiber = row->fiber;
	_cfg.fiberchan = row->fiberchan;
	_cfg.lut_type = 2;
	_cfg.creationtag = _tag;
	_cfg.creationstamp = get_time_stamp( time(0) );
	_cfg.targetfirmware = "1.0.0";
	_cfg.formatrevision = "1"; //???
	_cfg.generalizedindex =_cfg.iphi*10000+ (row->ieta>0)*100+abs(row->ieta); //is this used for anything?

	_cfg.lut = _coder.getCompressionLUT(_detid);
      
      
	if (split_by_crate ){
	    _xml[row->crate]->addLut( _cfg, lut_checksums_xml );  
	    _counter.count();
	}
	else{
	    _xml[0]->addLut( _cfg, lut_checksums_xml );  
	    _counter.count();
	}
    }
    edm::LogInfo("HcalLutManager") << "LUTs generated: " << _counter.getCount() << std::endl
      << "Generating compression (output) LUTs from CaloTPGTranscoderULUT...DONE" << std::endl;
    return _xml;
}



std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getCompressionLutXmlFromCoder( std::string _tag, bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating compression (output) LUTs from CaloTPGTranscoderULUT";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v5_080208.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);

  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map . size() << " channels";

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
    const int tp_version = row->idepth / 10;
    if ( row->subdet . find("HT") != std::string::npos && _coder.HTvalid(row->ieta, row->iphi, tp_version) ){
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->crate,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      _cfg.ieta = row->ieta;
      _cfg.iphi = row->iphi;
      _cfg.depth = row->idepth;
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom . find("t") != std::string::npos) _cfg.topbottom = 1;
      else if (row->topbottom . find("b") != std::string::npos) _cfg.topbottom = 0;
      else edm::LogWarning("HcalLutManager") << "fpga out of range...";
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
      
      _cfg.lut = _coder.getCompressionLUT(_detid);
      
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
  edm::LogInfo("HcalLutManager") << "LUTs generated: " << _counter.getCount() << std::endl
    << "Generating compression (output) LUTs from CaloTPGTranscoderULUT...DONE" << std::endl;
  return _xml;
}



int HcalLutManager::writeLutXmlFiles( std::map<int, boost::shared_ptr<LutXml> > & _xml, std::string _tag, bool split_by_crate )
{
  for (std::map<int,boost::shared_ptr<LutXml> >::const_iterator cr = _xml.begin(); cr != _xml.end(); cr++){
    std::stringstream output_file_name;
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

int HcalLutManager::createLinLutXmlFiles( std::string _tag, std::string _lin_file, bool split_by_crate )
{
  //std::cout << "DEBUG1: split_by_crate = " << split_by_crate << std::endl;
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    addLutMap( xml, getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate ) );
  }
  writeLutXmlFiles( xml, _tag, split_by_crate );

  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

int HcalLutManager::createAllLutXmlFiles( std::string _tag, std::string _lin_file, std::string _comp_file, bool split_by_crate )
{
  //std::cout << "DEBUG1: split_by_crate = " << split_by_crate << std::endl;
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    //addLutMap( xml, getLutXmlFromAsciiMaster( _lin_file, _tag, -1, split_by_crate ) );
    addLutMap( xml, getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate ) );
  }
  if ( _comp_file.size() != 0 ){
    //std::cout << "DEBUG1!!!!" << std::endl;
    addLutMap( xml, getCompressionLutXmlFromAsciiMaster( _comp_file, _tag, -1, split_by_crate ) );
    //std::cout << "DEBUG2!!!!" << std::endl;
  }
  writeLutXmlFiles( xml, _tag, split_by_crate );

  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

int HcalLutManager::createCompLutXmlFilesFromCoder( std::string _tag, bool split_by_crate )
{
  //std::cout << "DEBUG1: split_by_crate = " << split_by_crate << std::endl;
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  addLutMap( xml, getCompressionLutXmlFromCoder( _tag, split_by_crate ) );

  writeLutXmlFiles( xml, _tag, split_by_crate );

  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

int HcalLutManager::createAllLutXmlFilesFromCoder( const HcalTPGCoder & _coder, std::string _tag, bool split_by_crate )
{
  //std::cout << "DEBUG1: split_by_crate = " << split_by_crate << std::endl;
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  //addLutMap( xml, getLinearizationLutXmlFromCoder( _coder, _tag, split_by_crate ) );
  addLutMap( xml, getLinearizationLutXmlFromCoderEmap( _coder, _tag, split_by_crate ) );
  addLutMap( xml, getCompressionLutXmlFromCoder( _tag, split_by_crate ) );

  writeLutXmlFiles( xml, _tag, split_by_crate );

  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}

//
//_____ use this for creating a full set of LUTs ________________________
//
int HcalLutManager::createLutXmlFiles_HBEFFromCoder_HOFromAscii( std::string _tag, const HcalTPGCoder & _coder, const CaloTPGTranscoderULUT & _transcoder, std::string _lin_file, bool split_by_crate )
{
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    const std::map<int, boost::shared_ptr<LutXml> > _lin_lut_ascii_xml = getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate );
    addLutMap( xml, _lin_lut_ascii_xml );
  }
  const std::map<int, boost::shared_ptr<LutXml> > _lin_lut_xml = getLinearizationLutXmlFromCoderEmap( _coder, _tag, split_by_crate );
  addLutMap( xml, _lin_lut_xml );
  //
  const std::map<int, boost::shared_ptr<LutXml> > _comp_lut_xml = getCompressionLutXmlFromCoder( _transcoder, _tag, split_by_crate );
  addLutMap( xml, _comp_lut_xml );
  
  writeLutXmlFiles( xml, _tag, split_by_crate );
  
  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );
  
  return 0;
}


int HcalLutManager::createLutXmlFiles_HBEFFromCoder_HOFromAscii( std::string _tag, const HcalTPGCoder & _coder, std::string _lin_file, bool split_by_crate )
{
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    const std::map<int, boost::shared_ptr<LutXml> > _lin_lut_ascii_xml = getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate );
    addLutMap( xml, _lin_lut_ascii_xml );
  }
  const std::map<int, boost::shared_ptr<LutXml> > _lin_lut_xml = getLinearizationLutXmlFromCoderEmap( _coder, _tag, split_by_crate );
  addLutMap( xml, _lin_lut_xml );
  //
  const std::map<int, boost::shared_ptr<LutXml> > _comp_lut_xml = getCompressionLutXmlFromCoder( _tag, split_by_crate );
  addLutMap( xml, _comp_lut_xml );
  
  writeLutXmlFiles( xml, _tag, split_by_crate );
  
  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );
  
  return 0;
}


// use this to create HBEF only from coders (physics LUTs)
int HcalLutManager::createAllLutXmlFilesLinAsciiCompCoder( std::string _tag, std::string _lin_file, bool split_by_crate )
{
  //std::cout << "DEBUG1: split_by_crate = " << split_by_crate << std::endl;
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    addLutMap( xml, getLutXmlFromAsciiMaster( _lin_file, _tag, -1, split_by_crate ) );
  }
  addLutMap( xml, getCompressionLutXmlFromCoder( _tag, split_by_crate ) );
  writeLutXmlFiles( xml, _tag, split_by_crate );

  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );

  return 0;
}



void HcalLutManager::addLutMap(std::map<int, boost::shared_ptr<LutXml> > & result, const std::map<int, boost::shared_ptr<LutXml> > & other)
{
  for ( std::map<int, boost::shared_ptr<LutXml> >::const_iterator lut=other.begin(); lut!=other.end(); lut++ ){
    edm::LogInfo("HcalLutManager") << "Added LUTs for crate " << lut->first;
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
  std::string creationstamp = timebuf;

  return creationstamp;
}




int HcalLutManager::test_xml_access( std::string _tag, std::string _filename )
{
  local_connect( _filename, "backup/HCALmapHBEF.txt", "backup/HCALmapHO.txt" );

  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  int map_size = _map . size();
  edm::LogInfo("HcalLutManager") << "EMap contains " << map_size << " channels";

  // make sure that all init is done
  std::vector<unsigned int> _lut;
  _lut = getLutFromXml( _tag, 1107313727, hcal::ConfigurationDatabase::LinearizerLUT );


  edm::LogInfo("HcalLutManager") << "Testing direct parsing of the LUT XML";
  struct timeval _t;
  gettimeofday( &_t, NULL );
  double _time =(double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0;
  test_direct_xml_parsing(_filename);
  gettimeofday( &_t, NULL );
  edm::LogInfo("HcalLutManager") << "parsing took that much time: " << (double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0 - _time;


  gettimeofday( &_t, NULL );
  _time =(double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0;
  edm::LogInfo("HcalLutManager") << "before loop over random LUTs: " << _time;
  int _raw_id;

  // loop over random LUTs
  for (int _iter=0; _iter<100; _iter++){
    gettimeofday( &_t, NULL );
    //std::cout << "before getting a LUT: " << _t . tv_sec << "." << _t . tv_usec << std::endl;

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
  edm::LogInfo("HcalLutManager") << "after the loop over random LUTs: " << _time+d_time << std::endl
    << "total time: " << d_time << std::endl;  
  
  edm::LogInfo("HcalLutManager") << "LUT length = " << _lut . size();
  for ( std::vector<unsigned int>::const_iterator i = _lut . end() - 1; i != _lut . begin()-1; i-- )
    {
      edm::LogInfo("HcalLutManager") << (i-_lut.begin()) << "     " << _lut[(i-_lut.begin())];
      break;
    }
  
  db -> disconnect();
  
  delete db;
  db = 0;
  
  return 0;
}



int HcalLutManager::read_lmap( std::string lmap_hbef_file, std::string lmap_ho_file )
{
  delete lmap;
  lmap = new LMap();
  lmap -> read( lmap_hbef_file, "HBEF" );
  lmap -> read( lmap_ho_file, "HO" );
  edm::LogInfo("HcalLutManager") << "LMap contains " << lmap -> get_map() . size() << " channels (compare to 9072 of all HCAL channels)";
  return 0;
}



int HcalLutManager::read_luts( std::string lut_xml_file )
{
  delete db;
  db = new HCALConfigDB();
  db -> connect( lut_xml_file );
  return 0;
}





int HcalLutManager::local_connect( std::string lut_xml_file, std::string lmap_hbef_file, std::string lmap_ho_file )
{
  read_lmap( lmap_hbef_file, lmap_ho_file );
  read_luts( lut_xml_file );
  return 0;
}




std::vector<unsigned int> HcalLutManager::getLutFromXml( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt )
{
  edm::LogInfo("HcalLutManager") << "getLutFromXml (new version) is not implemented. Use getLutFromXml_old() for now";

  std::vector<unsigned int> result;



  return result;
}


// obsolete, use getLutFromXml() instead
std::vector<unsigned int> HcalLutManager::getLutFromXml_old( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt )
{
  if ( !lmap ){
    edm::LogError("HcalLutManager") << "Cannot find LUT without LMAP, exiting...";
    exit(-1);
  }
  if ( !db ){
    edm::LogError("HcalLutManager") << "Cannot find LUT, no source (local XML file), exiting...";
    exit(-1);
  }

  std::vector<unsigned int> result;

  std::map<int,LMapRow> & _map = lmap -> get_map();
  //std::cout << "HcalLutManager: LMap contains " << _map . size() << " channels (out of 9072 total)" << std::endl;

  HcalDetId _id( _rawid );
    
  unsigned int _crate, _slot, _fiber, _channel;
  std::string _fpga;
  int topbottom, luttype;

  // FIXME: check validity of _rawid
  if ( _map . find(_rawid) != _map.end() ){
    _crate   = _map[_rawid] . crate;
    _slot    = _map[_rawid] . htr;
    _fiber   = _map[_rawid] . htr_fi;
    _channel = _map[_rawid] . fi_ch;
    _fpga    = _map[_rawid] . fpga;
    
    if ( _fpga . find("top") != std::string::npos ) topbottom = 1;
    else if ( _fpga . find("bot") != std::string::npos ) topbottom = 0;
    else{
      edm::LogError("HcalLutManager") << "Irregular LMAP fpga value... do not know what to do - exiting";
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
  std::map<int, boost::shared_ptr<LutXml> > lut_map = get_brickSet_from_oracle( tag, db_accessor );
  if (split_by_crate){
    writeLutXmlFiles( lut_map, tag, split_by_crate );
  }      
  else{
    LutXml result;
    for( std::map<int, boost::shared_ptr<LutXml> >::const_iterator xml = lut_map.begin(); xml != lut_map.end(); xml++ ){
      result += *(xml->second);
    }
    std::stringstream out_file;
    out_file << tag << ".xml";
    result . write(out_file.str());    
  }

  return 0;
}

std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::get_brickSet_from_oracle( std::string tag, const std::string _accessor )
{
  HCALConfigDB * db = new HCALConfigDB();
  XMLProcessor::getInstance(); // initialize xerces-c engine
  //const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );
  oracle::occi::Connection * _connection = db -> getConnection();  

  edm::LogInfo("HcalLutManager") << "Preparing to request the LUT CLOBs from the database...";

  //int crate = 0;
  
  //
  // _____ query is different for the old validation DB _________________
  //
  //std::string query = ("SELECT TRIG_PRIM_LOOKUPTBL_DATA_CLOB, CRATE FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_TRIG_LOOKUP_TABLES");
  std::string query = ("SELECT TRIG_PRIM_LOOKUPTBL_DATA_CLOB, CRATE FROM CMS_HCL_HCAL_COND.V_HCAL_TRIG_LOOKUP_TABLES");
  //query+=toolbox::toString(" WHERE TAG_NAME='%s' AND CRATE=%d", tag.c_str(), crate);
  query+=toolbox::toString(" WHERE TAG_NAME='%s'", tag.c_str() );

  std::string brick_set;

  std::map<int, boost::shared_ptr<LutXml> > lut_map;

  try {
    //SELECT
    edm::LogInfo("HcalLutManager") << "Executing the query...";
    Statement* stmt = _connection -> createStatement();
    ResultSet *rs = stmt->executeQuery(query.c_str());
    edm::LogInfo("HcalLutManager") << "Executing the query... done";
    
    edm::LogInfo("HcalLutManager") << "Processing the query results...";
    //RooGKCounter _lines;
    while (rs->next()) {
      //_lines.count();
      oracle::occi::Clob clob = rs->getClob (1);
      int crate = rs->getInt(2);
      if ( crate != -1 ){ // not a brick with checksums
	edm::LogInfo("HcalLutManager") << "Getting LUTs for crate #" << crate << " out of the database...";
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
	boost::shared_ptr<LutXml> lut_xml = boost::shared_ptr<LutXml>( new LutXml( *lut_clob ) );
	lut_map[crate] = lut_xml;
	edm::LogInfo("HcalLutManager") << "done";
      }
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
    //std::cout << "Query line count: " << _lines.getCount() << std::endl;
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }

  //std::cout << lut_map.size() << std::endl;

  db -> disconnect();
  //delete db;
  return lut_map;
}


int HcalLutManager::create_lut_loader( std::string file_list, std::string _prefix, std::string tag_name, std::string comment, std::string version, int subversion )
{
  edm::LogInfo("HcalLutManager") << "Generating XML loader for LUTs...";
  //std::cout << _prefix << "..." << tag_name << std::endl;

  XMLLUTLoader::loaderBaseConfig baseConf;
  XMLLUTLoader::lutDBConfig conf;
  XMLLUTLoader::checksumsDBConfig CSconf;

  baseConf . tag_name = tag_name;
  //baseConf . comment_description = tag_name;
  baseConf . comment_description = comment;
  baseConf . iov_begin = "1";
  baseConf . iov_end = "-1";

  conf . version = version;

  std::stringstream _subversion;
  _subversion << subversion;
  conf . subversion = _subversion.str();

  CSconf . version = conf . version;
  CSconf . subversion = conf . subversion;
  CSconf . trig_prim_lookuptbl_data_file = _prefix + "_checksums.xml.dat";
  CSconf . comment_description = tag_name;

  XMLLUTLoader doc( &baseConf );

  std::vector<int> crate_number;
  std::vector<std::string> file_name = HcalQIEManager::splitString(file_list);
  for (std::vector<std::string>::const_iterator _f = file_name.begin(); _f != file_name.end(); _f++){
    int crate_begin = _f->rfind("_");
    int crate_end = _f->rfind(".xml.dat");
    crate_number . push_back(getInt(_f->substr(crate_begin+1,crate_end-crate_begin-1)));
  }
  //
  //_____ fix due to the new convention: version/subversion combo must be unique for every payload
  //
  char _buf[128];
  time_t _offset = time(NULL);
  sprintf( _buf, "%d", (uint32_t)_offset );
  conf.version.append(".");
  conf.version.append(_buf);
  CSconf.version = conf.version;
  //
  for ( std::vector<std::string>::const_iterator _file = file_name . begin(); _file != file_name . end(); _file++ )
    {
      conf . trig_prim_lookuptbl_data_file = *_file;
      //conf . trig_prim_lookuptbl_data_file += ".dat";
      conf . crate = crate_number[ _file - file_name . begin() ];
      //
      //_____ fix due to the new convention: version/subversion combo must be unique for every payload
      //
      sprintf( _buf, "%.2d", conf.crate );
      conf.subversion.clear();
      conf.subversion.append(_buf);
      sprintf( _buf, "CRATE%.2d", conf . crate );
      std::string _namelabel;
      _namelabel . append( _buf );
      conf . name_label = _namelabel;
      doc . addLUT( &conf );
    }
  
  doc . addChecksums( &CSconf );
  //doc . write( _prefix + "_Loader.xml" );
  doc . write( tag_name + "_Loader.xml" );

  edm::LogInfo("HcalLutManager") << "Generating XML loader for LUTs... done.";

  return 0;
}

void HcalLutManager::test_emap( void ){
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v5_080208.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.03_080817.txt");
  //EMap _emap("../../../CondFormats/HcalObjects/data/official_emap_v6.04_080905.txt");
  EMap _emap(emap);
  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  std::stringstream s;
  s << "EMap contains " << _map . size() << " channels" << std::endl;
  
  //loop over all EMap channels
  //RooGKCounter _c;
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    
    // only trigger tower channels
    if ( row->subdet . find("HT") != std::string::npos ){
      s << " -----> Subdet = " << row->subdet << std::endl;
      
      if (abs(row->ieta)>28){
	//if (row->iphi == 71){
	s << " ==> (ieta,iphi) = " << row->ieta << ",	" << row->iphi << std::endl;
      }
    }
  }
  edm::LogInfo("HcalLutManager") << s.str();
}





int HcalLutManager::test_direct_xml_parsing( std::string _filename ){
  /*
  XMLDOMBlock _xml(_filename);
  //DOMElement * data_set_elem = (DOMElement *)(document -> getElementsByTagName( XMLProcessor::_toXMLCh( "DATA_SET" ) ) -> item(0));  
  DOMNodeList * brick_list = _xml . getDocument() ->  getElementsByTagName( XMLProcessor::_toXMLCh( "CFGBrick" ));  

  double n_bricks = brick_list->getLength();
  std::cout << "amount of LUT bricks: " << n_bricks << std::endl;

  for (int iter=0; iter!=n_bricks; iter++){
    DOMElement * _brick = (DOMElement *)(brick_list->item(iter));
    
    DOMElement * _param = 0;
    // loop over brick parameters
    int par_iter = 0;
    while(1){
      _param = (DOMElement *)(_brick->getElementsByTagName(XMLProcessor::_toXMLCh("Parameter")));
      std::string _name = _param->getAttribute( XMLProcessor::_toXMLCh( "name" ) );
      if (_name.find("IETA")==string::npos) break;

      std::string _tag = "Parameter";
      std::cout << "### Parameter IETA = " << _xml.getTagValue( _tag, 0, _brick);
      par_iter++;
    }
  }
  */
  return 0;
}


//
//_____ attempt to include ZDC LUTs _____________________________________
//
int HcalLutManager::createLutXmlFiles_HBEFFromCoder_HOFromAscii_ZDC( std::string _tag, const HcalTPGCoder & _coder, const CaloTPGTranscoderULUT & _transcoder, std::string _lin_file, bool split_by_crate )
{
  std::map<int, boost::shared_ptr<LutXml> > xml;
  if ( !lut_checksums_xml ){
    lut_checksums_xml = new XMLDOMBlock( "CFGBrick", 1 );
  }
  
  if ( _lin_file.size() != 0 ){
    const std::map<int, boost::shared_ptr<LutXml> > _lin_lut_ascii_xml = getLinearizationLutXmlFromAsciiMasterEmap( _lin_file, _tag, -1, split_by_crate );
    addLutMap( xml, _lin_lut_ascii_xml );
  }
  const std::map<int, boost::shared_ptr<LutXml> > _lin_lut_xml = getLinearizationLutXmlFromCoderEmap( _coder, _tag, split_by_crate );
  addLutMap( xml, _lin_lut_xml );
  //
  const std::map<int, boost::shared_ptr<LutXml> > _comp_lut_xml = getCompressionLutXmlFromCoder( _transcoder, _tag, split_by_crate );
  addLutMap( xml, _comp_lut_xml );
  //
  const std::map<int, boost::shared_ptr<LutXml> > _zdc_lut_xml = getZdcLutXml( _tag, split_by_crate );
  addLutMap( xml, _zdc_lut_xml );
  
  writeLutXmlFiles( xml, _tag, split_by_crate );
  
  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml -> write( checksums_file . c_str() );
  
  return 0;
}


std::map<int, boost::shared_ptr<LutXml> > HcalLutManager::getZdcLutXml( std::string _tag,
								 bool split_by_crate )
{
  edm::LogInfo("HcalLutManager") << "Generating ZDC LUTs ...may the Force be with us...";
  std::map<int, boost::shared_ptr<LutXml> > _xml; // index - crate number

  EMap _emap(emap);

  ZdcLut zdc;

  std::vector<EMap::EMapRow> & _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map . size() << " channels";

  //loop over all EMap channels
  RooGKCounter _counter;
  for( std::vector<EMap::EMapRow>::const_iterator row=_map.begin(); row!=_map.end(); row++ ){
    LutXml::Config _cfg;

    // only ZDC channels
    if ( row->zdc_section . find("ZDC") != std::string::npos ){
      if ( _xml.count(row->crate) == 0 && split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(row->crate,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      else if ( _xml.count(0) == 0 && !split_by_crate ){
	_xml.insert( std::pair<int,boost::shared_ptr<LutXml> >(0,boost::shared_ptr<LutXml>(new LutXml())) );
      }
      //  FIXME: introduce proper tag names in ZDC bricks for logical channel info
      _cfg.ieta = row->zdc_channel; // int
      //_cfg.ieta = row->zdc_zside; // int
      //_cfg.iphi = row->zdc_section; // string
      _cfg.depth = row->idepth; // int
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom . find("t") != std::string::npos) _cfg.topbottom = 1;
      else if (row->topbottom . find("b") != std::string::npos) _cfg.topbottom = 0;
      else edm::LogWarning("HcalLutManager") << "fpga out of range...";
      _cfg.fiber = row->fiber;
      _cfg.fiberchan = row->fiberchan;
      _cfg.lut_type = 1;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp( time(0) );
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1"; //???
      _cfg.generalizedindex = 0;
      
      //HcalZDCDetId _detid(row->zdc_section, (row->zdc_zside>0), row->zdc_channel);
      
      std::vector<int> coder_lut = zdc.get_lut(row->zdc_section,
					       row->zdc_zside,
					       row->zdc_channel);
      edm::LogInfo("HcalLutManager") << "***DEBUG: ZDC lut size: " << coder_lut.size();
      if (coder_lut.size()!=0){
	for (std::vector<int>::const_iterator _i=coder_lut.begin(); _i!=coder_lut.end();_i++){
	  unsigned int _temp = (unsigned int)(*_i);
	  //if (_temp!=0) std::cout << "DEBUG non-zero LUT!!!!!!!!!!!!!!!" << (*_i) << "     " << _temp << std::endl;
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
      } //size of lut
    }
  }
  edm::LogInfo("HcalLutManager") << "LUTs generated: " << _counter.getCount() << std::endl
    << "Generating ZDC LUTs...DONE" << std::endl;

  return _xml;
}


