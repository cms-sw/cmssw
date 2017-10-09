#ifndef HcalLutManager_h
#define HcalLutManager_h

/**

   \class HcalLutManager
   \brief Various manipulations with trigger Lookup Tables
   \author Gena Kukartsev, Brown University, March 14, 2008

*/

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalAssistant.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelIterator.h"



class XMLDOMBlock;

class HcalLutSet{
 public:
  std::string label;
  std::vector<std::string> subdet;
  std::vector<int> eta_min, eta_max, phi_min, phi_max, depth_min, depth_max;
  std::vector<std::vector<unsigned int> > lut;
};



class HcalLutManager{
 public:
  
  HcalLutManager( );
  HcalLutManager(std::vector<HcalGenericDetId> & map);
  HcalLutManager(const HcalElectronicsMap * _emap,
		 const HcalChannelQuality * _cq = 0,
		 uint32_t _status_word_to_mask = 0x0000);

  HcalLutManager(const HcalDbService *conditions,
		 const HcalChannelQuality * _cq = 0,
		 uint32_t _status_word_to_mask = 0x0000);

  ~HcalLutManager( );

  void init( void );
  std::string & getLutXml( std::vector<unsigned int> & _lut );

  // crate=-1 stands for all crates
  // legacy - use old LMAP. Use the xxxEmap method instead
  std::map<int, boost::shared_ptr<LutXml> > getLutXmlFromAsciiMaster( std::string _filename,
							       std::string _tag,
							       int _crate = -1,
							       bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getLinearizationLutXmlFromAsciiMasterEmap( std::string _filename,
										std::string _tag,
										int _crate,
										bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getLinearizationLutXmlFromAsciiMasterEmap_new( std::string _filename,
										    std::string _tag,
										    int _crate,
										    bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getCompressionLutXmlFromAsciiMaster( std::string _filename,
									  std::string _tag,
									  int _crate = -1,
									  bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getLinearizationLutXmlFromCoder( const HcalTPGCoder & _coder,
								      std::string _tag,
								      bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getMasks(int var, std::string _tag, bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getLinearizationLutXmlFromCoderEmap( const HcalTPGCoder & _coder,
									  std::string _tag,
									  bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getCompressionLutXmlFromCoder( std::string _tag, bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getCompressionLutXmlFromCoder( const CaloTPGTranscoderULUT & _coder,
								    std::string _tag,
								    bool split_by_crate = true );

  std::map<int, boost::shared_ptr<LutXml> > getZdcLutXml( std::string _tag,
						   bool split_by_crate = true );

  // add two std::map<s with LUTs. Designed mainly for joining compression LUTs to linearization ones.
  void addLutMap(std::map<int,
		 boost::shared_ptr<LutXml> > & result,
		 const std::map<int,
		 boost::shared_ptr<LutXml> > & other);
  
  // read LUTs from ASCII master file. 
  HcalLutSet getLutSetFromFile( std::string _filename, int _type = 1 ); // _type = 1 - linearization, 2 - compression

  int writeLutXmlFiles( std::map<int, boost::shared_ptr<LutXml> > & _xml, std::string _tag = "default_tag", bool split_by_crate = true );

  int createLinLutXmlFiles( std::string _tag, std::string _lin_file, bool split_by_crate = true );
  int createCompLutXmlFilesFromCoder( std::string _tag, bool split_by_crate = true );
  int createAllLutXmlFiles( std::string _tag, std::string _lin_file, std::string _comp_file, bool split_by_crate = true );
  int createAllLutXmlFilesFromCoder( const HcalTPGCoder & _coder, std::string _tag, bool split_by_crate = true );
  int createLutXmlFiles_HBEFFromCoder_HOFromAscii( std::string _tag, const HcalTPGCoder & _coder, std::string _lin_file, bool split_by_crate = true );
  int createLutXmlFiles_HBEFFromCoder_HOFromAscii( std::string _tag, const HcalTPGCoder & _coder, const CaloTPGTranscoderULUT & _transcoder, std::string _lin_file, bool split_by_crate = true );

  int createLutXmlFiles_HBEFFromCoder_HOFromAscii_ZDC( std::string _tag, const HcalTPGCoder & _coder, const CaloTPGTranscoderULUT & _transcoder, std::string _lin_file, bool split_by_crate = true );

  int createAllLutXmlFilesLinAsciiCompCoder( std::string _tag, std::string _lin_file, bool split_by_crate = true );

  // tests
  //    reading LUTs from a local XML
  int test_xml_access( std::string _tag, std::string _filename );
  int test_direct_xml_parsing( std::string _filename );
  void test_emap(void);
  
  // connect to local XML file with LUTs and local ASCII file with LMAP
  // connection interface through protected members db and lmap
  int read_lmap( std::string lmap_hbef_file, std::string lmap_ho_file );
  int read_luts( std::string lut_xml_file );
  int local_connect( std::string lut_xml_file, std::string lmap_hbef_file, std::string lmap_ho_file );

  // hcal::ConfigurationDatabase::LinearizerLUT
  // hcal::ConfigurationDatabase::CompressionLUT
  std::vector<unsigned int> getLutFromXml_old( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt );
  std::vector<unsigned int> getLutFromXml( std::string tag, uint32_t _rawid, hcal::ConfigurationDatabase::LUTType _lt );

  std::map<int, boost::shared_ptr<LutXml> > get_brickSet_from_oracle( std::string tag, const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22" );

  int get_xml_files_from_db( std::string tag, const std::string db_accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22", bool split_by_crate = true );

  int create_lut_loader( std::string file_list, std::string _prefix, std::string tag_name, std::string comment="default comment", std::string version="V00-01-01", int subversion=1 );

  // get md5 checksums for LUTs
  std::string get_checksum( std::vector<unsigned int> & lut );

  static int getInt( std::string number );
  static HcalSubdetector get_subdetector( std::string _subdet );
  static std::string get_time_stamp( time_t _time );

  // gives the iterator a list of channels
  int initChannelIterator(std::vector<HcalGenericDetId> & map);

 protected:
  
  LutXml * lut_xml;
  XMLDOMBlock * lut_checksums_xml;
  HCALConfigDB * db;
  LMap * lmap;
  HcalChannelIterator _iter;
  HcalAssistant _ass;
  const HcalElectronicsMap * emap;
  const HcalChannelQuality * cq;
  const HcalDbService* conditions;
  uint32_t status_word_to_mask;
};


class HcalLutManager_test{
 public:
  
  static int getLutXml_test( std::vector<unsigned int> & _lut ){return 0;}

  static int getLutSetFromFile_test( std::string _filename );

  static int getInt_test( std::string number );

 protected:
  LutXml * lut_xml;
};

#endif
