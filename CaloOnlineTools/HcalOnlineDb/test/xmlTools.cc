#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <fstream>
#include <iterator>
#include <boost/program_options.hpp>

//#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLLUTLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatternLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRZeroSuppressionLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMapLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLRBXPedestalsLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/DBlmapReader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/DBlmapWriter.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplOracle.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalHardwareXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalQIEManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalTriggerKey.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelDataXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelQualityXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelIterator.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ZdcLut.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalDbOmds.h"

#include "OnlineDB/Oracle/interface/Oracle.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalO2OManager.h"

#include <sys/time.h>  // gettimeofday

#ifdef HAVE_XDAQ
#include "toolbox/string.h"
#else
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Replaces toolbox::toString
#endif


using namespace std;
using namespace oracle::occi;
using namespace hcal;
namespace po = boost::program_options;

int createLUTLoader( std::string prefix_="", std::string tag_="" );
int createHTRPatternLoader( void );
int createLMap( void );
int createRBXLoader( std::string & type_, std::string & tag_, std::string & list_file, std::string & _comment, std::string & _version );
//int createRBXentries( std::string );

// deprecated - to be removed
//int createZSLoader( void );

int testocci( void );
int testDB( std::string _tag, std::string _filename );
int lmaptest( std::string _param );
int hardware( void );
int test_db_access( void );
std::vector <std::string> splitString (const std::string& fLine);
int createZSLoader2( std::string & tag, std::string & comment, std::string & zs2HB, std::string & zs2HE, std::string & zs2HO, std::string & zs2HF );
void test_lut_gen( void );

int main( int argc, char **argv )
{
  //std::cout << "Running xmlTools..." << std::endl;

  //
  //===> command line options parser using boost  
  //
  int crate, sub_version;
  std::string db_accessor, version_name;
  po::options_description general("General options");
  general.add_options()
    ("help", "produce help message")
    ("quicktest", "Quick feature testing")
    ("test-string", po::value<string>(), "print test string")
    ("test-lmap", po::value<string>(), "test logical map functionality")
    ("test-emap", po::value<string>(), "test electronic map functionality")
    ("test-qie", po::value<string>(), "test QIE procedure elements")
    ("test-lut-manager", po::value<string>(), "test LUT functionality")
    ("test-lut-xml-access", "test and benchmark LUT reading from local XML file")
    ("test-lut-xml-file-access", "test and benchmark LUT reading from local XML file")
    ("test-lut-checksum", "test LUT MD5 checksum calculation")
    ("tag-name", po::value<string>(), "tag name")
    ("prefix-name", po::value<string>(), "prefix for file names and such")
    ("comment-line", po::value<string>(), "comment for a database entry")
    ("version-name", po::value<string>(&version_name)->default_value("V00-01-01"), "version name")
    ("sub-version", po::value<int>(&sub_version)->default_value( 1 ), "sub-version number")
    ("file-list", po::value<string>(), "list of files for further processing")
    ("crate", po::value<int>(&crate)->default_value( -1 ), "crate number")
    ("lut-type", po::value<int>(&crate)->default_value( 1 ), "LUT type: 1 - linearization, 2 - compression")
    ("create-lin-lut-xml", "create XML file(s) input LUTs from ASCII master")
    ("create-lut-xml", "create XML file(s) with LUTs from ASCII master")
    ("create-lut-xml-from-coder", "create XML file(s) with LUTs from TPG coder")
    ("create-lut-xml-lin-ascii-comp-coder", "create XML file(s) with linearizer LUTs from ASCII master and compression LUTs from coder")
    ("create-lut-loader", "create XML database loader for LUTs, and zip everything ready for uploading to the DB")
    ("create-trigger-key", "create a trigger key entry")
    ("get-lut-xml-from-oracle", "Get LUTs from Oracle database")
    ("database-accessor", po::value<string>(&db_accessor)->default_value("occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22"), "Database accessor std::string")
    ("lin-lut-master-file", po::value<string>(), "Linearizer LUT ASCII master file name")
    ("comp-lut-master-file", po::value<string>(), "Compression LUT ASCII master file name")
    ("do-not-split-by-crate", "output LUTs as a single XML instead of making a separate file for each crate")
    ("input-file", po::value<string>(), "Input file name")
    ("output-file", po::value<string>(), "Outputput file name")
    ("old-qie-file", po::value<string>(), "Old QIE table ASCII file")
    ("qie", "Generate new QIE table file")
    ("hf-qie", "Retrieve HF QIE ADC caps offsets and slopes")
    ("test-channel-data", "Test base class for DB entries per HCAL channel")
    ("test-channel-quality", "Test channel quality operations")
    ("test-channel-iterator", "Test iterator class for HCAL channels")
    ("test-new-developer", "Test area for a new developer")
    ("get-iovs-from-omds", "For a given tag, get the list of unique IOVs ordered by insertion time")
    ("test-tags-iovs-from-pool", "Test the extraction of lists of tags and IOVs from a Pool database")
    ("list-iovs-for-o2o", "For a given tag, dump a list of IOVs that need to be copied via O2O or report error if O2O is impossible")
    ("pool-connect-string", po::value<string>(), "Connect string to the Pool database")
    ("pool-auth-path", po::value<string>(), "Path to authenticate.xml for the Pool database")
    ("make-channel-quality-xml", "Create channel quiality loader XML with all channels")
    ;

  try{
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, general), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
      std::cout << general << "\n";
      return 1;
    }


    if (vm.count("quicktest")) {
      // db;
      std::cout << "HcalDbOmds: cms_hcal_dcs_02:HCAL_HV/HVcrate_HEMC/S17/RM4HV: " << HcalDbOmds::getDcsTypeFromDpName("cms_hcal_dcs_02:HCAL_HV/HVcrate_HEMC/S17/RM4HV") << std::endl;
      std::cout << "HcalDbOmds: cms_hcal_dcs_02:HCAL_HV/HVcrate_HO2P/S03/RM1BV: " << HcalDbOmds::getDcsTypeFromDpName("cms_hcal_dcs_02:HCAL_HV/HVcrate_HO2P/S03/RM1BV") << std::endl;
      return 0;
    }
    

    if (vm.count("test-lut-xml-file-access")) {
      std::cout << "Testing reading LUTs from local XML file..." << "\n";
      std::string in_ = vm["input-file"].as<std::string>();
      std::cout << "LUT XML file: " << in_ << "\n";
      //evaluate timing
      struct timeval _t;
      gettimeofday( &_t, NULL );
      double _time =(double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0;
      LutXml * _xml = new LutXml(in_);
      gettimeofday( &_t, NULL );
      std::cout << "Initialization took: " << (double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0 - _time << "sec" << std::endl;
      _xml->create_lut_map();
      _xml->test_access("noname");
      delete _xml;
      XMLProcessor::getInstance()->terminate();
      return 0;
    }


    if (vm.count("test-string")) {
      std::cout << "Test: "
	   << vm["test-string"].as<std::string>() << ".\n";
      XMLDOMBlock a("HCAL_TRIG_PRIM_LOOKUP_TABLE.dataset.template");
      a+=a;
      a.write("stdout");
      return 0;
    }

    if (vm.count("test-lmap")) {
      std::cout << "Testing lmap stuff..." << "\n";
      std::string _accessor = vm["test-lmap"].as<std::string>();
      std::cout << "Logical map accessor string: " << _accessor << "\n";
      std::string _type;
      if ( _accessor . find ("HO") != std::string::npos ){
	cout << "type: HO" << std::endl;
	_type = "HO";
      }
      else{
	cout << "type: HBEF" << std::endl;
	_type = "HBEF";
      }
      LMap_test test;
      test . test_read( _accessor, _type);
      return 0;
    }
    
    if (vm.count("test-emap")) {
      std::cout << "Testing emap stuff..." << "\n";
      std::string _accessor = vm["test-emap"].as<std::string>();
      std::cout << "Electronic map accessor string: " << _accessor << "\n";
      //EMap_test test;
      //test . test_read_map( _accessor );
      //
      HcalLutManager _m;
      _m . test_emap();
      return 0;
    }
    
    if (vm.count("test-qie")) {
      std::cout << "Testing QIE stuff..." << "\n";
      std::string _accessor = vm["test-qie"].as<std::string>();
      std::cout << "File with the query: " << _accessor << "\n";
      HcalQIEManager manager;
      manager . getTableFromDb( _accessor, "asdf" );
      return 0;
    }
    
    if (vm.count("test-lut-manager")) {
      std::cout << "Testing LUT manager stuff..." << "\n";
      std::string _accessor = vm["test-lut-manager"].as<std::string>();
      std::cout << "LUT ascii file: " << _accessor << "\n";
      HcalLutManager_test::getLutSetFromFile_test( _accessor );
      return 0;
    }
    
    if (vm.count("test-lut-xml-access")) {
      std::cout << "Testing reading LUTs from local XML file..." << "\n";
      std::string in_ = vm["input-file"].as<std::string>();
      std::cout << "LUT XML file: " << in_ << "\n";
      std::string tag_ = vm["tag-name"].as<std::string>();
      std::cout << "Tag: " << tag_ << "\n";
      HcalLutManager manager;
      manager . test_xml_access( tag_, in_ );
      return 0;
    }
    
    if (vm.count("test-lut-checksum")) {
      std::cout << "Testing evaluation of LUT MD5 checksums..." << "\n";
      std::string in_ = vm["input-file"].as<std::string>();
      std::cout << "LUT XML file: " << in_ << "\n";
      std::string tag_ = vm["tag-name"].as<std::string>();
      std::cout << "Tag: " << tag_ << "\n";
      HcalLutManager manager;
      manager . test_xml_access( tag_, in_ );
      return 0;
    }
    
    if (vm.count("qie")) {
      std::cout << "Generating new QIE table..." << "\n";
      std::cout << "Input file (from DB)... ";
      std::string _in = vm["input-file"].as<std::string>();
      std::cout << _in << std::endl;
      std::cout << "Output file... ";
      std::string _out = vm["output-file"].as<std::string>();
      std::cout << _out << std::endl;
      std::cout << "Old QIE table file (to fill missing channels)... ";
      std::string _old = vm["old-qie-file"].as<std::string>();
      std::cout << _old << std::endl;
      HcalQIEManager manager;
      manager . generateQieTable( _in, _old, _out );
      return 0;
    }
    
    if (vm.count("hf-qie")) {
      std::cout << "Retrieving HCAL HF QIE ADC data..." << "\n";
      std::cout << "Input file (from DB)... ";
      std::string _in = vm["input-file"].as<std::string>();
      std::cout << _in << std::endl;
      std::cout << "Output file... ";
      std::string _out = vm["output-file"].as<std::string>();
      std::cout << _out << std::endl;
      HcalQIEManager manager;
      manager . getHfQieTable( _in, _out );
      return 0;
    }
    
    if (vm.count("create-lin-lut-xml")) {
      while(1){
	cout << "Creating XML with LUTs for all channels..." << "\n";
	//int _cr = vm["crate"].as<int>();
	string lin_master_file, comp_master_file;
	if (!vm.count("lin-lut-master-file")){
	  std::cout << "Linearizer LUT master file name is not specified..." << std::endl;
	  lin_master_file = "";
	}
	else{
	  lin_master_file = vm["lin-lut-master-file"].as<std::string>();
	}
	if (!vm.count("tag-name")){
	  std::cout << "tag name is not specified...exiting" << std::endl;
	  break;
	}
	string _tag = vm["tag-name"].as<std::string>();
	HcalLutManager manager;
	manager . createLinLutXmlFiles( _tag, lin_master_file, !vm.count("do-not-split-by-crate") );
	break;
      }
      return 0;
    }

    if (vm.count("create-lut-xml")) {
      while(1){
	cout << "Creating XML with LUTs for all channels..." << "\n";
	//int _cr = vm["crate"].as<int>();
	string lin_master_file, comp_master_file;
	if (!vm.count("lin-lut-master-file")){
	  std::cout << "Linearizer LUT master file name is not specified..." << std::endl;
	  lin_master_file = "";
	}
	else{
	  lin_master_file = vm["lin-lut-master-file"].as<std::string>();
	}
	if (!vm.count("comp-lut-master-file")){
	  std::cout << "Compression LUT master file name is not specified..." << std::endl;
	  comp_master_file = "";
	}
	else{
	  comp_master_file = vm["comp-lut-master-file"].as<std::string>();
	}
	if (!vm.count("tag-name")){
	  std::cout << "tag name is not specified...exiting" << std::endl;
	  break;
	}
	string _tag = vm["tag-name"].as<std::string>();
	HcalLutManager manager;
	if (comp_master_file.find("nofile")==string::npos){
	  manager . createAllLutXmlFiles( _tag, lin_master_file, comp_master_file, !vm.count("do-not-split-by-crate") );
	}
	else{
	  manager . createLinLutXmlFiles( _tag, lin_master_file, !vm.count("do-not-split-by-crate") );
	}
	break;
      }
      return 0;
    }

    if (vm.count("create-lut-xml-from-coder")) {
      std::cout << "Creating XML with LUTs for all channels from TPG coder..." << "\n";
      if (!vm.count("tag-name")){
	cout << "tag name is not specified...exiting" << std::endl;
	exit(-1);
      }
      std::string _tag = vm["tag-name"].as<std::string>();
      HcalLutManager manager;
      manager . createCompLutXmlFilesFromCoder( _tag, !vm.count("do-not-split-by-crate") );
      return 0;
    }
    


    if (vm.count("create-lut-xml-lin-ascii-comp-coder")) {
      while(1){
	cout << "Creating XML with LUTs for all channels..." << "\n";
	//int _cr = vm["crate"].as<int>();
	string lin_master_file, comp_master_file;
	if (!vm.count("lin-lut-master-file")){
	  std::cout << "Linearizer LUT master file name is not specified..." << std::endl;
	  lin_master_file = "";
	}
	else{
	  lin_master_file = vm["lin-lut-master-file"].as<std::string>();
	}
	if (!vm.count("tag-name")){
	  std::cout << "tag name is not specified...exiting" << std::endl;
	  break;
	}
	string _tag = vm["tag-name"].as<std::string>();
	HcalLutManager manager;
	manager . createAllLutXmlFilesLinAsciiCompCoder( _tag, lin_master_file, !vm.count("do-not-split-by-crate") );
	break;
      }
      return 0;
    }



    if (vm.count("create-trigger-key")) {
      std::cout << "Creating trigger key XML..." << "\n";
      /*
      if (!vm.count("tag-name")){
	cout << "tag name is not specified...exiting" << std::endl;
	exit(-1);
      }
      std::string _tag = vm["tag-name"].as<std::string>();
      */
      HcalTriggerKey _key;
      _key.compose_key_dialogue();
      _key.write("HCAL_trigger_key.xml");
      //_key.add_data("aaa","bbb","ccc");
      //_key.add_data("aaa","bbb","ccc");
      //_key.add_data("aaa","bbb","ccc");
      return 0;
    }
    
    if (vm.count("get-lut-xml-from-oracle")) {
      std::cout << "Getting LUTs from Oracle database..." << "\n";
      if (!vm.count("tag-name")){
	cout << "tag name is not specified...exiting" << std::endl;
	exit(-1);
      }
      std::string _tag = vm["tag-name"].as<std::string>();
      std::string _accessor = vm["database-accessor"].as<std::string>();
      HcalLutManager manager;
      //manager . get_brickSet_from_oracle( _tag );
      std::cout << "Accessing the database as " << _accessor << std::endl;
      std::cout << "Tag name: " << _tag << std::endl;
      manager . get_xml_files_from_db( _tag, _accessor, !vm.count("do-not-split-by-crate") );
      return 0;
    }


    if (vm.count("create-lut-loader")){
      std::cout << "===> Processing LUT XML files, creating the database loader..." << "\n";
      std::cout << "prefix: ";
      std::string _prefix = vm["prefix-name"].as<std::string>();
      std::cout << _prefix << std::endl;
      std::cout << "TAG_NAME: ";
      std::string _tag = vm["tag-name"].as<std::string>();
      std::cout << _tag << std::endl;
      std::cout << "COMMENT: " << std::endl;
      std::string _comment = vm["comment-line"].as<std::string>();
      std::cout << _comment << std::endl;
      std::cout << "VERSION: ";
      std::string _version = vm["version-name"].as<std::string>();
      std::cout << _version << std::endl;
      std::cout << "SUBVERSION: ";
      int _subversion = vm["sub-version"].as<int>();
      std::cout << _subversion << std::endl;
      std::string _file_list = vm["file-list"].as<std::string>();
      HcalLutManager manager;
      manager . create_lut_loader( _file_list, _prefix, _tag, _comment, _version, _subversion );
      return 0;
    }


    if (vm.count("test-channel-data")) {
      HcalChannelQualityXml xml;
      //DOMNode * ds = xml.add_dataset();
      xml.set_header_run_number(1);
      xml.set_elements_tag_name("gak_v2");
      xml.set_elements_iov_begin(1000);
      //xml.add_hcal_channel_dataset(39, 15, 1, "HF", 1, 0, "test channel quality comment field");
      //xml.set_all_channels_on_off( 0, 1, 1, 1);
      xml.set_all_channels_status( 2, 2, 2, 2);
      /*
      HcalChannelQualityXml::ChannelQuality cq;
      cq.status=2;
      cq.onoff=1;
      cq.comment = "test 18 Jul 2009";
      std::map<int, HcalChannelQualityXml::ChannelQuality> _mcq;
      _mcq.insert(std::pair<int, HcalChannelQualityXml::ChannelQuality>(-30311,cq));
      cq.onoff=0;
      _mcq.insert(std::pair<int, HcalChannelQualityXml::ChannelQuality>(-40311,cq));
      cq.status=1;
      _mcq.insert(std::pair<int, HcalChannelQualityXml::ChannelQuality>(50311,cq));
      xml.addChannelQualityGeom(_mcq);
      */
      xml.write("channel_quality.xml");
      return 0;
    }
    
    
    if (vm.count("make-channel-quality-xml")) {
      HcalChannelQualityXml xml;
      xml.set_header_run_number(1);
      xml.set_elements_tag_name("gak_v2");
      xml.set_elements_iov_begin(10000);
      xml.set_all_channels_status( 4, 4, 4, 4);
      xml.write("channel_quality.xml");
      return 0;
    }
    
    
    if (vm.count("test-channel-quality")) {
      HcalChannelQualityXml xml;
      int nChan = xml.getBaseLineFromOmds("AllChannelsMasked16Jul2009v1", 82000);
      xml.addChannelToGeomIdMap(41,71,2, "HF", 23,23,"huj");
      std::cout << "Channels quality obtained from OMDS for " << nChan << " channels" << std::endl;
      for(std::map<int, HcalChannelQualityXml::ChannelQuality>::const_iterator cq = xml.geomid_cq.begin();
	  cq != xml.geomid_cq.end();
	  cq++){
	cout << cq->first << "     " << cq->second.status << std::endl;
      }
      std::cout << "Channels quality obtained from OMDS for " << nChan << " channels" << std::endl;
      return 0;
    }
    
    
   
    if (vm.count("test-channel-iterator")) {
      HcalChannelIterator iter;
      iter.clearChannelList();
      iter.addListFromLmapAscii("HCALmapHBEF_Jan.27.2009.txt");
      iter.addListFromLmapAscii("HCALmapHO_Jan.27.2009.txt");
      std::cout << "The iterator list contains " << iter.size() << " entries (not necessarily unique)" << std::endl;
      std::cout << "Testing the iterator over all entries now..." << std::endl;
      for (iter.begin(); !iter.end(); iter.next()){
	cout << iter.getHcalGenericDetId() << std::endl;
      }
      return 0;
    }
    
    
    if (vm.count("get-iovs-from-omds")) {
      if (!vm.count("tag-name")) {
	cout << "Tag name is not specified! Exiting..." << "\n";
	return -1;
      }
      else{
	string _tag = vm["tag-name"].as<std::string>();
	HcalO2OManager m;
	std::vector<uint32_t> _iovs;
	m.getListOfOmdsIovs(_iovs, _tag);
	std::copy (_iovs.begin(),
		   _iovs.end(),
		   std::ostream_iterator<uint32_t>(std::cout,"\n")
		   );
	return 0;
      }
    }
    
    
    if (vm.count("test-tags-iovs-from-pool")) {
      HcalO2OManager m;
      // default connect string
      //std::string connect = "frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL";
      // for the private network (hcaldqm04):
      //std::string connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_HCAL";
      std::string connect = "sqlite:testExample.db";
      std::string pool_auth_path = "/afs/cern.ch/cms/DB/conddb";
      std::vector<std::string> alltags = m.getListOfPoolTags(connect,
							     pool_auth_path);
      std::copy (alltags.begin(),
		 alltags.end(),
		 std::ostream_iterator<std::string>(std::cout,"\n")
		 );
      std::string tag = "HcalPedestals_ADC_v7.03_hlt";
      std::vector<uint32_t> allIovs;
      m.getListOfPoolIovs(allIovs,
			  tag,
			  connect,
			  pool_auth_path);
      std::copy (allIovs.begin(),
		 allIovs.end(),
		 std::ostream_iterator<uint32_t>(std::cout,"\n")
		 );
      return 0;
    }
    
    // FIXME: transfering this to HcalO2OManager...
    if (vm.count("list-iovs-for-o2o")) {
      if (!vm.count("tag-name")) {
	cout << "Tag name is not specified! Exiting..." << "\n";
	return -1;
      }
      else if (!vm.count("pool-connect-string")) {
	cout << "Pool connect string is not specified! Exiting..." << "\n";
	return -1;
      }
      else if (!vm.count("pool-auth-path")) {
	cout << "Pool path to authenticate.xml is not specified! Exiting..." << "\n";
	return -1;
      }
      else{
	string _tag = vm["tag-name"].as<std::string>();
	string pool_connect_string = vm["pool-connect-string"].as<std::string>();
	string pool_auth_path = vm["pool-auth-path"].as<std::string>();
	//std::cout << "DEBUG: " << pool_connect_string << std::endl;
	HcalO2OManager m;
	std::vector<uint32_t> _iovs;
	std::vector<uint32_t> omds_iovs;
	std::vector<uint32_t> pool_iovs;
	m.getListOfOmdsIovs(omds_iovs, _tag);

	m.getListOfPoolIovs(pool_iovs, _tag, pool_connect_string, pool_auth_path);
	int n_iovs = m.getListOfNewIovs(_iovs,
					omds_iovs,
					pool_iovs);
	if (n_iovs == -1){
	  std::cout << "HcalO2OManager: O2O is not possible" << std::endl;
	}
	else if (n_iovs == 0){
	  std::cout << "HcalO2OManager: O2O is not needed, the tag is up to date" << std::endl;
	}
	else{
	  // Report a new POOL tag. This will be picked up by the o2o.py script,
	  // and the corresponding IOV in the offline tag will be changed to 1
	  // because the first IOV in the offline tag must always be 1.
	  // In some cases, special actions are taken by the o2o script
	  // for the very first payload in a new tag.
	  if (pool_iovs.size()==0){
	    std::cout << "NEW_POOL_TAG_TRUE: " << _tag << std::endl;
	  }
	  for (std::vector<uint32_t>::const_iterator iov = _iovs.begin();
	       iov != _iovs.end();
	       ++iov){
	    std::cout << "O2O_IOV_LIST: " << *iov << std::endl;
	  }
	}
	return 0;
      }
      std::cout << "This should never be printed out. Something is fishy!" << std::endl;
    }
    
    
    if (vm.count("test-new-developer")) {
      std::cout << "Wazzup, dude?! What would you like to do?.." << "\n";
      return 0;
    }
    
    
    
  } catch(boost::program_options::unknown_option) {
    std::cout << "No command line options known to boost... continuing to getopt parser..." << std::endl;
  }

  //
  // FIXME: deprecated parse command line options - switch to boost above
  //
  int c;
  //int digit_optind = 0;
  
  // default parameter values
  // bool luts = false;
  bool rbx = false;
  bool tag_b = false;
  //bool testdb_b = false;
  bool lmaptest_b = false;
  bool hardware_b = false;
  bool test_db_access_b = false;
  bool zs2_b = false;
  bool zs2HB_b = false;
  bool zs2HE_b = false;
  bool zs2HO_b = false;
  bool zs2HF_b = false;
  bool test_lut_gen_b = false;

  std::string filename = "";
  std::string path = "";
  std::string tag = "";
  std::string comment = "";
  std::string version = "";
  std::string prefix = "";
  std::string rbx_type = "";
  std::string aramsParameter = "";
  std::string zs2HB = "";
  std::string zs2HE = "";
  std::string zs2HO = "";
  std::string zs2HF = "";

  while (1) {
    //int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
      {"filename", 1, 0, 1},
      {"path", 1, 0, 2},
      {"tag", 1, 0, 3},
      {"prefix", 1, 0, 4},
      {"comment", 1, 0, 5},
      {"version", 1, 0, 6},
      {"luts", 0, 0, 10},
      {"patterns", 0, 0, 20},
      {"lmap", 0, 0, 30},
      {"rbxpedestals", 0, 0, 40},
      // deprecated     {"zs", 0, 0, 50},
      {"rbx", 1, 0, 60},
      {"luts2", 0, 0, 70},
      {"testocci", 0, 0, 1000},
      //{"testdb", 1, 0, 1010},
      {"lmaptest", 1, 0, 2000},
      {"hardware", 0, 0, 1050},
      {"test-db-access", 0, 0, 1070},
      {"zs2", 0, 0, 1080},
      {"zs2HB", 1, 0, 1090},
      {"zs2HE", 1, 0, 1091},
      {"zs2HO", 1, 0, 1092},
      {"zs2HF", 1, 0, 1093},
      {"test-lut-gen", 0, 0, 1100},
      {0, 0, 0, 0}
    };
        

    c = getopt_long (argc, argv, "",
		     long_options, &option_index);

    //std::cout << c << std::endl;

    if (c == -1)
      {
	break;
      }
    
    switch (c) {

    case 1:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //std::cout << "filename: " << _buf << std::endl;
	  filename .append( _buf );
	}
      else
	{
	  std::cout << "Missing file name!" << std::endl;
	}
      break;
      
    case 2:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //std::cout << "path: " << _buf << std::endl;
	  path .append( _buf );
	}
      else
	{
	  std::cout << "Empty path!" << std::endl;
	}
      break;
      
    case 3:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //std::cout << "path: " << _buf << std::endl;
	  tag .append( _buf );
	  tag_b = true;
	}
      else
	{
	  std::cout << "Empty tag!" << std::endl;
	}
      break;
      
    case 4:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //std::cout << "path: " << _buf << std::endl;
	  prefix .append( _buf );
	}
      else
	{
	  std::cout << "Empty prefix!" << std::endl;
	}
      break;
      
    case 5:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  comment . append( _buf );
	}
      else
	{
	  std::cout << "Empty comment!" << std::endl;
	}
      break;
      
    case 6:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  version . append( _buf );
	}
      else
	{
	  std::cout << "Empty comment!" << std::endl;
	}
      break;
      
    case 10:
      createLUTLoader();
      break;
      
    case 20:
      createHTRPatternLoader();
      break;
      
    case 30:
      createLMap();
      break;
      
    case 40:
      //createRBXLoader();
      break;
      
      // deprecated - to be removed
      //    case 50:
      //      createZSLoader();
      //      break;
      
      // rbx
    case 60:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  rbx_type .append( _buf );
	  rbx = true;
	}
      else
	{
	  std::cout << "RBX data type not defined!.." << std::endl;
	}
      break;

    case 70:
      //luts = true;
      break;
      
    case 1000: // testocci
      testocci();
      break;
      
      //case 1010: // testdb
      //if ( optarg )
      //{
      //  char _buf[1024];
      //  sprintf( _buf, "%s", optarg );
      //  //std::cout << "filename: " << _buf << std::endl;
      //  filename .append( _buf );
      //  testdb_b = true;
      //}
      //else
      //{
      //  std::cout << "No XML file name specified! " << std::endl;
      //}
      //break;

    case 2000: // lmaptest
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  aramsParameter .append( _buf );
	  lmaptest_b = true;
	}
      else
	{
	  std::cout << "No parameter specified! " << std::endl;
	}
      break;

    case 1050: // HCAL hardware map
      hardware_b=true;
      break;
      
    case 1070: // oracle access example to lmap and stuff for Dmitry
      test_db_access_b=true;
      break;
      
    case 1080: // ZS generator ver.2
      zs2_b=true;
      break;
      
    case 1090: // ZS for HB
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  zs2HB .append( _buf );
	  zs2HB_b = true;
	}
      else
	{
	  std::cout << "No zero suppression value for HB specified... " << std::endl;
	}
      break;

    case 1091: // ZS for HE
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  zs2HE .append( _buf );
	  zs2HE_b = true;
	}
      else
	{
	  std::cout << "No zero suppression value for HE specified... " << std::endl;
	}
      break;

    case 1092: // ZS for HO
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  zs2HO .append( _buf );
	  zs2HO_b = true;
	}
      else
	{
	  std::cout << "No zero suppression value for HO specified... " << std::endl;
	}
      break;

    case 1093: // ZS for HF
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  zs2HF .append( _buf );
	  zs2HF_b = true;
	}
      else
	{
	  std::cout << "No zero suppression value for HF specified... " << std::endl;
	}
      break;

    case 1100: // test LUT XML generator
      test_lut_gen_b=true;
      break;
      
    default:
      printf ("?? getopt returned character code 0%o ??\n", c);
    }
  }
  
  if (optind < argc) {
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      printf ("%s ", argv[optind++]);
    printf ("\n");
  }

  // FIXME: deprecated - remove
  /*
  if ( luts )
    {
      std::cout << "path: " << path << std::endl;
      std::cout << "prefix: " << prefix << std::endl;
      std::cout << "TAG_NAME: " << tag << std::endl;
      createLUTLoader( prefix, tag );
    }
  */
  //else
  if ( rbx )
    {
      //
      //_____ fix due to the new convention: version/subversion combo must be unique for every payload
      //
      char _buf[128];
      time_t _offset = time(NULL);
      sprintf( _buf, "%d", (uint32_t)_offset );
      version.append(".");
      version.append(_buf);

      std::cout << "type: " << rbx_type << std::endl;
      std::cout << "TAG_NAME: " << tag << std::endl;
      std::cout << "comment: " << comment << std::endl;
      std::cout << "version: " << version << std::endl;
      //std::cout << "list file: " << filename << std::endl;

      if ( tag_b ){
	createRBXLoader( rbx_type, tag, filename, comment, version );      
      }
      else{
	cout << "Tag name not specified... exiting" << std::endl;
      }
    }
  //else if ( testdb_b && tag_b )
  //{
  //  testDB( tag, filename );      
  //}

  else if ( lmaptest_b )
    {
      lmaptest( aramsParameter );      
    }
  else if ( hardware_b )
    {
      hardware();      
    }
  else if ( test_db_access_b )
    {
      test_db_access();      
    }
  else if ( zs2_b )
    {
      while(1){
	if ( !tag_b ){
	  std::cout << "No tag specified... exiting" << std::endl;
	  break;
	}
	if ( !zs2HB_b ){
	  std::cout << "No zero suppression value dor HB specified... exiting" << std::endl;
	  break;
	}
	if ( !zs2HE_b ){
	  std::cout << "No zero suppression value dor HE specified... exiting" << std::endl;
	  break;
	}
	if ( !zs2HO_b ){
	  std::cout << "No zero suppression value dor HO specified... exiting" << std::endl;
	  break;
	}
	if ( !zs2HF_b ){
	  std::cout << "No zero suppression value dor HF specified... exiting" << std::endl;
	  break;
	}
	createZSLoader2( tag, comment, zs2HB, zs2HE, zs2HO, zs2HF );
	break;
      }
    }
  else if ( test_lut_gen_b )
    {
      test_lut_gen();      
    }
  else
    {
      //std::cout << "Nothing to do!" << std::endl;
    }
  

  delete XMLProcessor::getInstance();

  std::cout << "xmlTools ...done" << std::endl;
  exit (0);  
}







/* deprecated - to be removed
//
// Zero suppression Loader
int createZSLoader( void )
{
  XMLHTRZeroSuppressionLoader::loaderBaseConfig baseConf;
  XMLHTRZeroSuppressionLoader::datasetDBConfig conf;

  std::string _prefix = "GREN_ZS_9adc_v2";
  std::string _comment = "ZS for GREN07, (tune=3.5)*2+2";
  std::string _version = "GREN07:1";
  std::string _subversion = "1";

  baseConf . tag_name = _prefix;
  baseConf . comment_description = _comment;
  baseConf . elements_comment_description = _comment;
  baseConf . run_number = 1;
  baseConf . iov_begin = 1 ;
  baseConf . iov_end = -1 ;

  XMLHTRZeroSuppressionLoader doc( &baseConf );
  //doc . createLoader( );

  LMap map_hbef;
  map_hbef . read( "HCALmapHBEF_11.9.2007.txt", "HBEF" ); // HBEF logical map

  for( std::vector<LMapRow>::const_iterator row = map_hbef . _table . begin(); row != map_hbef . _table . end(); row++ )
    {
      conf . comment_description = _comment;
      conf . version = _version;
      conf . subversion = _subversion;
      conf . eta = (row -> eta);
      conf . z = (row -> side);
      conf . phi = row -> phi;
      conf . depth = row -> depth;
      conf . detector_name = row -> det;

      HcalSubdetector _subdet;
      if ( row->det . c_str() == "HB" ) _subdet = HcalBarrel;
      if ( row->det . c_str() == "HE" ) _subdet = HcalEndcap;
      if ( row->det . c_str() == "HO" ) _subdet = HcalOuter;
      if ( row->det . c_str() == "HF" ) _subdet = HcalForward;
      HcalDetId _hcaldetid( _subdet, (row->side)*(row->eta), row->phi, row->depth );
      conf . hcal_channel_id = _hcaldetid . rawId();

      conf . zero_suppression = 9;

      doc . addZS( &conf );
    }
  
  LMap map_ho;
  map_ho . read( "HCALmapHO_11.9.2007.txt", "HO" ); // HO logical map

  for( std::vector<LMapRow>::const_iterator row = map_ho . _table . begin(); row != map_ho . _table . end(); row++ )
    {
      conf . comment_description = _comment;
      conf . version = _version;
      conf . subversion = _subversion;
      conf . eta = (row -> eta);
      conf . z = (row -> side);
      conf . phi = row -> phi;
      conf . depth = row -> depth;
      conf . detector_name = row -> det;

      HcalSubdetector _subdet;
      if ( row->det . c_str() == "HB" ) _subdet = HcalBarrel;
      if ( row->det . c_str() == "HE" ) _subdet = HcalEndcap;
      if ( row->det . c_str() == "HO" ) _subdet = HcalOuter;
      if ( row->det . c_str() == "HF" ) _subdet = HcalForward;
      HcalDetId _hcaldetid( _subdet, (row->side)*(row->eta), row->phi, row->depth );
      conf . hcal_channel_id = _hcaldetid . rawId();

      conf . zero_suppression = 9;

      doc . addZS( &conf );
    }
  
  doc . write( _prefix + "_ZeroSuppressionLoader.xml" );
  
  return 0;
}
*/

// Zero suppression Loader version 2
int createZSLoader2( std::string & tag, std::string & comment, std::string & zs2HB, std::string & zs2HE, std::string & zs2HO, std::string & zs2HF )
{

  XMLHTRZeroSuppressionLoader::loaderBaseConfig baseConf;
  XMLHTRZeroSuppressionLoader::datasetDBConfig conf;

  std::string lmap_version = "30";

  std::string _prefix = tag;
  std::string _comment = comment;

  //
  //_____ fix due to the new convention: version/subversion combo must be unique for every payload
  //
  char _buf[128];
  time_t _offset = time(NULL);
  sprintf( _buf, "%d", (uint32_t)_offset );
  std::string _version;
  _version.clear();
  _version.append(tag);
  _version.append(".");
  _version.append(_buf);
  int dataset_count = 0;

  baseConf . tag_name = _prefix;
  baseConf . comment_description = _comment;
  baseConf . elements_comment_description = _comment;
  baseConf . run_number = 1;
  baseConf . iov_begin = 1 ;
  baseConf . iov_end = -1 ;

  XMLHTRZeroSuppressionLoader doc( &baseConf );

  // loop over LMAP
  HCALConfigDB * db = new HCALConfigDB();
  const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );

  oracle::occi::Connection * _connection = db -> getConnection();  

  int eta_abs, side, phi, depth;
  std::string subdet;

  std::cout << "Preparing to request the LMAP from the database..." << std::endl;

  try {
    std::cout << "Preparing the query...";
    Statement* stmt = _connection -> createStatement();
    std::string query = ("SELECT eta, side, phi, depth, subdetector, cds.version ");
    query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V3 lmap";
    query += " join cms_hcl_core_condition_owner.cond_data_sets cds ";
    query += " on cds.condition_data_set_id=lmap.condition_data_set_id ";
    query += toolbox::toString(" WHERE version='%s'", lmap_version . c_str() );
    std::cout << " done" << std::endl;    

    //SELECT
    std::cout << "Executing the query...";
    ResultSet *rs = stmt->executeQuery(query.c_str());
    std::cout << " done" << std::endl;

    RooGKCounter _channels(1,100);
    _channels . setNewLine( false );

    std::cout << "Going through HCAL channels..." << std::endl;
    while (rs->next()) {
      _channels . count();
      eta_abs  = rs -> getInt(1);
      side    = rs -> getInt(2);
      phi     = rs -> getInt(3);
      depth   = rs -> getInt(4);
      subdet  = rs -> getString(5);
      
      conf . comment_description = _comment;
      conf . version = _version;
      sprintf( _buf, "%.2d", dataset_count );
      conf.subversion.clear();
      conf.subversion.append(_buf);
      conf . eta = eta_abs;
      conf . z = side;
      conf . phi = phi;
      conf . depth = depth;
      conf . detector_name = subdet;

      int _zs = 0;
      HcalSubdetector _subdet;
      if ( subdet == "HB" ){
	_subdet = HcalBarrel;
	sscanf(zs2HB.c_str(),"%d", &_zs);
      }
      else if ( subdet == "HE" ){
	_subdet = HcalEndcap;
	sscanf(zs2HE.c_str(),"%d", &_zs);
      }
      else if ( subdet == "HO" ){
	_subdet = HcalOuter;
	sscanf(zs2HO.c_str(),"%d", &_zs);
      }
      else if ( subdet == "HF" ){
	_subdet = HcalForward;
	sscanf(zs2HF.c_str(),"%d", &_zs);
      }
      else{
	_subdet = HcalOther;
      }
      HcalDetId _hcaldetid( _subdet, side*eta_abs, phi, depth );
      conf . hcal_channel_id = _hcaldetid . rawId();

      //conf . zero_suppression = 9;
      conf . zero_suppression = _zs;

      doc . addZS( &conf );
      ++dataset_count;
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }

  db -> disconnect();
  
  doc . write( _prefix + "_ZeroSuppressionLoader.xml" );
  
  return 0;
}



// LUT Loader
int createLUTLoader( std::string _prefix, std::string tag_name )
{
  std::cout << "Generating XML loader for LUTs..." << std::endl;
  std::cout << _prefix << "..." << tag_name << std::endl;

  XMLLUTLoader::loaderBaseConfig baseConf;
  XMLLUTLoader::lutDBConfig conf;
  XMLLUTLoader::checksumsDBConfig CSconf;

  baseConf . tag_name = tag_name;
  //baseConf . comment_description = tag_name;
  baseConf . comment_description = "Version 2 (HO bug fixed, now input and output LUTs for HO should be in.) Input: pedestal compensation by 3 for HB, HE, HF, and HO, nominal linearization, output: |ieta|<=14 thr=7 (equal or more), |ieta|=15 thr=9, |ieta|=16 thr=10, |ieta|>=17 & |ieta|<=28 thr=9, |ieta|>=29 thr=5, checksums correct.";
  baseConf . iov_begin = "1";
  baseConf . iov_end = "-1";

  conf . version = _prefix + ":1";
  //conf . version = "GREN2007:1";
  conf . subversion = "1";

  CSconf . version = conf . version;
  CSconf . subversion = conf . subversion;
  CSconf . trig_prim_lookuptbl_data_file = _prefix + "_checksums.xml.dat";
  //CSconf . comment_description = _prefix + ": GREN 26Nov2007 checksums LUTs";
  CSconf . comment_description = tag_name;

  XMLLUTLoader doc( &baseConf );

  std::vector<int> crate_number;
  crate_number . push_back(0);
  crate_number . push_back(1);
  crate_number . push_back(2);
  crate_number . push_back(3);
  crate_number . push_back(4);
  crate_number . push_back(5);
  crate_number . push_back(6);
  crate_number . push_back(7);
  crate_number . push_back(9);
  crate_number . push_back(10);
  crate_number . push_back(11);
  crate_number . push_back(12);
  crate_number . push_back(13);
  crate_number . push_back(14);
  crate_number . push_back(15);
  crate_number . push_back(17);
  std::vector<std::string> file_name;
  file_name . push_back( "./" + _prefix + "_0.xml.dat" );
  file_name . push_back( "./" + _prefix + "_1.xml.dat" );
  file_name . push_back( "./" + _prefix + "_2.xml.dat" );
  file_name . push_back( "./" + _prefix + "_3.xml.dat" );
  file_name . push_back( "./" + _prefix + "_4.xml.dat" );
  file_name . push_back( "./" + _prefix + "_5.xml.dat" );
  file_name . push_back( "./" + _prefix + "_6.xml.dat" );
  file_name . push_back( "./" + _prefix + "_7.xml.dat" );
  file_name . push_back( "./" + _prefix + "_9.xml.dat" );
  file_name . push_back( "./" + _prefix + "_10.xml.dat" );
  file_name . push_back( "./" + _prefix + "_11.xml.dat" );
  file_name . push_back( "./" + _prefix + "_12.xml.dat" );
  file_name . push_back( "./" + _prefix + "_13.xml.dat" );
  file_name . push_back( "./" + _prefix + "_14.xml.dat" );
  file_name . push_back( "./" + _prefix + "_15.xml.dat" );
  file_name . push_back( "./" + _prefix + "_17.xml.dat" );
  for ( std::vector<std::string>::const_iterator _file = file_name . begin(); _file != file_name . end(); _file++ )
    {
      conf . trig_prim_lookuptbl_data_file = *_file;
      //conf . trig_prim_lookuptbl_data_file += ".dat";
      conf . crate = crate_number[ _file - file_name . begin() ];
      
      char _buf[128];
      sprintf( _buf, "CRATE%.2d", conf . crate );
      std::string _namelabel;
      _namelabel . append( _buf );
      conf . name_label = _namelabel;
      doc . addLUT( &conf );
    }
  
  doc . addChecksums( &CSconf );
  //doc . write( _prefix + "_Loader.xml" );
  doc . write( tag_name + "_Loader.xml" );

  std::cout << "Generating XML loader for LUTs... done." << std::endl;

  return 0;
}

int createHTRPatternLoader( void )
{
  // HTR Patterns Loader

  std::cout << "Generating XML loader for HTR patterns..." << std::endl;

  XMLHTRPatternLoader::loaderBaseConfig baseConf;
  baseConf . tag_name = "Test tag 1";
  baseConf . comment_description = "Test loading to the DB";
  
  XMLHTRPatternLoader doc( &baseConf );
  baseConf . run_mode = "no-run";
  baseConf . data_set_id = "-1";
  baseConf . iov_id = "1";
  baseConf . iov_begin = "1"; // beginning of the interval of validity 
  baseConf . iov_end = "-1"; // end of IOV, "-1" stands for +infinity
  baseConf . tag_name = "test tag";
  baseConf . comment_description = "comment";
  
  XMLHTRPatternLoader::datasetDBConfig conf;
  
  // add a file with the crate 2 patterns
  // repeat for each crate
  //--------------------->
  conf . htr_data_patterns_data_file = "file_crate02.xml.dat";
  conf . crate = 2;
  conf . name_label = "CRATE02";
  conf . version = "ver:1";
  conf . subversion = "1";
  conf . create_timestamp = 1100000000; // UNIX timestamp
  conf . created_by_user = "user";
  doc . addPattern( &conf );
  //---------------------->

  // write the XML to a file
  doc . write( "HTRPatternLoader.xml" );

  std::cout << "Generating XML loader for HTR patterns... done." << std::endl;

  // end of HTR Patterns Loader

  return 0;
}


int createLMap( void ){
  DBlmapWriter lw;
  lw.createLMap();
  return 0;
}


int createRBXLoader( std::string & type_, std::string & tag_, std::string & list_file, std::string & _comment, std::string & _version )
{
  std::string _prefix = "oracle_"; 
  std::string _tag = tag_;
  int dataset_count = 0; // subversion-to-be, must be unique within the version

  std::vector<std::string> brickFileList;
  char filename[1024];
  std::string listFileName = list_file;
  ifstream inFile( listFileName . c_str(), std::ios::in );
  if (!inFile)
    {
      std::cout << " Unable to open list file" << std::endl;
    }
  else
    {
      std::cout << "List file opened successfully: " << listFileName << std::endl;
    }
  while (inFile >> filename)
    {
      std::string fullFileName = filename;
      brickFileList . push_back( fullFileName );
    }
  inFile.close();

  for ( std::vector<std::string>::const_iterator _file = brickFileList . begin(); _file != brickFileList . end(); _file++ )
    {
      XMLRBXPedestalsLoader::loaderBaseConfig _baseConf;
      XMLRBXPedestalsLoader::datasetDBConfig _conf;

      _baseConf . elements_comment_description = _comment;
      _baseConf . tag_name = _tag;

      if ( type_ == "pedestals" ){
	_baseConf . extention_table_name = "HCAL_RBX_CONFIGURATION_TYPE01";
	_baseConf . name = "HCAL RBX configuration [PEDESTAL]";
      }
      else if ( type_ == "delays" ){
	_baseConf . extention_table_name = "HCAL_RBX_CONFIGURATION_TYPE01";
	_baseConf . name = "HCAL RBX configuration [DELAY]";
      }
      else if ( type_ == "gols" ){
	_baseConf . extention_table_name = "HCAL_RBX_CONFIGURATION_TYPE03";
	_baseConf . name = "HCAL RBX configuration [GOL]";
      }
      else if ( type_ == "leds" ){
	_baseConf . extention_table_name = "HCAL_RBX_CONFIGURATION_TYPE02";
	_baseConf . name = "HCAL RBX configuration [type 2]";
      }
      else{
	cout << "Unknown config type... exiting" << std::endl;
	exit(1);
      }

      _conf . version = _version;
      //
      //_____ making a unique substring within the version __________________
      //
      char _buf[128];
      sprintf( _buf, "%.2d", dataset_count );
      _conf.subversion.clear();
      _conf.subversion.append(_buf);
      _conf . comment_description = _comment;

      XMLRBXPedestalsLoader p( &_baseConf );

      p . addRBXSlot( &_conf, (*_file), type_ );
      p . write( (*_file) + ".oracle.xml" );
      ++dataset_count;
    }

  return 0;

}


int testocci( void )
{

  HCALConfigDB * db = new HCALConfigDB();
  const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );
  std::vector<unsigned int> _lut = db -> getOnlineLUT( "gren_P3Thr5", 17, 2, 1, 1, 0, 1 );

  std::cout << "LUT length = " << _lut . size() << std::endl;
  for ( std::vector<unsigned int>::const_iterator i = _lut . begin(); i != _lut . end(); i++ )
    {
      std::cout << (i-_lut.begin()) << "     " << _lut[(i-_lut.begin())] << std::endl;
    }

  db -> disconnect();
  

  return 0;
}

int test_db_access( void )
{
  // despite the name of the class, can be used with any Oracle DB
  HCALConfigDB * db = new HCALConfigDB();
  const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );

  oracle::occi::Connection * _connection = db -> getConnection();  

  unsigned int _version, _crate, _slot, _fiber, _channel;

  int side   = -1;
  int etaAbs =  1;
  int phi    =  1;
  int depth  =  1;
  std::string subdetector = "HB";

  std::cout << "version	" << "eta	" << "phi	" << "depth	" << "subdetector	";
  std::cout << "crate	" << "slot	" << "fiber	" << "channel	" << std::endl;

  try {
    Statement* stmt = _connection -> createStatement();
    std::string query = ("SELECT cds.version, CRATE, HTR_SLOT, HTR_FPGA, HTR_FIBER, FIBER_CHANNEL ");
    query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V3 lmap";
    query += " join cms_hcl_core_condition_owner.cond_data_sets cds ";
    query += " on cds.condition_data_set_id=lmap.condition_data_set_id ";
    query += toolbox::toString(" WHERE SIDE=%d AND ETA=%d AND PHI=%d AND DEPTH=%d AND SUBDETECTOR='%s'", side, etaAbs, phi, depth, subdetector . c_str() );
    
    //SELECT
    ResultSet *rs = stmt->executeQuery(query.c_str());

    while (rs->next()) {
      _version  = rs -> getInt(1);
      _crate    = rs -> getInt(2);
      _slot     = rs -> getInt(3);
      std::string fpga_ = rs -> getString(4);
      _fiber    = rs -> getInt(5);
      _channel  = rs -> getInt(6);
      
      std::cout << _version << "	" << side*etaAbs << "	" << phi << "	" << depth << "	" << subdetector << "		";
      std::cout << _crate << "	" << _slot << "	" << _fiber << "	" << _channel << std::endl;
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }

  db -> disconnect();
  
  return 0;
}




int lmaptest( std::string _param ){
  std::cout << "lmaptest() is running, param = " << _param << std::endl;

  DBlmapReader * dbr = new DBlmapReader();
  dbr->lrTestFunction();
 
  VectorLMAP* curLMAP = dbr->GetLMAP(30);

  FILE* HBEFfile = fopen ("HCALmapHBEF.txt","w");
  FILE* HOfile   = fopen ("HCALmapHO.txt","w");
  FILE* EMAPfile = fopen ("HCALemap.txt", "w");

  dbr->PrintLMAP(HBEFfile, HOfile, curLMAP);
  
  dbr->PrintEMAPfromLMAP(EMAPfile, curLMAP);
  return 0;
}



int hardware( void )
{
  HcalHardwareXml _hw;

  std::map<std::string,std::map<std::string,std::map<std::string,std::map<int,std::string> > > > hw_map;

  ifstream infile("HBHOHE.ascii");
  std::string buf;

  if ( infile . is_open() ){
  std::cout << "File is open" << std::endl;
    while (getline( infile, buf ))
      {
	vector<std::string> _line = splitString( buf );
	//std::cout << _line . size() << std::endl;
	if ( _line[0] != "XXXXXX" && _line[1] != "XXXXXX" && _line[2] != "XXXXXX" ){
	  if (_line[3] != "XXXXXX") hw_map[_line[0]][_line[1]]["3040000000000" + _line[2]][1] = "3040000000000" + _line[3];
	  if (_line[4] != "XXXXXX") hw_map[_line[0]][_line[1]]["3040000000000" + _line[2]][2] = "3040000000000" + _line[4];
	  if (_line[5] != "XXXXXX") hw_map[_line[0]][_line[1]]["3040000000000" + _line[2]][3] = "3040000000000" + _line[5];
	}
      }
  }
  
  std::cout << hw_map . size() << std::endl;

  _hw . addHardware( hw_map );
  _hw . write("HCAL_hardware.xml");

  return 0;
}


// courtesy of Fedor Ratnikov
std::vector <std::string> splitString (const std::string& fLine) {
  std::vector <std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ' ' || fLine [i] == '	' || i == fLine.size ()) {
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



void test_lut_gen( void )
{
  HcalLutManager _manager;
  std::vector<unsigned int> _l;
  _l.push_back(0);
  _l.push_back(0);
  _l.push_back(0);
  _l.push_back(1);
  _l.push_back(2);
  std::cout << _manager . getLutXml( _l );
}
