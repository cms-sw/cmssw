#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <fstream>
#include <boost/program_options.hpp>

//#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLLUTLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatterns.h"
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

#include "xgi/Utils.h"
#include "toolbox/string.h"
#include "occi.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"

using namespace std;
using namespace oracle::occi;
using namespace hcal;
namespace po = boost::program_options;

int createLUTLoader( string prefix_="", string tag_="" );
int createHTRPatternLoader( void );
int createLMap( void );
int createRBXLoader( string & type_, string & tag_, string & list_file, string & _comment, string & _version );
//int createRBXentries( string );

// deprecated - to be removed
//int createZSLoader( void );

int testocci( void );
int testDB( string _tag, string _filename );
int lmaptest( string _param );
int hardware( void );
int test_db_access( void );
std::vector <std::string> splitString (const std::string& fLine);
int createZSLoader2( string & tag, string & comment, string & zs2HB, string & zs2HE, string & zs2HO, string & zs2HF );
void test_lut_gen( void );

int main( int argc, char **argv )
{
  cout << "Running xmlTools..." << endl;

  //
  //===> command line options parser using boost  
  //
  int crate, sub_version;
  std::string db_accessor, version_name;
  po::options_description general("General options");
  general.add_options()
    ("help", "produce help message")
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
    ("database-accessor", po::value<string>(&db_accessor)->default_value("occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22"), "Database accessor string")
    ("lin-lut-master-file", po::value<string>(), "Linearizer LUT ASCII master file name")
    ("comp-lut-master-file", po::value<string>(), "Compression LUT ASCII master file name")
    ("do-not-split-by-crate", "output LUTs as a single XML instead of making a separate file for each crate")
    ("input-file", po::value<string>(), "Input file name")
    ("output-file", po::value<string>(), "Outputput file name")
    ("old-qie-file", po::value<string>(), "Old QIE table ASCII file")
    ("qie", "Generate new QIE table file")
    ("hf-qie", "Retrieve HF QIE ADC caps offsets and slopes")
    ("test-channel-data", "Test base class for DB entries per HCAL channel")
    ("test-new-developer", "Test area for a new developer")
    ;

  try{
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, general), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
      cout << general << "\n";
      return 1;
    }


    if (vm.count("test-lut-xml-file-access")) {
      cout << "Testing reading LUTs from local XML file..." << "\n";
      std::string in_ = vm["input-file"].as<string>();
      cout << "LUT XML file: " << in_ << "\n";
      //evaluate timing
      struct timeval _t;
      gettimeofday( &_t, NULL );
      double _time =(double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0;
      LutXml * _xml = new LutXml(in_);
      gettimeofday( &_t, NULL );
      cout << "Initialization took: " << (double)(_t . tv_sec) + (double)(_t . tv_usec)/1000000.0 - _time << "sec" << endl;
      _xml->create_lut_map();
      _xml->test_access("noname");
      delete _xml;
      XMLProcessor::getInstance()->terminate();
      return 0;
    }


    if (vm.count("test-string")) {
      cout << "Test: "
	   << vm["test-string"].as<string>() << ".\n";
      XMLDOMBlock a("HCAL_TRIG_PRIM_LOOKUP_TABLE.dataset.template");
      a+=a;
      a.write("stdout");
      return 0;
    }

    if (vm.count("test-lmap")) {
      cout << "Testing lmap stuff..." << "\n";
      string _accessor = vm["test-lmap"].as<string>();
      cout << "Logical map accessor string: " << _accessor << "\n";
      string _type;
      if ( _accessor . find ("HO") != string::npos ){
	cout << "type: HO" << endl;
	_type = "HO";
      }
      else{
	cout << "type: HBEF" << endl;
	_type = "HBEF";
      }
      LMap_test test;
      test . test_read( _accessor, _type);
      return 0;
    }
    
    if (vm.count("test-emap")) {
      cout << "Testing emap stuff..." << "\n";
      string _accessor = vm["test-emap"].as<string>();
      cout << "Electronic map accessor string: " << _accessor << "\n";
      //EMap_test test;
      //test . test_read_map( _accessor );
      //
      HcalLutManager _m;
      _m . test_emap();
      return 0;
    }
    
    if (vm.count("test-qie")) {
      cout << "Testing QIE stuff..." << "\n";
      string _accessor = vm["test-qie"].as<string>();
      cout << "File with the query: " << _accessor << "\n";
      HcalQIEManager manager;
      manager . getTableFromDb( _accessor, "asdf" );
      return 0;
    }
    
    if (vm.count("test-lut-manager")) {
      cout << "Testing LUT manager stuff..." << "\n";
      string _accessor = vm["test-lut-manager"].as<string>();
      cout << "LUT ascii file: " << _accessor << "\n";
      HcalLutManager_test::getLutSetFromFile_test( _accessor );
      return 0;
    }
    
    if (vm.count("test-lut-xml-access")) {
      cout << "Testing reading LUTs from local XML file..." << "\n";
      string in_ = vm["input-file"].as<string>();
      cout << "LUT XML file: " << in_ << "\n";
      string tag_ = vm["tag-name"].as<string>();
      cout << "Tag: " << tag_ << "\n";
      HcalLutManager manager;
      manager . test_xml_access( tag_, in_ );
      return 0;
    }
    
    if (vm.count("test-lut-checksum")) {
      cout << "Testing evaluation of LUT MD5 checksums..." << "\n";
      string in_ = vm["input-file"].as<string>();
      cout << "LUT XML file: " << in_ << "\n";
      string tag_ = vm["tag-name"].as<string>();
      cout << "Tag: " << tag_ << "\n";
      HcalLutManager manager;
      manager . test_xml_access( tag_, in_ );
      return 0;
    }
    
    if (vm.count("qie")) {
      cout << "Generating new QIE table..." << "\n";
      cout << "Input file (from DB)... ";
      string _in = vm["input-file"].as<string>();
      cout << _in << endl;
      cout << "Output file... ";
      string _out = vm["output-file"].as<string>();
      cout << _out << endl;
      cout << "Old QIE table file (to fill missing channels)... ";
      string _old = vm["old-qie-file"].as<string>();
      cout << _old << endl;
      HcalQIEManager manager;
      manager . generateQieTable( _in, _old, _out );
      return 0;
    }
    
    if (vm.count("hf-qie")) {
      cout << "Retrieving HCAL HF QIE ADC data..." << "\n";
      cout << "Input file (from DB)... ";
      string _in = vm["input-file"].as<string>();
      cout << _in << endl;
      cout << "Output file... ";
      string _out = vm["output-file"].as<string>();
      cout << _out << endl;
      HcalQIEManager manager;
      manager . getHfQieTable( _in, _out );
      return 0;
    }
    
    if (vm.count("create-lin-lut-xml")) {
      while(1){
	cout << "Creating XML with LUTs for all channels..." << "\n";
	int _cr = vm["crate"].as<int>();
	string lin_master_file, comp_master_file;
	if (!vm.count("lin-lut-master-file")){
	  cout << "Linearizer LUT master file name is not specified..." << endl;
	  lin_master_file = "";
	}
	else{
	  lin_master_file = vm["lin-lut-master-file"].as<string>();
	}
	if (!vm.count("tag-name")){
	  cout << "tag name is not specified...exiting" << endl;
	  break;
	}
	string _tag = vm["tag-name"].as<string>();
	HcalLutManager manager;
	manager . createLinLutXmlFiles( _tag, lin_master_file, !vm.count("do-not-split-by-crate") );
	break;
      }
      return 0;
    }

    if (vm.count("create-lut-xml")) {
      while(1){
	cout << "Creating XML with LUTs for all channels..." << "\n";
	int _cr = vm["crate"].as<int>();
	string lin_master_file, comp_master_file;
	if (!vm.count("lin-lut-master-file")){
	  cout << "Linearizer LUT master file name is not specified..." << endl;
	  lin_master_file = "";
	}
	else{
	  lin_master_file = vm["lin-lut-master-file"].as<string>();
	}
	if (!vm.count("comp-lut-master-file")){
	  cout << "Compression LUT master file name is not specified..." << endl;
	  comp_master_file = "";
	}
	else{
	  comp_master_file = vm["comp-lut-master-file"].as<string>();
	}
	if (!vm.count("tag-name")){
	  cout << "tag name is not specified...exiting" << endl;
	  break;
	}
	string _tag = vm["tag-name"].as<string>();
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
      cout << "Creating XML with LUTs for all channels from TPG coder..." << "\n";
      if (!vm.count("tag-name")){
	cout << "tag name is not specified...exiting" << endl;
	exit(-1);
      }
      string _tag = vm["tag-name"].as<string>();
      HcalLutManager manager;
      manager . createCompLutXmlFilesFromCoder( _tag, !vm.count("do-not-split-by-crate") );
      return 0;
    }
    


    if (vm.count("create-lut-xml-lin-ascii-comp-coder")) {
      while(1){
	cout << "Creating XML with LUTs for all channels..." << "\n";
	int _cr = vm["crate"].as<int>();
	string lin_master_file, comp_master_file;
	if (!vm.count("lin-lut-master-file")){
	  cout << "Linearizer LUT master file name is not specified..." << endl;
	  lin_master_file = "";
	}
	else{
	  lin_master_file = vm["lin-lut-master-file"].as<string>();
	}
	if (!vm.count("tag-name")){
	  cout << "tag name is not specified...exiting" << endl;
	  break;
	}
	string _tag = vm["tag-name"].as<string>();
	HcalLutManager manager;
	manager . createAllLutXmlFilesLinAsciiCompCoder( _tag, lin_master_file, !vm.count("do-not-split-by-crate") );
	break;
      }
      return 0;
    }



    if (vm.count("create-trigger-key")) {
      cout << "Creating trigger key XML..." << "\n";
      /*
      if (!vm.count("tag-name")){
	cout << "tag name is not specified...exiting" << endl;
	exit(-1);
      }
      string _tag = vm["tag-name"].as<string>();
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
      cout << "Getting LUTs from Oracle database..." << "\n";
      if (!vm.count("tag-name")){
	cout << "tag name is not specified...exiting" << endl;
	exit(-1);
      }
      string _tag = vm["tag-name"].as<string>();
      string _accessor = vm["database-accessor"].as<string>();
      HcalLutManager manager;
      //manager . get_brickSet_from_oracle( _tag );
      cout << "Accessing the database as " << _accessor << endl;
      cout << "Tag name: " << _tag << endl;
      manager . get_xml_files_from_db( _tag, _accessor, !vm.count("do-not-split-by-crate") );
      return 0;
    }


    if (vm.count("create-lut-loader")){
      cout << "===> Processing LUT XML files, creating the database loader..." << "\n";
      cout << "prefix: ";
      string _prefix = vm["prefix-name"].as<string>();
      cout << _prefix << endl;
      cout << "TAG_NAME: ";
      string _tag = vm["tag-name"].as<string>();
      cout << _tag << endl;
      cout << "COMMENT: " << endl;
      string _comment = vm["comment-line"].as<string>();
      cout << _comment << endl;
      cout << "VERSION: ";
      string _version = vm["version-name"].as<string>();
      cout << _version << endl;
      cout << "SUBVERSION: ";
      int _subversion = vm["sub-version"].as<int>();
      cout << _subversion << endl;
      string _file_list = vm["file-list"].as<string>();
      HcalLutManager manager;
      manager . create_lut_loader( _file_list, _prefix, _tag, _comment, _version, _subversion );
      return 0;
    }


    if (vm.count("test-channel-data")) {
      HcalChannelQualityXml xml;
      //DOMNode * ds = xml.add_dataset();
      xml.add_hcal_channel_dataset(2, 2, 1, "HO", 1, 0, "test channel quality comment field");
      xml.write();
      return 0;
    }
    
    
    if (vm.count("test-new-developer")) {
      cout << "Wazzup, dude?! What would you like to do?.." << "\n";
      return 0;
    }
    
    
    
  } catch(boost::program_options::unknown_option) {
    cout << "No command line options known to boost... continuing to getopt parser..." << endl;
  }

  //
  // FIXME: deprecated parse command line options - switch to boost above
  //
  int c;
  int digit_optind = 0;
  
  // default parameter values
  bool luts = false;
  bool rbx = false;
  bool tag_b = false;
  bool comment_b = false;
  bool testdb_b = false;
  bool lmaptest_b = false;
  bool hardware_b = false;
  bool test_db_access_b = false;
  bool zs2_b = false;
  bool zs2HB_b = false;
  bool zs2HE_b = false;
  bool zs2HO_b = false;
  bool zs2HF_b = false;
  bool test_lut_gen_b = false;

  string filename = "";
  string path = "";
  string tag = "";
  string comment = "";
  string version = "";
  string prefix = "";
  string rbx_type = "";
  string aramsParameter = "";
  string zs2HB = "";
  string zs2HE = "";
  string zs2HO = "";
  string zs2HF = "";

  while (1) {
    int this_option_optind = optind ? optind : 1;
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

    //cout << c << endl;

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
	  //cout << "filename: " << _buf << endl;
	  filename .append( _buf );
	}
      else
	{
	  cout << "Missing file name!" << endl;
	}
      break;
      
    case 2:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //cout << "path: " << _buf << endl;
	  path .append( _buf );
	}
      else
	{
	  cout << "Empty path!" << endl;
	}
      break;
      
    case 3:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //cout << "path: " << _buf << endl;
	  tag .append( _buf );
	  tag_b = true;
	}
      else
	{
	  cout << "Empty tag!" << endl;
	}
      break;
      
    case 4:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //cout << "path: " << _buf << endl;
	  prefix .append( _buf );
	}
      else
	{
	  cout << "Empty prefix!" << endl;
	}
      break;
      
    case 5:
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  comment . append( _buf );
	  comment_b = true;
	}
      else
	{
	  cout << "Empty comment!" << endl;
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
	  cout << "Empty comment!" << endl;
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
	  cout << "RBX data type not defined!.." << endl;
	}
      break;

    case 70:
      luts = true;
      break;
      
    case 1000: // testocci
      testocci();
      break;
      
      //case 1010: // testdb
      //if ( optarg )
      //{
      //  char _buf[1024];
      //  sprintf( _buf, "%s", optarg );
      //  //cout << "filename: " << _buf << endl;
      //  filename .append( _buf );
      //  testdb_b = true;
      //}
      //else
      //{
      //  cout << "No XML file name specified! " << endl;
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
	  cout << "No parameter specified! " << endl;
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
	  cout << "No zero suppression value for HB specified... " << endl;
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
	  cout << "No zero suppression value for HE specified... " << endl;
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
	  cout << "No zero suppression value for HO specified... " << endl;
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
	  cout << "No zero suppression value for HF specified... " << endl;
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
      cout << "path: " << path << endl;
      cout << "prefix: " << prefix << endl;
      cout << "TAG_NAME: " << tag << endl;
      createLUTLoader( prefix, tag );
    }
  */
  //else
  if ( rbx )
    {
      cout << "type: " << rbx_type << endl;
      cout << "TAG_NAME: " << tag << endl;
      cout << "comment: " << comment << endl;
      cout << "version: " << version << endl;
      //cout << "list file: " << filename << endl;

      if ( tag_b ){
	createRBXLoader( rbx_type, tag, filename, comment, version );      
      }
      else{
	cout << "Tag name not specified... exiting" << endl;
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
	  cout << "No tag specified... exiting" << endl;
	  break;
	}
	if ( !zs2HB_b ){
	  cout << "No zero suppression value dor HB specified... exiting" << endl;
	  break;
	}
	if ( !zs2HE_b ){
	  cout << "No zero suppression value dor HE specified... exiting" << endl;
	  break;
	}
	if ( !zs2HO_b ){
	  cout << "No zero suppression value dor HO specified... exiting" << endl;
	  break;
	}
	if ( !zs2HF_b ){
	  cout << "No zero suppression value dor HF specified... exiting" << endl;
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
      //cout << "Nothing to do!" << endl;
    }
  

  delete XMLProcessor::getInstance();

  cout << "xmlTools ...done" << endl;
  exit (0);  
}







/* deprecated - to be removed
//
// Zero suppression Loader
int createZSLoader( void )
{
  XMLHTRZeroSuppressionLoader::loaderBaseConfig baseConf;
  XMLHTRZeroSuppressionLoader::datasetDBConfig conf;

  string _prefix = "GREN_ZS_9adc_v2";
  string _comment = "ZS for GREN07, (tune=3.5)*2+2";
  string _version = "GREN07:1";
  string _subversion = "1";

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

  for( vector<LMapRow>::const_iterator row = map_hbef . _table . begin(); row != map_hbef . _table . end(); row++ )
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

  for( vector<LMapRow>::const_iterator row = map_ho . _table . begin(); row != map_ho . _table . end(); row++ )
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
int createZSLoader2( string & tag, string & comment, string & zs2HB, string & zs2HE, string & zs2HO, string & zs2HF )
{

  XMLHTRZeroSuppressionLoader::loaderBaseConfig baseConf;
  XMLHTRZeroSuppressionLoader::datasetDBConfig conf;

  string lmap_version = "30";

  string _prefix = tag;
  string _comment = comment;
  string _version = "GRuMM_test:1";
  string _subversion = "1";

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
  string subdet;

  cout << "Preparing to request the LMAP from the database..." << endl;

  try {
    cout << "Preparing the query...";
    Statement* stmt = _connection -> createStatement();
    std::string query = ("SELECT eta, side, phi, depth, subdetector, cds.version ");
    query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V3 lmap";
    query += " join cms_hcl_core_condition_owner.cond_data_sets cds ";
    query += " on cds.condition_data_set_id=lmap.condition_data_set_id ";
    query += toolbox::toString(" WHERE version='%s'", lmap_version . c_str() );
    cout << " done" << endl;    

    //SELECT
    cout << "Executing the query...";
    ResultSet *rs = stmt->executeQuery(query.c_str());
    cout << " done" << endl;

    RooGKCounter _channels(1,100);
    _channels . setNewLine( false );

    cout << "Going through HCAL channels..." << endl;
    while (rs->next()) {
      _channels . count();
      eta_abs  = rs -> getInt(1);
      side    = rs -> getInt(2);
      phi     = rs -> getInt(3);
      depth   = rs -> getInt(4);
      subdet  = rs -> getString(5);
      
      conf . comment_description = _comment;
      conf . version = _version;
      conf . subversion = _subversion;
      conf . eta = eta_abs;
      conf . z = side;
      conf . phi = phi;
      conf . depth = depth;
      conf . detector_name = subdet;

      int _zs;
      HcalSubdetector _subdet;
      if ( subdet == "HB" ){
	_subdet = HcalBarrel;
	sscanf(zs2HB.c_str(),"%d", &_zs);
      }
      if ( subdet == "HE" ){
	_subdet = HcalEndcap;
	sscanf(zs2HE.c_str(),"%d", &_zs);
      }
      if ( subdet == "HO" ){
	_subdet = HcalOuter;
	sscanf(zs2HO.c_str(),"%d", &_zs);
      }
      if ( subdet == "HF" ){
	_subdet = HcalForward;
	sscanf(zs2HF.c_str(),"%d", &_zs);
      }
      HcalDetId _hcaldetid( _subdet, side*eta_abs, phi, depth );
      conf . hcal_channel_id = _hcaldetid . rawId();

      //conf . zero_suppression = 9;
      conf . zero_suppression = _zs;

      doc . addZS( &conf );
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
int createLUTLoader( string _prefix, string tag_name )
{
  cout << "Generating XML loader for LUTs..." << endl;
  cout << _prefix << "..." << tag_name << endl;

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

  vector<int> crate_number;
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
  vector<string> file_name;
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

int createHTRPatternLoader( void )
{
  // HTR Patterns Loader

  cout << "Generating XML loader for HTR patterns..." << endl;

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

  cout << "Generating XML loader for HTR patterns... done." << endl;

  // end of HTR Patterns Loader

  return 0;
}


int createLMap( void ){
  DBlmapWriter lw;
  lw.createLMap();
  return 0;
}


int createRBXLoader( string & type_, string & tag_, string & list_file, string & _comment, string & _version )
{
  string _prefix = "oracle_"; 
  string _tag = tag_;
  string _subversion = "1";

  std::vector<string> brickFileList;
  char filename[1024];
  string listFileName = list_file;
  ifstream inFile( listFileName . c_str(), ios::in );
  if (!inFile)
    {
      cout << " Unable to open list file" << endl;
    }
  else
    {
      cout << "List file opened successfully: " << listFileName << endl;
    }
  while (inFile >> filename)
    {
      string fullFileName = filename;
      brickFileList . push_back( fullFileName );
    }
  inFile.close();

  for ( std::vector<string>::const_iterator _file = brickFileList . begin(); _file != brickFileList . end(); _file++ )
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
	cout << "Unknown config type... exiting" << endl;
	exit(1);
      }

      _conf . version = _version;
      _conf . subversion = _subversion;
      _conf . comment_description = _comment;

      XMLRBXPedestalsLoader p( &_baseConf );

      p . addRBXSlot( &_conf, (*_file), type_ );
      p . write( (*_file) + ".oracle.xml" );
    }

  return 0;

}


int testocci( void )
{

  HCALConfigDB * db = new HCALConfigDB();
  const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );
  vector<unsigned int> _lut = db -> getOnlineLUT( "gren_P3Thr5", 17, 2, 1, 1, 0, 1 );

  cout << "LUT length = " << _lut . size() << endl;
  for ( vector<unsigned int>::const_iterator i = _lut . begin(); i != _lut . end(); i++ )
    {
      cout << (i-_lut.begin()) << "     " << _lut[(i-_lut.begin())] << endl;
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
  hcal::ConfigurationDatabase::FPGASelection _fpga;

  int side   = -1;
  int etaAbs =  1;
  int phi    =  1;
  int depth  =  1;
  string subdetector = "HB";

  cout << "version	" << "eta	" << "phi	" << "depth	" << "subdetector	";
  cout << "crate	" << "slot	" << "fiber	" << "channel	" << endl;

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
      if ( fpga_ == "top" ) _fpga = hcal::ConfigurationDatabase::Top;
      else _fpga  = hcal::ConfigurationDatabase::Bottom;
      _fiber    = rs -> getInt(5);
      _channel  = rs -> getInt(6);
      
      cout << _version << "	" << side*etaAbs << "	" << phi << "	" << depth << "	" << subdetector << "		";
      cout << _crate << "	" << _slot << "	" << _fiber << "	" << _channel << endl;
    }
    //Always terminate statement
    _connection -> terminateStatement(stmt);
  } catch (SQLException& e) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }

  db -> disconnect();
  
  return 0;
}




int lmaptest( string _param ){
  cout << "lmaptest() is running, param = " << _param << endl;

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

  std::map<string,map<string,map<string,map<int,string> > > > hw_map;

  ifstream infile("HBHOHE.ascii");
  string buf;

  if ( infile . is_open() ){
  cout << "File is open" << endl;
    while ( getline( infile, buf ) > 0 )
      {
	vector<string> _line = splitString( buf );
	//cout << _line . size() << endl;
	if ( _line[0] != "XXXXXX" && _line[1] != "XXXXXX" && _line[2] != "XXXXXX" ){
	  if (_line[3] != "XXXXXX") hw_map[_line[0]][_line[1]]["3040000000000" + _line[2]][1] = "3040000000000" + _line[3];
	  if (_line[4] != "XXXXXX") hw_map[_line[0]][_line[1]]["3040000000000" + _line[2]][2] = "3040000000000" + _line[4];
	  if (_line[5] != "XXXXXX") hw_map[_line[0]][_line[1]]["3040000000000" + _line[2]][3] = "3040000000000" + _line[5];
	}
      }
  }
  
  cout << hw_map . size() << endl;

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
  cout << _manager . getLutXml( _l );
}
