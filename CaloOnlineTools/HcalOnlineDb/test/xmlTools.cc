#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <fstream>

#include <sys/time.h>

//#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLLUTLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatterns.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatternLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRZeroSuppressionLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMapLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLRBXPedestalsLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HCALConfigDB.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplOracle.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"

using namespace std;

int createLUTLoader( string prefix_="", string tag_="" );
int createHTRPatternLoader( void );
int createLMap( void );
int createRBXLoader( string type_, string tag_, string list_file );
//int createRBXentries( string );
int createZSLoader( void );
int testocci( void );
int testDB( string _tag, string _filename );

int main( int argc, char **argv )
{
  cout << "Running xmlTools..." << endl;
  
  // parse command line options
  int c;
  int digit_optind = 0;
  
  // default parameter values
  bool luts = false;
  bool rbx = false;
  bool tag_b = false;
  bool testdb_b = false;

  string filename = "";
  string path = "";
  string tag = "";
  string prefix = "";
  string rbx_type = "";

  while (1) {
    int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
      {"filename", 1, 0, 1},
      {"path", 1, 0, 2},
      {"tag", 1, 0, 3},
      {"prefix", 1, 0, 4},
      {"luts", 0, 0, 10},
      {"patterns", 0, 0, 20},
      {"lmap", 0, 0, 30},
      {"rbxpedestals", 0, 0, 40},
      {"zs", 0, 0, 50},
      {"rbx", 1, 0, 60},
      {"luts2", 0, 0, 70},
      {"testocci", 0, 0, 1000},
      {"testdb", 1, 0, 1010},
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
      
    case 10:
      createLUTLoader();
      break;
      
    case 70:
      luts = true;
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
      
    case 50:
      createZSLoader();
      break;
      
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

    case 1000: // testocci
      testocci();
      break;
      
    case 1010: // testdb
      if ( optarg )
	{
	  char _buf[1024];
	  sprintf( _buf, "%s", optarg );
	  //cout << "filename: " << _buf << endl;
	  filename .append( _buf );
	  testdb_b = true;
	}
      else
	{
	  cout << "No XML file name specified! " << endl;
	}
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

  // decide what to depending on the params
  if ( luts )
    {
      cout << "path: " << path << endl;
      cout << "prefix: " << prefix << endl;
      cout << "TAG_NAME: " << tag << endl;
      createLUTLoader( prefix, tag );
    }
  else if ( rbx )
    {
      cout << "type: " << rbx_type << endl;
      cout << "TAG_NAME: " << tag << endl;
      cout << "list file: " << filename << endl;
      
      createRBXLoader( rbx_type, tag, filename );      

      if ( tag . size() < 1 ) cout << "===> WARNING: tag field is empty!" << endl;
    }
  else if ( testdb_b && tag_b )
    {
      testDB( tag, filename );      
    }
  else
    {
      //cout << "Nothing to do!" << endl;
    }
  
  cout << "xmlTools ...done" << endl;

  exit (0);
  
}








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

// LUT Loader
int createLUTLoader( string _prefix, string tag_name )
{
  cout << "Generating XML loader for LUTs..." << endl;

  XMLLUTLoader::loaderBaseConfig baseConf;
  XMLLUTLoader::lutDBConfig conf;
  XMLLUTLoader::checksumsDBConfig CSconf;

  //_prefix = "GREN_olddetid_fake_p3_thr9_all";
  //tag_name = "GREN_olddetid_fake_p3_thr9_all";

  baseConf . tag_name = tag_name;
  //baseConf . comment_description = _prefix + ": LUTs for GREN 26Nov2007";
  baseConf . comment_description = tag_name;
  baseConf . iov_begin = "1";
  baseConf . iov_end = "-1";

  //conf . version = _prefix + ":1";
  conf . version = "GREN2007:1";
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
  crate_number . push_back(4);
  crate_number . push_back(5);
  crate_number . push_back(9);
  crate_number . push_back(10);
  crate_number . push_back(11);
  crate_number . push_back(12);
  crate_number . push_back(14);
  crate_number . push_back(15);
  crate_number . push_back(17);
  vector<string> file_name;
  file_name . push_back( "./" + _prefix + "_0.xml.dat" );
  file_name . push_back( "./" + _prefix + "_1.xml.dat" );
  file_name . push_back( "./" + _prefix + "_2.xml.dat" );
  file_name . push_back( "./" + _prefix + "_4.xml.dat" );
  file_name . push_back( "./" + _prefix + "_5.xml.dat" );
  file_name . push_back( "./" + _prefix + "_9.xml.dat" );
  file_name . push_back( "./" + _prefix + "_10.xml.dat" );
  file_name . push_back( "./" + _prefix + "_11.xml.dat" );
  file_name . push_back( "./" + _prefix + "_12.xml.dat" );
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

}


int createLMap( void ){

  cout << "XML Test..." << endl;
  
  //XMLProcessor * theProcessor = XMLProcessor::getInstance();

  //XMLDOMBlock * doc = theProcessor -> createLMapHBEFXMLBase( "FullLmapBase.xml" );
  LMapLoader doc;

  LMapLoader::LMapRowHBEF aRow;
  LMapLoader::LMapRowHO bRow;


  ifstream fcha("/afs/fnal.gov/files/home/room3/avetisya/public_html/HCAL/Maps/HCALmapHBEF_11.9.2007v2.txt");
  ifstream fcho("/afs/fnal.gov/files/home/room3/avetisya/public_html/HCAL/Maps/HCALmapHO_11.9.2007v2.txt");

  //List in order HB HE HF
  //side     eta     phi     dphi       depth      det
  //rbx      wedge   rm      pixel      qie   
  //adc      rm_fi   fi_ch   crate      htr          
  //fpga     htr_fi  dcc_sl  spigo      dcc 
  //slb      slbin   slbin2  slnam      rctcra     rctcar
  //rctcon   rctnam  fedid

  const int NCHA = 6912;
  const int NCHO = 2160;
  int ncho = 0;
  int i, j;
  string ndump;

  int sideC[NCHA], etaC[NCHA], phiC[NCHA], dphiC[NCHA], depthC[NCHA], wedgeC[NCHA], crateC[NCHA], rmC[NCHA], rm_fiC[NCHA], htrC[NCHA];
  int htr_fiC[NCHA], fi_chC[NCHA], spigoC[NCHA], dccC[NCHA], dcc_slC[NCHA], fedidC[NCHA], pixelC[NCHA], qieC[NCHA], adcC[NCHA];
  int slbC[NCHA], rctcraC[NCHA], rctcarC[NCHA], rctconC[NCHA];
  string detC[NCHA], rbxC[NCHA], fpgaC[NCHA], slbinC[NCHA], slbin2C[NCHA], slnamC[NCHA], rctnamC[NCHA];

  int sideO[NCHO], etaO[NCHO], phiO[NCHO], dphiO[NCHO], depthO[NCHO], sectorO[NCHO], crateO[NCHO], rmO[NCHO], rm_fiO[NCHO], htrO[NCHO];
  int htr_fiO[NCHO], fi_chO[NCHO], spigoO[NCHO], dccO[NCHO], dcc_slO[NCHO], fedidO[NCHO], pixelO[NCHO], qieO[NCHO], adcO[NCHO];
  string detO[NCHO], rbxO[NCHO], fpgaO[NCHO], let_codeO[NCHO];

  int counter = 0;
  for (i = 0; i < NCHA; i++){
    if(i == counter){
      for (j = 0; j < 31; j++){
	fcha>>ndump;
	ndump = "";
      }
      counter += 21;
    }
    fcha>>sideC[i];
    fcha>>etaC[i]>>phiC[i]>>dphiC[i]>>depthC[i]>>detC[i];
    fcha>>rbxC[i]>>wedgeC[i]>>rmC[i]>>pixelC[i]>>qieC[i];
    fcha>>adcC[i]>>rm_fiC[i]>>fi_chC[i]>>crateC[i]>>htrC[i];
    fcha>>fpgaC[i]>>htr_fiC[i]>>dcc_slC[i]>>spigoC[i]>>dccC[i];
    fcha>>slbC[i]>>slbinC[i]>>slbin2C[i]>>slnamC[i]>>rctcraC[i]>>rctcarC[i];
    fcha>>rctconC[i]>>rctnamC[i]>>fedidC[i];
  }
    
  for(i = 0; i < NCHA; i++){
    aRow . side   = sideC[i];
    aRow . eta    = etaC[i];
    aRow . phi    = phiC[i];
    aRow . dphi   = dphiC[i];
    aRow . depth  = depthC[i];
    aRow . det    = detC[i];
    aRow . rbx    = rbxC[i];
    aRow . wedge  = wedgeC[i];
    aRow . rm     = rmC[i];
    aRow . pixel  = pixelC[i];
    aRow . qie    = qieC[i];
    aRow . adc    = adcC[i];
    aRow . rm_fi  = rm_fiC[i];
    aRow . fi_ch  = fi_chC[i];
    aRow . crate  = crateC[i];
    aRow . htr    = htrC[i];
    aRow . fpga   = fpgaC[i];
    aRow . htr_fi = htr_fiC[i];
    aRow . dcc_sl = dcc_slC[i];
    aRow . spigo  = spigoC[i];
    aRow . dcc    = dccC[i];
    aRow . slb    = slbC[i];
    aRow . slbin  = slbinC[i];
    aRow . slbin2 = slbin2C[i];
    aRow . slnam  = slnamC[i];
    aRow . rctcra = rctcraC[i];
    aRow . rctcar = rctcarC[i];
    aRow . rctcon = rctconC[i];
    aRow . rctnam = rctnamC[i];
    aRow . fedid  = fedidC[i];
    
    doc . addLMapHBEFDataset( &aRow, "FullHCALDataset.xml" );
  }

  counter = 0;
  for (i = 0; i < NCHO; i++){
    if(i == counter){
      for (j = 0; j < 24; j++){
	fcho>>ndump;
	ndump = "";
      }
      counter += 21;
    }
    fcho>>sideO[i];
    if (sideO[i] != 1 && sideO[i] != -1){
      cerr<<ncho<<'\t'<<sideO[i]<<endl;
      break;
    }
    fcho>>etaO[i]>>phiO[i]>>dphiO[i]>>depthO[i]>>detO[i];
    fcho>>rbxO[i]>>sectorO[i]>>rmO[i]>>pixelO[i]>>qieO[i];
    fcho>>adcO[i]>>rm_fiO[i]>>fi_chO[i]>>let_codeO[i]>>crateO[i]>>htrO[i];
    fcho>>fpgaO[i]>>htr_fiO[i]>>dcc_slO[i]>>spigoO[i]>>dccO[i];
    fcho>>fedidO[i];

    ncho++;
  }
    
  for(i = 0; i < NCHO; i++){
    bRow . sideO     = sideO[i];
    bRow . etaO      = etaO[i];
    bRow . phiO      = phiO[i];
    bRow . dphiO     = dphiO[i];
    bRow . depthO    = depthO[i];

    bRow . detO      = detO[i];
    bRow . rbxO      = rbxO[i];
    bRow . sectorO   = sectorO[i];
    bRow . rmO       = rmO[i];
    bRow . pixelO    = pixelO[i];
  
    bRow . qieO      = qieO[i];
    bRow . adcO      = adcO[i];
    bRow . rm_fiO    = rm_fiO[i];
    bRow . fi_chO    = fi_chO[i];
    bRow . let_codeO = let_codeO[i];

    bRow . crateO    = crateO[i];
    bRow . htrO      = htrO[i];
    bRow . fpgaO     = fpgaO[i];
    bRow . htr_fiO   = htr_fiO[i];
    bRow . dcc_slO   = dcc_slO[i];

    bRow . spigoO    = spigoO[i]; 
    bRow . dccO      = dccO[i];
    bRow . fedidO    = fedidO[i];
    
    doc . addLMapHODataset( &bRow, "FullHCALDataset.xml" );

  }
  
  doc . write( "FullHCALmap.xml" );


  cout << "XML Test...done" << endl;
}


int createRBXLoader( string type_, string tag_, string list_file )
{
  string _prefix = "oracle_"; 
  string _comment = "RBX pedestals for GREN 2007"; 
  //string _tag = "AllHCALGRENpartitionPeds4"; // may be overriden by the tag from the brickset
  string _tag = tag_;
  string _version = "GREN2007:1";
  string _subversion = "1";

  std::vector<string> brickFileList;
  char filename[1024];
  //string listFileName = "rbx_ped.list";
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

      _conf . version = _version;
      _conf . subversion = _subversion;
      _conf . comment_description = _comment;

      XMLRBXPedestalsLoader p( &_baseConf );

      p . addRBXSlot( &_conf, (*_file) );
      //p . write( _prefix + (*_file) );
      p . write( (*_file) + ".oracle.xml" );
    }

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

int testDB( string _tag, string _filename )
{

  HCALConfigDB * db = new HCALConfigDB();
  db -> connect( _filename, "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22" );

  //vector<unsigned int> _lut = db -> getOnlineLUTFromXML( "emap_hcal_emulator_test_luts", 17, 2, 1, 1, 0, 1 );
  //vector<unsigned int> _lut = db -> getOnlineLUT( _tag, 17, 2, 1, 1, 0, 1 );

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

  cout << "LUT length = " << _lut . size() << endl;
  for ( vector<unsigned int>::const_iterator i = _lut . end() - 1; i != _lut . begin()-1; i-- )
    {
      cout << (i-_lut.begin()) << "     " << _lut[(i-_lut.begin())] << endl;
      break;
    }

  db -> disconnect();
  

  return 0;
}

