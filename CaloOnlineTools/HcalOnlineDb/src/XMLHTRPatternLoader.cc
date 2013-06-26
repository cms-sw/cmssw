// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLHTRPatternLoader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
// $Id: XMLHTRPatternLoader.cc,v 1.5 2013/05/23 15:17:36 gartung Exp $
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRPatternLoader.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include <cstdio>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
XMLHTRPatternLoader::loaderBaseConfig::_loaderBaseConfig() : XMLProcessor::loaderBaseConfig()
{
  extention_table_name = "HCAL_HTR_DATA_PATTERNS";
  name = "HCAL HRT data patterns";
  run_mode = "no-run";
  data_set_id = "-1";
  iov_id = "1";
  iov_begin = "0";
  iov_end = "1";
  tag_id = "2";
  tag_mode = "auto";
  tag_name = "HTR data pattern test";
  detector_name = "HCAL";
  comment_description = "empty comment, by kukarzev";
}

XMLHTRPatternLoader::datasetDBConfig::_datasetDBConfig() : XMLProcessor::DBConfig()
{
  kind_of_part = "HCAL HTR Crate";
  name_label = "CRATE17";
  htr_data_patterns_data_file = "testHTRPatterns_17.xml";
  crate = 17;
}

XMLHTRPatternLoader::checksumsDBConfig::_checksumsDBConfig() : XMLProcessor::DBConfig()
{
  comment_description = "checksum for all crates version test:2";
  name_label = "CMS-HCAL-ROOT";
  htr_data_patterns_data_file = "./testHTRPattern_checksums.xml";
  crate = -1;
}

XMLHTRPatternLoader::XMLHTRPatternLoader()
{
}

XMLHTRPatternLoader::XMLHTRPatternLoader( XMLProcessor::loaderBaseConfig * config, std::string templateBase ) : XMLDOMBlock( templateBase )
{
  setTagValue( "EXTENSION_TABLE_NAME", config -> extention_table_name );
  setTagValue( "NAME", config -> name );
  setTagAttribute( "RUN", "mode", config -> run_mode );
  setTagAttribute( "DATA_SET", "id", config -> data_set_id );
  setTagAttribute( "IOV", "id", config -> iov_id );
  setTagValue( "INTERVAL_OF_VALIDITY_BEGIN", config -> iov_begin );
  setTagValue( "INTERVAL_OF_VALIDITY_END", config -> iov_end );
  setTagAttribute( "TAG", "id", config -> tag_id );
  setTagAttribute( "TAG", "mode", config -> tag_mode );
  setTagValue( "TAG_NAME", config -> tag_name );
  setTagValue( "DETECTOR_NAME", config -> detector_name );
  setTagValue( "COMMENT_DESCRIPTION", config -> comment_description );

  setTagAttribute( "TAG", "idref", config -> tag_id, 1 );
  setTagAttribute( "IOV", "idref", config -> iov_id, 1 );
  setTagAttribute( "DATA_SET", "idref", config -> data_set_id, 1 );
}

// XMLHTRPatternLoader::XMLHTRPatternLoader(const XMLHTRPatternLoader& rhs)
// {
//    // do actual copying here;
// }

XMLHTRPatternLoader::~XMLHTRPatternLoader()
{
}

//
// assignment operators
//
// const XMLHTRPatternLoader& XMLHTRPatternLoader::operator=(const XMLHTRPatternLoader& rhs)
// {
//   //An exception safe implementation is
//   XMLHTRPatternLoader temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
int XMLHTRPatternLoader::addPattern( datasetDBConfig * config, std::string templateFileName )
{
  DOMElement * root = document -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();

  // changes to the HTRPATTERN <data_set> node
  setTagValue( "VERSION", config -> version, 0, dataSet );
  setTagValue( "SUBVERSION", config -> subversion, 0, dataSet );
  char timebuf[50];
  //strftime( timebuf, 50, "%c", gmtime( &(config -> create_timestamp) ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &(config -> create_timestamp) ) );
  setTagValue( "CREATE_TIMESTAMP", timebuf , 0, dataSet );
  setTagValue( "CREATED_BY_USER", config -> created_by_user, 0, dataSet );
  setTagValue( "KIND_OF_PART", config -> kind_of_part, 0, dataSet );
  setTagValue( "NAME_LABEL", config -> name_label, 0, dataSet );
  setTagValue( "HTR_DATA_PATTERNS_DATA_FILE", config -> htr_data_patterns_data_file, 0, dataSet );
  setTagValue( "CRATE", config -> crate, 0, dataSet );
  
  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

int XMLHTRPatternLoader::addChecksums( checksumsDBConfig * config, std::string templateFileName )
{
  DOMElement * root = document -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();

  // changes to the Checksums <data_set> node
  setTagValue( "VERSION", config -> version, 0, dataSet );
  setTagValue( "SUBVERSION", config -> subversion, 0, dataSet );
  char timebuf[50];
  //strftime( timebuf, 50, "%c", gmtime( &(config -> create_timestamp) ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &(config -> create_timestamp) ) );
  setTagValue( "CREATE_TIMESTAMP", timebuf , 0, dataSet );
  setTagValue( "CREATED_BY_USER", config -> created_by_user, 0, dataSet );
  setTagValue( "COMMENT_DESCRIPTION", config -> comment_description, 0, dataSet );
  setTagValue( "NAME_LABEL", config -> name_label, 0, dataSet );
  setTagValue( "HTR_DATA_PATTERNS_DATA_FILE", config -> htr_data_patterns_data_file, 0, dataSet );
  setTagValue( "CRATE", config -> crate, 0, dataSet );

  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

int XMLHTRPatternLoader::createLoader( const std::vector<int>& crate_number, const std::vector<std::string>& file_name )
{
  XMLHTRPatternLoader::datasetDBConfig conf;

  for ( std::vector<std::string>::const_iterator _file = file_name . begin(); _file != file_name . end(); _file++ )
    {
      conf . htr_data_patterns_data_file = *_file;
      conf . htr_data_patterns_data_file += ".dat";
      conf . crate = crate_number[ _file - file_name . begin() ];

      char _buf[128];
      sprintf( _buf, "CRATE%.2d", conf . crate );
      std::string _namelabel;
      _namelabel . append( _buf );
      conf . name_label = _namelabel;
      addPattern( &conf );
    }
  
  write( "HTRPatternLoader.xml" );

  return 0;
}

//
// const member functions
//

//
// static member functions
//
