// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLLUTLoader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLLUTLoader.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include <cstdio>
XERCES_CPP_NAMESPACE_USE 

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
XMLLUTLoader::lutDBConfig::_lutDBConfig() : XMLProcessor::DBConfig()
{
  kind_of_part = "HCAL HTR Crate";
  name_label = "CRATE17";
  trig_prim_lookuptbl_data_file = "testLUT_17.xml";
  crate = 17;
}

XMLLUTLoader::checksumsDBConfig::_checksumsDBConfig() : XMLProcessor::DBConfig()
{
  comment_description = "checksum for all crates version test:2";
  name_label = "CMS-HCAL-ROOT";
  trig_prim_lookuptbl_data_file = "./testLUT_checksums.xml";
  crate = -1;
}

XMLLUTLoader::XMLLUTLoader()
{
}

XMLLUTLoader::XMLLUTLoader( XMLProcessor::loaderBaseConfig * config, std::string templateBase ) : XMLDOMBlock( templateBase )
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

// XMLLUTLoader::XMLLUTLoader(const XMLLUTLoader& rhs)
// {
//    // do actual copying here;
// }

XMLLUTLoader::~XMLLUTLoader()
{
}

//
// assignment operators
//
// const XMLLUTLoader& XMLLUTLoader::operator=(const XMLLUTLoader& rhs)
// {
//   //An exception safe implementation is
//   XMLLUTLoader temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
int XMLLUTLoader::addLUT( lutDBConfig * config, std::string templateFileName )
{
  DOMElement * root = document -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();

  // changes to the LUT <data_set> node
  setTagValue( "VERSION", config -> version, 0, dataSet );
  setTagValue( "SUBVERSION", config -> subversion, 0, dataSet );
  char timebuf[50];
  //strftime( timebuf, 50, "%c", gmtime( &(config -> create_timestamp) ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &(config -> create_timestamp) ) );
  setTagValue( "CREATE_TIMESTAMP", timebuf , 0, dataSet );
  setTagValue( "CREATED_BY_USER", config -> created_by_user, 0, dataSet );
  setTagValue( "KIND_OF_PART", config -> kind_of_part, 0, dataSet );
  setTagValue( "NAME_LABEL", config -> name_label, 0, dataSet );
  setTagValue( "TRIG_PRIM_LOOKUPTBL_DATA_FILE", config -> trig_prim_lookuptbl_data_file, 0, dataSet );
  setTagValue( "CRATE", config -> crate, 0, dataSet );
  
  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

int XMLLUTLoader::addChecksums( checksumsDBConfig * config, std::string templateFileName )
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
  setTagValue( "TRIG_PRIM_LOOKUPTBL_DATA_FILE", config -> trig_prim_lookuptbl_data_file, 0, dataSet );
  setTagValue( "CRATE", config -> crate, 0, dataSet );

  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

int XMLLUTLoader::createLoader( const std::vector<int>& crate_number, const std::vector<std::string>& file_name )
{
  XMLLUTLoader::lutDBConfig conf;
  XMLLUTLoader::checksumsDBConfig CSconf;

  for ( std::vector<std::string>::const_iterator _file = file_name . begin(); _file != file_name . end(); _file++ )
    {
      conf . trig_prim_lookuptbl_data_file = *_file;
      conf . trig_prim_lookuptbl_data_file += ".dat";
      conf . crate = crate_number[ _file - file_name . begin() ];

      char _buf[128];
      sprintf( _buf, "CRATE%.2d", conf . crate );
      std::string _namelabel;
      _namelabel . append( _buf );
      conf . name_label = _namelabel;
      addLUT( &conf );
    }
  
  CSconf . trig_prim_lookuptbl_data_file += ".dat";
  addChecksums( &CSconf );
  write( "LUTLoader.xml" );

  return 0;
}

//
// const member functions
//

//
// static member functions
//
