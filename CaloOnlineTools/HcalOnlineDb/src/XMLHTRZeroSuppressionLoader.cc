// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     XMLHTRZeroSuppressionLoader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLHTRZeroSuppressionLoader.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"

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
XMLHTRZeroSuppressionLoader::loaderBaseConfig::_loaderBaseConfig()
{
  extention_table_name = "HCAL_ZERO_SUPPRESSION_TYPE01";
  name = "HCAL zero suppression [type 1]";
  run_type = "hcal-test";
  run_number = 1;
  run_begin_timestamp = 1194394284;
  comment_description = "Fake set of ZS by kukartsev";
  data_set_id = "-1";
  iov_id = "1";
  iov_begin = 1;
  iov_end = -1;
  tag_id = "2";
  tag_mode = "auto";
  tag_name = "test ZS kukartsev";
  detector_name = "HCAL";
  elements_comment_description = "Fake ZS for testing (kukartsev)";
}

XMLHTRZeroSuppressionLoader::datasetDBConfig::_datasetDBConfig() : XMLProcessor::DBConfig()
{
    comment_description = "Default ZS comment description";
    //extention_table_name = "HCAL_CHANNELS";
    extention_table_name = "HCAL_CHANNELS_OLD_V01";
    eta = 1;
    phi = 1;
    depth = 1;
    z = -1;
    hcal_channel_id = 1107312769;
    std::string detector_name = "HB";
    zero_suppression = 2;
}

XMLHTRZeroSuppressionLoader::XMLHTRZeroSuppressionLoader( loaderBaseConfig * config, std::string templateBase ) : XMLDOMBlock( templateBase )
{
  setTagValue( "EXTENSION_TABLE_NAME", config -> extention_table_name );
  setTagValue( "NAME", config -> name );
  setTagValue( "RUN_TYPE", config -> run_type );
  setTagValue( "RUN_NUMBER", config -> run_number );
  char timebuf[50];
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &(config -> run_begin_timestamp) ) );
  setTagValue( "COMMENT_DESCRIPTION", config -> comment_description );
  setTagAttribute( "DATA_SET", "id", config -> data_set_id );
  setTagAttribute( "IOV", "id", config -> iov_id );
  setTagValue( "INTERVAL_OF_VALIDITY_BEGIN", config -> iov_begin );
  setTagValue( "INTERVAL_OF_VALIDITY_END", config -> iov_end );
  setTagAttribute( "TAG", "id", config -> tag_id );
  setTagAttribute( "TAG", "mode", config -> tag_mode );
  setTagValue( "TAG_NAME", config -> tag_name );
  setTagValue( "DETECTOR_NAME", config -> detector_name );
  setTagValue( "COMMENT_DESCRIPTION", config -> elements_comment_description, 1 );

  setTagAttribute( "TAG", "idref", config -> tag_id, 1 );
  setTagAttribute( "IOV", "idref", config -> iov_id, 1 );
  setTagAttribute( "DATA_SET", "idref", config -> data_set_id, 1 );
}

// XMLHTRZeroSuppressionLoader::XMLHTRZeroSuppressionLoader(const XMLHTRZeroSuppressionLoader& rhs)
// {
//    // do actual copying here;
// }

XMLHTRZeroSuppressionLoader::~XMLHTRZeroSuppressionLoader()
{
}

//
// assignment operators
//
// const XMLHTRZeroSuppressionLoader& XMLHTRZeroSuppressionLoader::operator=(const XMLHTRZeroSuppressionLoader& rhs)
// {
//   //An exception safe implementation is
//   XMLHTRZeroSuppressionLoader temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
int XMLHTRZeroSuppressionLoader::addZS( datasetDBConfig * config, std::string templateFileName )
{
  DOMElement * root = document -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();

  // changes to the HTR Zero Suppression <data_set> node
  setTagValue( "VERSION", config -> version, 0, dataSet );
  setTagValue( "SUBVERSION", config -> subversion, 0, dataSet );
  char timebuf[50];
  //strftime( timebuf, 50, "%c", gmtime( &(config -> create_timestamp) ) );
  strftime( timebuf, 50, "%Y-%m-%d %H:%M:%S.0", gmtime( &(config -> create_timestamp) ) );
  setTagValue( "CREATE_TIMESTAMP", timebuf , 0, dataSet );

  setTagValue( "CREATED_BY_USER", config -> created_by_user, 0, dataSet );
  setTagValue( "COMMENT_DESCRIPTION", config -> comment_description, 0, dataSet );
  setTagValue( "EXTENSION_TABLE_NAME", config -> extention_table_name, 0, dataSet );

  setTagValue( "ETA", config -> eta, 0, dataSet );
  setTagValue( "PHI", config -> phi, 0, dataSet );
  setTagValue( "DEPTH", config -> depth, 0, dataSet );
  setTagValue( "Z", config -> z, 0, dataSet );
  setTagValue( "HCAL_CHANNEL_ID", config -> hcal_channel_id, 0, dataSet );
  setTagValue( "DETECTOR_NAME", config -> detector_name, 0, dataSet );
  setTagValue( "ZERO_SUPPRESSION", config -> zero_suppression, 0, dataSet );

  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = document -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}






/* deprecated - to be removed
int XMLHTRZeroSuppressionLoader::createLoader( void )
{
  XMLHTRZeroSuppressionLoader::datasetDBConfig conf;

  LMap map_hbef;
  map_hbef . read( "HCALmapHBEF_11.9.2007.txt", "HBEF" ); // HBEF logical map

  for( std::vector<LMapRow>::const_iterator row = map_hbef . _table . begin(); row != map_hbef . _table . end(); row++ )
    {
      conf . comment_description = "kukarzev: test, fixed eta side";
      conf . subversion = "3";
      conf . eta = (row -> eta);
      conf . z = (row -> side);
      conf . phi = row -> phi;
      conf . depth = row -> depth;
      conf . detector_name = row -> det;

      conf . zero_suppression = 2;

      addZS( &conf );
    }
  
  LMap map_ho;
  map_ho . read( "HCALmapHO_11.9.2007.txt", "HO" ); // HO logical map

  for( std::vector<LMapRow>::const_iterator row = map_ho . _table . begin(); row != map_ho . _table . end(); row++ )
    {
      conf . comment_description = "kukarzev: test, fixed eta side";
      conf . subversion = "3";
      conf . eta = (row -> eta);
      conf . z = (row -> side);
      conf . phi = row -> phi;
      conf . depth = row -> depth;
      conf . detector_name = row -> det;

      conf . zero_suppression = 2;

      addZS( &conf );
    }
  
  write( "XMLHTRZeroSuppressionLoader.xml" );

  return 0;
}

*/
