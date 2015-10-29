// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     LMapLoader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Aram Avetisyan, avetisya@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
//

// system include files

// user include files
#include "CaloOnlineTools/HcalOnlineDb/interface/LMapLoader.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
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


LMapLoader::LMapLoader() : XMLDOMBlock( "FullLmapBase.xml" )
{
  createLMapHBEFXMLBase( );
}

LMapLoader::LMapLoader( std::string templateLoaderBase ) : XMLDOMBlock( templateLoaderBase )
{
  createLMapHBEFXMLBase( );
}

int LMapLoader::createLMapHBEFXMLBase( void ){
  document -> getElementsByTagName( XMLProcessor::_toXMLCh( "NAME" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( "HCAL hardware logical channel maps v3" ) );

  return 0;
}

// LMapLoader::LMapLoader(const LMapLoader& rhs)
// {
//    // do actual copying here;
// }

LMapLoader::~LMapLoader()
{
}

//
// assignment operators
//
// const LMapLoader& LMapLoader::operator=(const LMapLoader& rhs)
// {
//   //An exception safe implementation is
//   LMapLoader temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

int LMapLoader::addLMapHBEFDataset( LMapRowHBEF * row, std::string templateFileName )
{
  DOMDocument * loader = document;
  //DOMElement * root = loader -> getDocumentElement();
  DOMElement * root = (DOMElement *)(loader -> getElementsByTagName( XMLProcessor::_toXMLCh( "DATA_SET" ) ) -> item(2));

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();
  
  //Dataset
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SIDE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> side ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "ETA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  eta) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> phi ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DELTA_PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dphi ) );  

  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DEPTH" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> depth  ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SUBDETECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> det ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RBX" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rbx ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "WEDGE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> wedge ) );
 
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RM_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rm  ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HPD_PIXEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> pixel ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "QIE_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> qie ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "ADC" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> adc ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RM_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rm_fi ) );

  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "FIBER_CHANNEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> fi_ch));
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "CRATE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> crate ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HTR_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  htr ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HTR_FPGA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> fpga ) );

  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HTR_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> htr_fi ));  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DCC_SL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dcc_sl ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SPIGOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  spigo ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DCC_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dcc ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SLB_SITE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> slb ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SLB_CHANNEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> slbin )); 
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SLB_CHANNEL2" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> slbin2));

  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SLB_CABLE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> slnam ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RCT_CRATE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rctcra ) ); 
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RCT_CARD" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rctcar ) );
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RCT_CONNECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->rctcon));
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RCT_NAME" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rctnam ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "FED_ID" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> fedid ) );
  
      
  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = loader -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}

int LMapLoader::addLMapHODataset( LMapRowHO * row, std::string templateFileName )
{
  DOMDocument * loader = document;
  //DOMElement * root = loader -> getDocumentElement();
  DOMElement * root = (DOMElement *)(loader -> getElementsByTagName( XMLProcessor::_toXMLCh( "DATA_SET" ) ) -> item(2));

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();
  
  //Dataset
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SIDE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> sideO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "ETA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  etaO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> phiO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DELTA_PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dphiO ) );
  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DEPTH" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> depthO  ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SUBDETECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> detO ) ); 
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RBX" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rbxO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> sectorO ) );
 
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RM_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rmO  ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HPD_PIXEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> pixelO ) ); 
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "QIE_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> qieO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "ADC" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> adcO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RM_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rm_fiO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "FIBER_CHANNEL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->fi_chO));

  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "LETTER_CODE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh(row ->let_codeO));
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "CRATE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> crateO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HTR_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  htrO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HTR_FPGA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> fpgaO ) );

  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "HTR_FIBER" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> htr_fiO ) );
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DCC_SL" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dcc_slO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SPIGOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  spigoO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DCC_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dccO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "FED_ID" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> fedidO ) );
  
      
  // copy the <data_set> node into the final XML
  DOMNode * cloneDataSet = loader -> importNode( dataSet -> getDocumentElement(), true );
  root -> appendChild( cloneDataSet );

  return 0;
}


//
// const member functions
//

//
// static member functions
//
