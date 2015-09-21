#include <fstream>

#include "CaloOnlineTools/HcalOnlineDb/interface/LMapLoader.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/DBlmapWriter.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

XERCES_CPP_NAMESPACE_USE 

DBlmapWriter::DBlmapWriter(){
  
}

DBlmapWriter::~DBlmapWriter(){
  
}

XMLDOMBlock * DBlmapWriter::createLMapHBEFXMLBase( std::string templateFileName )
{
  XMLDOMBlock * result = new XMLDOMBlock( templateFileName );
  DOMDocument * loader = result -> getDocument();
  //DOMElement * root = loader -> getDocumentElement();

  loader -> getElementsByTagName( XMLProcessor::_toXMLCh( "NAME" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( "HCAL LMAP for HB, HE, HF" ) );
  //DOMElement * _tag = (DOMElement *)(loader -> getElementsByTagName( XMLProcessor::_toXMLCh( "TAG" ) ) -> item(0));
  //_tag -> setAttribute( XMLProcessor::_toXMLCh("mode"), XMLProcessor::_toXMLCh("test_mode") );

  return result;
}



int DBlmapWriter::addLMapHBEFDataset( XMLDOMBlock * doc, LMapRowHBEF * row, std::string templateFileName )
{
  DOMDocument * loader = doc -> getDocument();
  DOMElement * root = loader -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();
  
  //Dataset
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SIDE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> side ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "ETA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  eta) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> phi ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DELTA_PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dphi ) );  

  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DEPTH" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> depth  ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SUBDETECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> det ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RBX_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rbx ) );  
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

XMLDOMBlock * DBlmapWriter::createLMapHOXMLBase( std::string templateFileName )
{
  XMLDOMBlock * result = new XMLDOMBlock( templateFileName );
  DOMDocument * loader = result -> getDocument();
  //DOMElement * root = loader -> getDocumentElement();

  loader -> getElementsByTagName( XMLProcessor::_toXMLCh( "NAME" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( "HCAL LMAP for HO" ) );
  //DOMElement * _tag = (DOMElement *)(loader -> getElementsByTagName( XMLProcessor::_toXMLCh( "TAG" ) ) -> item(0));
  //_tag -> setAttribute( XMLProcessor::_toXMLCh("mode"), XMLProcessor::_toXMLCh("test_mode") );

  return result;
}


int DBlmapWriter::addLMapHODataset( XMLDOMBlock * doc, LMapRowHO * row, std::string templateFileName )
{
  DOMDocument * loader = doc -> getDocument();
  DOMElement * root = loader -> getDocumentElement();

  XMLDOMBlock dataSetDoc( templateFileName );
  DOMDocument * dataSet = dataSetDoc . getDocument();
  
  //Dataset
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SIDE" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> sideO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "ETA" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row ->  etaO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> phiO ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DELTA_PHI" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> dphiO ) );
  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "DEPTH" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> depthO  ) );  
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "SUBDETECTOR" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> detO ) ); 
  dataSet -> getElementsByTagName( XMLProcessor::_toXMLCh( "RBX_SLOT" ) ) -> item(0) -> getFirstChild() -> setNodeValue( XMLProcessor::_toXMLCh( row -> rbxO ) );  
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



int DBlmapWriter::createLMap( void ){
  std::cout << "XML Test..." << std::endl;
  
  //XMLProcessor * theProcessor = XMLProcessor::getInstance();

  //XMLDOMBlock * doc = theProcessor -> createLMapHBEFXMLBase( "FullLmapBase.xml" );
  LMapLoader doc;

  LMapLoader::LMapRowHBEF aRow;
  LMapLoader::LMapRowHO bRow;


  std::ifstream fcha("/afs/fnal.gov/files/home/room3/avetisya/public_html/HCAL/Maps/HCALmapHBEF_Feb.21.2008.txt");
  std::ifstream fcho("/afs/fnal.gov/files/home/room3/avetisya/public_html/HCAL/Maps/HCALmapHO_Feb.21.2008.txt");

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
  std::string ndump;

  int sideC[NCHA], etaC[NCHA], phiC[NCHA], dphiC[NCHA], depthC[NCHA], wedgeC[NCHA], crateC[NCHA], rmC[NCHA], rm_fiC[NCHA], htrC[NCHA];
  int htr_fiC[NCHA], fi_chC[NCHA], spigoC[NCHA], dccC[NCHA], dcc_slC[NCHA], fedidC[NCHA], pixelC[NCHA], qieC[NCHA], adcC[NCHA];
  int slbC[NCHA], rctcraC[NCHA], rctcarC[NCHA], rctconC[NCHA];
  std::string detC[NCHA], rbxC[NCHA], fpgaC[NCHA], slbinC[NCHA], slbin2C[NCHA], slnamC[NCHA], rctnamC[NCHA];

  int sideO[NCHO], etaO[NCHO], phiO[NCHO], dphiO[NCHO], depthO[NCHO], sectorO[NCHO], crateO[NCHO], rmO[NCHO], rm_fiO[NCHO], htrO[NCHO];
  int htr_fiO[NCHO], fi_chO[NCHO], spigoO[NCHO], dccO[NCHO], dcc_slO[NCHO], fedidO[NCHO], pixelO[NCHO], qieO[NCHO], adcO[NCHO];
  int geoO[NCHO], blockO[NCHO], lcO[NCHO];
  std::string detO[NCHO], rbxO[NCHO], fpgaO[NCHO], let_codeO[NCHO];

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
      for (j = 0; j < 27; j++){
	fcho>>ndump;
	ndump = "";
      }
      counter += 21;
    }
    fcho>>sideO[i];
    if (sideO[i] != 1 && sideO[i] != -1){
      std::cerr<<ncho<<'\t'<<sideO[i]<<std::endl;
      break;
    }
    fcho>>etaO[i]>>phiO[i]>>dphiO[i]>>depthO[i]>>detO[i];
    fcho>>rbxO[i]>>sectorO[i]>>rmO[i]>>pixelO[i]>>qieO[i];
    fcho>>adcO[i]>>rm_fiO[i]>>fi_chO[i]>>let_codeO[i]>>crateO[i]>>htrO[i];
    fcho>>fpgaO[i]>>htr_fiO[i]>>dcc_slO[i]>>spigoO[i]>>dccO[i];
    fcho>>fedidO[i]>>geoO[i]>>blockO[i]>>lcO[i];

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


  std::cout << "XML Test...done" << std::endl;

  return 0;
}
