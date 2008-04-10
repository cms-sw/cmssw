#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"

using namespace std;

/**

   \class HcalLutManager
   \brief Various manipulations with trigger Lookup Tables
   \author Gena Kukartsev, Brown University, March 14, 2008

*/

HcalLutManager::HcalLutManager( void )
{    
  init();
}



void HcalLutManager::init( void )
{    
  lut_xml = NULL;
}



HcalLutManager::~HcalLutManager( void )
{    
  if (lut_xml) delete lut_xml;
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
