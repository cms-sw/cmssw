// $Id:$

/**  
*  See header file for a description of this class.
*
*
*  $Date: 2006/04/19 14:52:03 $
*  $Revision: 1.8 $
*  \author Jo. Weng  - CERN, Ph Division & Uni Karlsruhe
*  \author F.Moortgat - CERN, Ph Division
*/

#include <iostream>
#include <string>


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOMC/Input/interface/HepMCFileReader.h" 
#include "IOMC/Input/interface/MCFileSource.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"


using namespace edm;
using namespace std;


//-------------------------------------------------------------------------
MCFileSource::MCFileSource(const ParameterSet & pset, InputSourceDescription const& desc) :
  ExternalInputSource(pset, desc),
  reader_(HepMCFileReader::instance()), evt_(0),
  useExtendedAscii_(pset.getUntrackedParameter<bool>("useExtendedAscii",false))
{
  edm::LogInfo("MCFileSource") << "Reading HepMC file:" << fileNames()[0];
  string fileName = fileNames()[0];
  // strip the file: 
  if (fileName.find("file:") == 0){
    fileName.erase(0,5);
  }  
  
  reader_->initialize(fileName, useExtendedAscii_);  
  produces<HepMCProduct>();
}


//-------------------------------------------------------------------------
MCFileSource::~MCFileSource(){
}


//-------------------------------------------------------------------------
bool MCFileSource::produce(Event &e) {
  // Read one HepMC event and store it in the Event.

  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  

  edm::LogInfo("MCFileSource") << "Start Reading";
  evt_ = reader_->fillCurrentEventData(); 
  if (evt_ == 0) return false;

  bare_product->addHepMCData(evt_);
  e.put(bare_product);

  return true;
}
