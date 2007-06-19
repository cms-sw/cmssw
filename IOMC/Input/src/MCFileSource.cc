// $Id: MCFileSource.cc,v 1.9 2007/05/29 21:00:00 weng Exp $

/**  
*  See header file for a description of this class.
*
*
*  $Date: 2007/05/29 21:00:00 $
*  $Revision: 1.9 $
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
<<<<<<< MCFileSource.cc
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

=======
>>>>>>> 1.9


using namespace edm;
using namespace std;
<<<<<<< MCFileSource.cc
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
=======


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
>>>>>>> 1.9
  
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
