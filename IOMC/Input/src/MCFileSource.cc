/*
 *  $Date: 2005/10/05 04:45:33 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - S. Argiro'
 */

#include "IOMC/Input/interface/HepMCFileReader.h"
//#include <Core/DaqInput/interface/DaqFedRawDataProduct.h"
//#include <Core/DaqInput/interface/DaqRawDataProduct.h" 
#include "FWCore/Framework/src/TypeID.h" 
#include "IOMC/Input/interface/MCFileSource.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include <FWCore/EDProduct/interface/Wrapper.h>
#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/EDProduct/interface/EventID.h"


using namespace edm;
using namespace std;

//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

MCFileSource::MCFileSource( const ParameterSet & pset, InputSourceDescription const& desc ) :
  InputSource( desc ) ,  
  remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)), 
  numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents_+1)),
  presentRun_( pset.getUntrackedParameter<unsigned int>("firstRun",1)  ),
  nextTime_(pset.getUntrackedParameter<unsigned int>("firstTime",1)),  //time in ns
  timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents",kNanoSecPerSec/kAveEventPerSec) ),
  numberEventsInThisRun_(0),
  nextID_(presentRun_, 1 ),    
  reader_( HepMCFileReader::instance() ),
  filename_( pset.getParameter<std::string>( "fileName" ) ) {

  cout << "MCFileSource:Reading HepMC file: " << filename_ << endl;
  reader_->initialize(filename_);  
  ModuleDescription      modDesc_; 
  modDesc_.pid = PS_ID("MCFileSource");
  modDesc_.moduleName_ = "MCFileSource";
  modDesc_.moduleLabel_ = "MCFileInput";
  modDesc_.versionNumber_ = 1UL;
  modDesc_.processName_ = "HepMC";
  modDesc_.pass = 1UL;  
    
  branchDesc_.module = modDesc_;   
  branchDesc_.fullClassName_= "HepMCProduct";
  branchDesc_.friendlyClassName_ = "HepMCProduct";   
  preg_->addProduct(branchDesc_);
 }


MCFileSource::~MCFileSource(){
  clear();
}

void MCFileSource::clear() {
  // for(map<int, DaqFEDRawData *>::iterator it = daqevdata_.begin();
  //   it != daqevdata_.end(); ++it) {
  // delete (*it).second;
  //  }
// daqevdata_.clear();
}


auto_ptr<EventPrincipal> MCFileSource::read() {

  auto_ptr<EventPrincipal> result(0);
 

  // event loop
  if (remainingEvents_-- != 0) {
  
    result = auto_ptr<EventPrincipal>(new EventPrincipal(nextID_, Timestamp(nextTime_), *preg_));
    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
    cout << "MCFileSource: Start Reading  " << endl;
    evt = reader_->fillCurrentEventData(); 
    if(evt)  bare_product->addHepMCData(evt );
    edm::Wrapper<HepMCProduct> *wrapped_product = 
      new edm::Wrapper<HepMCProduct> (bare_product); 
    auto_ptr<EDProduct>  prod(wrapped_product);
    auto_ptr<Provenance> prov(new Provenance(branchDesc_));

    result->put(prod, prov);
    cout << "MCFileSource:Reading Done " << endl;
    if( ++numberEventsInThisRun_ < numberEventsInRun_ ) {
        nextID_ = nextID_.next();
      } else {
        nextID_ = nextID_.nextRunFirstEvent();
        numberEventsInThisRun_ = 0;
      }
    nextTime_ += timeBetweenEvents_;
  }
  return result;



}

