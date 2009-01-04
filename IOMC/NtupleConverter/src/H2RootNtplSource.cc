/* \author: Joanna Weng
*  $Date: 1/2006
*/

#include "IOMC/NtupleConverter/interface/NtupleROOTFile.h"
#include "IOMC/NtupleConverter/interface/H2RootNtplSource.h"
#include "IOMC/NtupleConverter/interface/Ntuple2HepMCFiller.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>


using namespace edm;
using namespace std;

//used for defaults

H2RootNtplSource::H2RootNtplSource( const ParameterSet & pset, InputSourceDescription const& desc ) :
ExternalInputSource(pset, desc),  
evt(0), firstEvent_ (pset.getUntrackedParameter<unsigned int>("firstEvent",0)),
reader_( Ntuple2HepMCFiller::instance() ){
	

	cout << "H2RootNtplSource: Reading HepMC file: " << fileNames()[0] << endl;
	string fileName = fileNames()[0];
	// strip the file: 
	if ( ! fileName.find("file:")){
	  fileName.erase(0,5);
	}   
	//Max number of events processed  
	cout << "H2RootNtplSource: Number of events to be processed = " << maxEvents() << endl;
	
	//First event
	firstEvent_ = pset.getUntrackedParameter<unsigned int>("firstEvent",0);
	cout << "H2RootNtplSource: Number of first event  = " << firstEvent_ << endl;	

	reader_->initialize(fileName,101);  
	reader_->setEvent(firstEvent_);
	produces<HepMCProduct>();

}


H2RootNtplSource::~H2RootNtplSource(){
	clear();
}

void H2RootNtplSource::clear() {
}

bool H2RootNtplSource::produce(Event & e) {
	

	// no need to clean up GenEvent memory - now done in HepMCProduct
	//if ( evt != NULL ) delete evt ;
   
		
		auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
		cout << "H2RootNtplSource: Start Reading  " << endl;
		evt = reader_->fillCurrentEventData(); 
		if(evt)  {
		  bare_product->addHepMCData(evt );
		  e.put(bare_product);
		  return true;
		}
		else return false;
	
	
}

