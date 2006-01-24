/* \author: Joanna Weng
*  $Date: 1/2006
*/

#include "IOMC/NtupleConverter/interface/NtupleROOTFile.h"
#include "IOMC/NtupleConverter/interface/H2RootNtplSource.h"
#include "IOMC/NtupleConverter/interface/Ntuple2HepMCFiller.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>


using namespace edm;
using namespace std;

//used for defaults

H2RootNtplSource::H2RootNtplSource( const ParameterSet & pset, InputSourceDescription const& desc ) :
ExternalInputSource(pset, desc),  
evt(0),    
reader_(  Ntuple2HepMCFiller::instance() ){
	
	cout << "H2RootNtplSource: Reading HepMC file: " << fileNames()[0] << endl;
	reader_->initialize(fileNames()[0],101);  
	produces<HepMCProduct>();

}


H2RootNtplSource::~H2RootNtplSource(){
	clear();
}

void H2RootNtplSource::clear() {
}

bool H2RootNtplSource::produce(Event & e) {
	

	// clean up GenEvent memory : also deletes all vtx/part in it
	if ( evt != NULL ) delete evt ;
   
		
		auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
		cout << "H2RootNtplSource: Start Reading  " << endl;
		evt = reader_->fillCurrentEventData(); 
		if(evt)  bare_product->addHepMCData(evt );
		e.put(bare_product);
		return true;
	
	
}

