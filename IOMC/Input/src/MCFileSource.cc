/**  
*  See header file for a description of this class.
*
*
*  $Date: 2005/12/06 14:09:52 $
*  $Revision: 1.3 $
*  \author Jo. Weng  - CERN, Ph Division & Uni Karlsruhe
*  \author F.Moortgat - CERN, Ph Division
*/


#include "IOMC/Input/interface/HepMCFileReader.h" 
#include "IOMC/Input/interface/MCFileSource.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>



using namespace edm;
using namespace std;

MCFileSource::MCFileSource( const ParameterSet & pset, InputSourceDescription const& desc ) :
ExternalInputSource(pset, desc),  
reader_( HepMCFileReader::instance() ) {
	
	cout << "MCFileSource:Reading HepMC file: " << fileNames()[0] << endl;
	reader_->initialize(fileNames()[0]);  
	produces<HepMCProduct>();
}


MCFileSource::~MCFileSource(){
	clear();
}

void MCFileSource::clear() {

}


bool MCFileSource::produce(Event & e) {
	
	// clean up GenEvent memory : also deletes all vtx/part in it
	if ( evt != NULL ) delete evt ;
	
	auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
	cout << "MCFileSource: Start Reading  " << endl;
	evt = reader_->fillCurrentEventData(); 
	if(evt) bare_product->addHepMCData(evt);
		
	e.put(bare_product);

        return true;
}
