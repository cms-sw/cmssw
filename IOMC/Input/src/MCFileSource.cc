/**  
*  See header file for a description of this class.
*
*
*  $Date: 2006/04/07 04:04:19 $
*  $Revision: 1.7 $
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
reader_( HepMCFileReader::instance() ),
evt(0) {
	
	cout << "MCFileSource:Reading HepMC file: " << fileNames()[0] << endl;
	string fileName = fileNames()[0];
	// strip the file: 
	if ( ! fileName.find("file:")){
	  fileName.erase(0,5);
	}  
  
	reader_->initialize(fileName);  
	produces<HepMCProduct>();
}


MCFileSource::~MCFileSource(){
	clear();
}

void MCFileSource::clear() {

}


bool MCFileSource::produce(Event & e) {
	
	// no need to clean up GenEvent memory - now done in HepMCProduct
//	if (evt != 0) {
//	  delete evt;
//	  evt = 0;
//	}
	auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
	cout << "MCFileSource: Start Reading  " << endl;
	evt = reader_->fillCurrentEventData(); 
        if (evt == 0) return false;
	bare_product->addHepMCData(evt);
		
	e.put(bare_product);

        return true;
}
