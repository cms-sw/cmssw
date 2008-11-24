
#include "GeneratorInterface/Pythia6Interface/interface/HepMCProductAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;

HepMCProductAnalyzer::HepMCProductAnalyzer(const edm::ParameterSet& iConfig) :
label_(iConfig.getUntrackedParameter("moduleLabel",std::string("source")))
{
   //now do what ever initialization is needed

}


HepMCProductAnalyzer::~HepMCProductAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HepMCProductAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<HepMCProduct> evt;
//   iEvent.getByLabel("PythiaSource",evt);
//   iEvent.getByLabel("MCFileSource",evt);

//   if there is an ambiguity: get by label
//   iEvent.getByLabel(label_, evt);

// if no ambiguity one can do get by type
   iEvent.getByType(evt);

   evt->GetEvent()->print();


}

