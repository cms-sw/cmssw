/*
 *  $Date: 2005/08/03 14:59:53 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - S. Argiro'
 */

#include "DAQEcalTBInputService.h"

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h> 



#include "IORawData/EcalTBInputService/src/EcalTBDaqFileReader.h"
#include "FakeRetriever.h"

#include <FWCore/EDProduct/interface/CollisionID.h>
#include <FWCore/EDProduct/interface/EDProduct.h>
#include <FWCore/Framework/src/TypeID.h> 
#include <FWCore/Framework/interface/InputServiceDescription.h>
#include <FWCore/Framework/interface/EventPrincipal.h>
#include <FWCore/EDProduct/interface/Wrapper.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <iostream>

using namespace edm;
using namespace std;
using namespace raw;


DAQEcalTBInputService::DAQEcalTBInputService(const ParameterSet& pset, 
			 const InputServiceDescription& desc) : 
  InputService(desc.process_name),
  description_(desc),
  // FIXME: DAQEcalTBReader is implemented a  la singleton...
  retriever_(new FakeRetriever()),
  reader_(EcalTBDaqFileReader::instance()), 
  remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1))

{    

   
  std::string filename= pset.getParameter<string>("fileName");

  reader_->initialize(filename);

}



DAQEcalTBInputService::~DAQEcalTBInputService(){
  delete retriever_;
  //  clear();
}


// void DAQEcalTBInputService::clear() {
//   for(map<int, DaqFEDRawData *>::iterator it = daqevdata_.begin();
//       it != daqevdata_.end(); ++it) {
//     delete (*it).second;
//   }
//   daqevdata_.clear();
// }


auto_ptr<EventPrincipal>
DAQEcalTBInputService::read() {

  auto_ptr<EventPrincipal> result(0);

  //  clear();

  CollisionID id=0;

  FEDRawDataCollection* bare_product = new FEDRawDataCollection;

  //  if (!reader_->fillDaqEventData(id, *bare_product)) return result;
  
  if (remainingEvents_-- == 0 || !reader_->fillDaqEventData(id, *bare_product)   )
    return result;


  result = auto_ptr<EventPrincipal>(new EventPrincipal(id, *retriever_));
  


    ModuleDescription      mdesc;
    mdesc.pid = PS_ID("DAQEcalTBInputService");
    mdesc.module_name = "DAQEcalTBInputService";

    mdesc.module_label = "EcalDaqRawData";

#warning version number is hardcoded

    mdesc.version_number = 1UL;
    mdesc.process_name = description_.process_name;
    mdesc.pass = description_.pass;

    edm::Wrapper<FEDRawDataCollection> *wrapped_product = 
      new edm::Wrapper<FEDRawDataCollection> (*bare_product);

    auto_ptr<EDProduct>  prod(wrapped_product);

    auto_ptr<Provenance> prov(new Provenance(mdesc)); 

    edm::TypeID fedcoll_type(prod);

    //prov->full_product_type_name=fedcoll_type.userClassName();
    //prov->friendly_product_type_name=fedcoll_type.friendlyClassName();
    
    prov->full_product_type_name=string("raw::FEDRawDataCollection");
    prov->friendly_product_type_name=string("FEDRawDataCollection");

    result->put(prod, prov);


  return result;
}


