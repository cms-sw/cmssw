/*
 *  $Date: 2005/09/30 08:17:48 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - S. Argiro'
 */

#include "DaqSource.h"
#include <FWCore/EDProduct/interface/EventID.h>
#include <FWCore/EDProduct/interface/Timestamp.h>
#include <FWCore/EDProduct/interface/EDProduct.h>
#include <FWCore/EDProduct/interface/Wrapper.h>
#include <FWCore/Framework/src/TypeID.h> 
#include <FWCore/Framework/interface/InputSourceDescription.h>
#include <FWCore/Framework/interface/EventPrincipal.h>
#include <FWCore/Framework/interface/ProductRegistry.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>

#include <iostream>
#include <string>

using namespace edm;
using namespace std;


DaqSource::DaqSource(const ParameterSet& pset, 
		     const InputSourceDescription& desc) : 
  InputSource(desc),
  reader_(0),
  remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1))
{
  //Generate the ModuleDescription // FIXME: hardcoded?
  ModuleDescription mdesc;
  mdesc.pid = PS_ID("DaqSource");
  mdesc.moduleName_ = "DaqSource";
  mdesc.moduleLabel_ = "DaqRawData";  
  mdesc.versionNumber_ = 1UL;
  mdesc.processName_ = desc.processName_;
  mdesc.pass = desc.pass;  
  
  // Fill the ProductDescription // FIXME : ??
  FEDRawDataCollection tmp;
  edm::TypeID fedcoll_type(tmp);
  fedrawdataDescription_.module = mdesc;
  fedrawdataDescription_.fullClassName_=fedcoll_type.userClassName();
  fedrawdataDescription_.friendlyClassName_=fedcoll_type.friendlyClassName();
  productRegistry().addProduct(fedrawdataDescription_); 

  // Instantiate the requested data source
  string reader = pset.getParameter<string>("reader");
  reader_ = DaqReaderPluginFactory::get()->create(reader,pset.getParameter<ParameterSet>("pset"));
}



DaqSource::~DaqSource(){
  // delete reader_; // FIXME: who is the owner???
}


auto_ptr<EventPrincipal>
DaqSource::read() {
  auto_ptr<EventPrincipal> result(0);

  //  clear();

  EventID eventId;
  Timestamp tstamp;

  FEDRawDataCollection* bare_product = new FEDRawDataCollection;

  if (remainingEvents_-- == 0 ||!reader_->fillRawData(eventId, 
						      tstamp, 
						      *bare_product)){
    return result;
  }
  result = 
    auto_ptr<EventPrincipal>(new EventPrincipal(eventId,tstamp,productRegistry()));
   
  edm::Wrapper<FEDRawDataCollection> *wrapped_product = 
    new edm::Wrapper<FEDRawDataCollection> (*bare_product);

  auto_ptr<EDProduct>  prod(wrapped_product);
  
  auto_ptr<Provenance> prov(new Provenance(fedrawdataDescription_));
    
  result->put(prod, prov);

  return result;
}


