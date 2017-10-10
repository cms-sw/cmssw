/** \file
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/GEMRawToDigi/plugins/GEMDigiToRawModule.h"
#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"

#include "CondFormats/DataRecord/interface/GEMChamberMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMChamberMap.h"

GEMDigiToRawModule::GEMDigiToRawModule(const edm::ParameterSet & pset) 
{
  event_type_ = pset.getParameter<int>("eventType");
  digi_token = consumes<GEMDigiCollection>( pset.getParameter<edm::InputTag>("gemDigi") );
  produces<FEDRawDataCollection>("GEMRawData");
}

void GEMDigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gemDigi", edm::InputTag("simMuonGEMDigis"));
}


void GEMDigiToRawModule::produce( edm::Event & e, const edm::EventSetup& c ){

  bool verbose_ = true;
  ///reverse mapping for packer
  edm::ESHandle<GEMChamberMap> gemChamberMap;
  c.get<GEMChamberMapRcd>().get(gemChamberMap); 
  const GEMChamberMap* theMapping = gemChamberMap.product();

  auto fedRawDataCol = std::make_unique<FEDRawDataCollection>();

  // Take digis from the event
  edm::Handle<GEMDigiCollection> gemDigis;
  e.getByToken( digi_token, gemDigi );

  e.put(std::move(fedRawDataCol));
}
