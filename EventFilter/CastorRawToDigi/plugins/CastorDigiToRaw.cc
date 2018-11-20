#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

#include "EventFilter/CastorRawToDigi/interface/CastorCollections.h"
#include "EventFilter/CastorRawToDigi/plugins/CastorDigiToRaw.h"

using namespace std;

CastorDigiToRaw::CastorDigiToRaw(edm::ParameterSet const& conf) :
  castorTag_(conf.getParameter<edm::InputTag>("CASTOR")),
  usingctdc_(conf.getParameter<bool>("CastorCtdc")),
  tok_input_(consumes<CastorDigiCollection>(castorTag_)),
  tok_put_(produces<FEDRawDataCollection>())
{
}

// Functions that gets called by framework every event
void CastorDigiToRaw::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const
{
  CastorCollections colls;
 
  // Step A: Get Inputs 
  edm::Handle<CastorDigiCollection> castor;
  if (!castorTag_.label().empty()) {
    e.getByToken(tok_input_,castor);
    colls.castorCont=castor.product();	
  }
  // get the mapping
  edm::ESHandle<CastorDbService> pSetup;
  es.get<CastorDbRecord>().get( pSetup );
  const CastorElectronicsMap* readoutMap=pSetup->getCastorMapping();
  // Step B: Create empty output
  FEDRawDataCollection raw;

  constexpr int ifed_first=FEDNumbering::MINCASTORFEDID;  //690
  constexpr int ifed_last=FEDNumbering::MAXCASTORFEDID;   //693

  int orbitN=e.id().event();
  int bcnN=2000;

  // Step C: pack all requested FEDs
  for (int ifed=ifed_first; ifed<=ifed_last; ++ifed) {
    FEDRawData& fed = raw.FEDData(ifed);
    try {
      if ( usingctdc_ ) {
        CastorCtdcPacker::pack(ifed,ifed-ifed_first, e.id().event(),
                               orbitN, bcnN, colls, *readoutMap, fed);
      } else {
        CastorPacker::pack(ifed,ifed-ifed_first, e.id().event(),
                           orbitN, bcnN, colls, *readoutMap, fed);
      }
    } catch (cms::Exception& e) {
      edm::LogWarning("Unpacking error") << e.what();
    } catch (...) {
      edm::LogWarning("Unpacking exception");
    }
  }

  e.emplace(tok_put_,std::move(raw));
}


