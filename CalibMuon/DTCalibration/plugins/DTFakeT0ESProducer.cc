/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/04/20 09:34:56 $
 *  $Revision: 1.3 $
 *  \author S. Bolognesi - INFN Torino
 */

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "CalibMuon/DTCalibration/plugins/DTFakeT0ESProducer.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include <DetectorDescription/Core/interface/DDCompactView.h>
#include "CalibMuon/DTCalibration/plugins/DTGeometryParserFromDDD.h"

using namespace std;

DTFakeT0ESProducer::DTFakeT0ESProducer(const edm::ParameterSet& pset)
{
  //framework
  setWhatProduced(this,&DTFakeT0ESProducer::produce);
  //  setWhatProduced(this,dependsOn(& DTGeometryESModule::parseDDD()));
  findingRecord<DTT0Rcd>();
  
  //read constant value for t0 from cfg
  t0Mean = pset.getParameter<double>("t0Mean");
  t0Sigma = pset.getParameter<double>("t0Sigma");
}


DTFakeT0ESProducer::~DTFakeT0ESProducer(){
}


// ------------ method called to produce the data  ------------
DTT0* DTFakeT0ESProducer::produce(const DTT0Rcd& iRecord){
  
  parseDDD(iRecord);
  DTT0* t0Map = new DTT0();
  
  //Loop on layerId-nwires map
 for(map<DTLayerId, pair <unsigned int,unsigned int> >::const_iterator lIdWire = theLayerIdWiresMap.begin();
     lIdWire != theLayerIdWiresMap.end();
     lIdWire++){
   int firstWire = ((*lIdWire).second).first;
   int nWires = ((*lIdWire).second).second;
   //Loop on wires of each layer
   for(int wire=0; wire < nWires; wire++){
     t0Map->set(DTWireId((*lIdWire).first, wire + firstWire), t0Mean, t0Sigma, DTTimeUnits::counts);
   }
 }

  return t0Map;
}

void DTFakeT0ESProducer::parseDDD(const DTT0Rcd& iRecord){

  edm::ESTransientHandle<DDCompactView> cpv;
  edm::ESHandle<MuonDDDConstants> mdc;

  iRecord.getRecord<IdealGeometryRecord>().get(cpv);
  iRecord.getRecord<MuonNumberingRecord>().get(mdc);

  DTGeometryParserFromDDD parser(&(*cpv), *mdc, theLayerIdWiresMap);
}

 void DTFakeT0ESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity){
   oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
  }


