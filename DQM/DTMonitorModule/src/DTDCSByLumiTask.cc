/*
 * \file DTDCSByLumiTask.cc
 * 
 * $Date: 2011/03/02 13:58:32 $
 * $Revision: 1.1 $
 * \author C. Battilana - CIEMAT
 * \author P. Bellan - INFN PD
 * \author A. Branca = INFN PD

 *
 */

#include <DQM/DTMonitorModule/src/DTDCSByLumiTask.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "CondFormats/DataRecord/interface/DTHVStatusRcd.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <FWCore/Framework/interface/EventSetupRecord.h>
#include <FWCore/Framework/interface/EventSetupRecordKey.h>
#include <iostream>


using namespace edm;
using namespace std;


DTDCSByLumiTask::DTDCSByLumiTask(const edm::ParameterSet& ps) : theEvents(0) , theLumis(0) {

  theDQMStore = Service<DQMStore>().operator->();
  LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask") 
    << "[DTDCSByLumiTask]: Constructor" << endl;

  // If needed put getParameter here
  // dtDCSByLumiLabel = ps.getParameter<InputTag>("dtDCSByLumiLabel");

}


DTDCSByLumiTask::~DTDCSByLumiTask(){

  LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask") 
    << "DTDCSByLumiTask: processed " << theEvents << 
    " events in " << theLumis << " lumi sections" << endl;

}


void DTDCSByLumiTask::endJob(){

  LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask") 
    <<"[DTDCSByLumiTask] endjob called!"<<endl;

}


void DTDCSByLumiTask::beginJob(){

  LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask") 
    <<"[DTDCSByLumiTask]: BeginJob"<<endl;

}

void DTDCSByLumiTask::beginRun(const edm::Run& run, const edm::EventSetup& context) {

  LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask") 
    << "[DTDCSByLumiTask]: begin run" << endl;

  bookHistos();

  context.get<MuonGeometryRecord>().get(theDTGeom);

  DTHVRecordFound=true;

  eventsetup::EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType("DTHVStatusRcd"));

  std::vector<eventsetup::EventSetupRecordKey> recordKeys;
  context.fillAvailableRecordKeys(recordKeys);
  vector<eventsetup::EventSetupRecordKey>::iterator it = find(recordKeys.begin(),recordKeys.end(),recordKey);

  if( it == recordKeys.end() ) {
    //record not found
    LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask")<<"Record DTHVStatusRcd does not exist "<<std::endl;

    DTHVRecordFound=false;
  }

}


void DTDCSByLumiTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  theLumis++;

  LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask") 
    << "[DTDCSByLumiTask]: Begin of processed lumi # " << lumiSeg.id().luminosityBlock() 
    << " " << theLumis << " lumi processed by this job" <<  endl;

  for(int wheel=0; wheel <5; wheel++) {
    hActiveUnits[wheel]->Reset(); // Cb by lumi histo need to be resetted in between lumi boundaries
  } 

}


void DTDCSByLumiTask::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {


  if(DTHVRecordFound) context.get<DTHVStatusRcd>().get(hvStatus);

  vector<DTLayer*>::const_iterator layersIt  = theDTGeom->layers().begin();
  vector<DTLayer*>::const_iterator layersEnd = theDTGeom->layers().end();

  for(; layersIt!=layersEnd; ++layersIt) {

    int wheel = (*layersIt)->id().wheel();

    int nWiresLayer = (*layersIt)->specificTopology().channels();

    hActiveUnits[wheel+2]->Fill(1,nWiresLayer); // CB first bin is # of layers
    int nActiveWires = nWiresLayer;

    int flagA = -100;
    int flagC = -100;
    int flagS = -100;
    int first = -100;
    int last  = -100;

    // CB info is not stored if HV is ON -> in this case get returns 1
    // process all other cases and removed wires with "BAD HV" from active
    // wires list

    if( DTHVRecordFound ) {
      if ( !hvStatus->get((*layersIt)->id(),0,first,last,flagA,flagC,flagS) 
          && (flagA || flagC || flagS) ) {
        nActiveWires -= (last - first + 1);
      }    

      if ( !hvStatus->get((*layersIt)->id(),1,first,last,flagA,flagC,flagS) 
          && (flagA || flagC || flagS) ) {
        nActiveWires -= (last - first + 1);
      }
    } else {
      nActiveWires=-1.;
    }

    hActiveUnits[wheel+2]->Fill(2,nActiveWires); // CB 2nd bin is the list of wires wit HV ON

  }

  /* LogTrace("DTDQM|DTMonitorModule|DTDCSByLumiTask") 
     << "[DTDCSByLumiTask]: processed lumi # : " << lumiSeg.id().luminosityBlock()  
     << "\t# of wires " << hActiveUnits->getBinContent(1) 
     << "\t# of Active wires (anodes && cathodes && strips ON) : " 
     << hActiveUnits->getBinContent(2) << endl; 
   */
}


void DTDCSByLumiTask::bookHistos() {

  // Book bylumi histo (# of bins as reduced as possible)
  theDQMStore->setCurrentFolder(topFolder());

  for(int wheel=-2; wheel <=2; wheel++) {

    stringstream wheel_str; wheel_str << wheel;	

    MonitorElement* ME = theDQMStore->book1D("hActiveUnits"+wheel_str.str(),"Active Untis x LS Wh"+wheel_str.str(),2,0.5,2.5);
    ME->setLumiFlag();// Set LumiFlag in order to save histo every LS

    hActiveUnits.push_back(ME);
  }

}

void DTDCSByLumiTask::analyze(const edm::Event& event, const edm::EventSetup& c) {

  theEvents++;

}


string DTDCSByLumiTask::topFolder() const {

  return string("DT/EventInfo/DCSContents");

}
