
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/02/20 15:11:54 $
 *  $Revision: 1.13 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/CalibMuon/interface/DTt0DBValidation.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// t0
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

#include "TFile.h"

#include <sstream>
#include <iomanip>

using namespace edm;
using namespace std;

DTt0DBValidation::DTt0DBValidation(const ParameterSet& pset) {

  metname_ = "InterChannelSynchDBValidation";
  LogVerbatim(metname_) << "[DTt0DBValidation] Constructor called!";

  // Get the DQM needed services
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder("DT/DtCalib/InterChannelSynchDBValidation");

  // Get dataBase label
  labelDBRef_ = pset.getParameter<string>("labelDBRef");
  labelDB_ = pset.getParameter<string>("labelDB");

  t0TestName_ = "t0DifferenceInRange";
  if( pset.exists("t0TestName") ) t0TestName_ = pset.getParameter<string>("t0TestName");
  
  outputMEsInRootFile_ = false;
  if( pset.exists("OutputFileName") ){
     outputMEsInRootFile_ = true;
     outputFileName_ = pset.getParameter<std::string>("OutputFileName");
  }
}


DTt0DBValidation::~DTt0DBValidation(){}


void DTt0DBValidation::beginRun(const edm::Run& run, const EventSetup& setup) {

  metname_ = "InterChannelSynchDBValidation";
  LogVerbatim(metname_) << "[DTt0DBValidation] Parameters initialization";
 
  ESHandle<DTT0> t0_Ref;
  setup.get<DTT0Rcd>().get(labelDBRef_, t0_Ref);
  tZeroRefMap_ = &*t0_Ref;
  LogVerbatim(metname_) << "[DTt0DBValidation] reference T0 version: " << t0_Ref->version();

  ESHandle<DTT0> t0;
  setup.get<DTT0Rcd>().get(labelDB_, t0);
  tZeroMap_ = &*t0;
  LogVerbatim(metname_) << "[DTt0DBValidation] T0 to validate version: " << t0->version();

  //book&reset the summary histos
  for(int wheel=-2; wheel<=2; wheel++){
    bookHistos(wheel);
    wheelSummary_[wheel]->Reset();
  }

  // Get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom_);

  // Loop over Ref DB entries
  for(DTT0::const_iterator tzero = tZeroRefMap_->begin();
                           tzero != tZeroRefMap_->end(); tzero++) {
    // t0s and rms are TDC counts
// @@@ NEW DTT0 FORMAT
//    DTWireId wireId((*tzero).first.wheelId,
//		    (*tzero).first.stationId,
//		    (*tzero).first.sectorId,
//		    (*tzero).first.slId,
//		    (*tzero).first.layerId,
//		    (*tzero).first.cellId);
    int channelId = tzero->channelId;
    if ( channelId == 0 ) continue;
    DTWireId wireId(channelId);
// @@@ NEW DTT0 END
    float t0mean;
    float t0rms;
    tZeroRefMap_->get( wireId, t0mean, t0rms, DTTimeUnits::counts );
    LogTrace(metname_)<< "Ref Wire: " <<  wireId <<endl
		     << " T0 mean (TDC counts): " << t0mean
		     << " T0_rms (TDC counts): " << t0rms;

    t0RefMap_[wireId].push_back(t0mean);
    t0RefMap_[wireId].push_back(t0rms);
  }

  // Loop over Ref DB entries
  for(DTT0::const_iterator tzero = tZeroMap_->begin();
                           tzero != tZeroMap_->end(); tzero++) {
    // t0s and rms are TDC counts
// @@@ NEW DTT0 FORMAT
//    DTWireId wireId((*tzero).first.wheelId,
//		    (*tzero).first.stationId,
//		    (*tzero).first.sectorId,
//		    (*tzero).first.slId,
//		    (*tzero).first.layerId,
//		    (*tzero).first.cellId);
    int channelId = tzero->channelId;
    if ( channelId == 0 ) continue;
    DTWireId wireId(channelId);
// @@@ NEW DTT0 END
    float t0mean;
    float t0rms;
    tZeroMap_->get( wireId, t0mean, t0rms, DTTimeUnits::counts );
    LogTrace(metname_)<< "Wire: " <<  wireId <<endl
		     << " T0 mean (TDC counts): " << t0mean
		     << " T0_rms (TDC counts): " << t0rms;

    t0Map_[wireId].push_back(t0mean);
    t0Map_[wireId].push_back(t0rms);
  }

  double difference = 0;
  for(map<DTWireId, vector<float> >::const_iterator theMap = t0RefMap_.begin();
      theMap != t0RefMap_.end();
      theMap++) {  
    if(t0Map_.find((*theMap).first) != t0Map_.end()) {

      // Compute the difference
      difference = t0Map_[(*theMap).first][0]-(*theMap).second[0];

      //book histo
      DTLayerId layerId = (*theMap).first.layerId();
      if(t0DiffHistos_.find(layerId) == t0DiffHistos_.end()) {
	const DTTopology& dtTopo = dtGeom_->layer(layerId)->specificTopology();
	const int firstWire = dtTopo.firstChannel();
	const int lastWire = dtTopo.lastChannel();
	bookHistos(layerId, firstWire, lastWire);
      }

      LogTrace(metname_)<< "Filling the histo for wire: "<<(*theMap).first
	                <<"  difference: "<<difference;
      t0DiffHistos_[layerId]->Fill((*theMap).first.wire(),difference);

    }
  } // Loop over the t0 map reference
   
}

void DTt0DBValidation::endRun(edm::Run const& run, edm::EventSetup const& setup) {

  // Check the histos
  string testCriterionName = t0TestName_; 
  for(map<DTLayerId, MonitorElement*>::const_iterator hDiff = t0DiffHistos_.begin();
      hDiff != t0DiffHistos_.end();
      hDiff++) {

     const QReport * theDiffQReport = (*hDiff).second->getQReport(testCriterionName);
     if(theDiffQReport) {
        int xBin = ((*hDiff).first.station()-1)*12+(*hDiff).first.layer()+4*((*hDiff).first.superlayer()-1);
        if( (*hDiff).first.station()==4 && (*hDiff).first.superlayer()==3 )
           xBin = ((*hDiff).first.station()-1)*12+(*hDiff).first.layer()+4*((*hDiff).first.superlayer()-2);

        int qReportStatus = theDiffQReport->getStatus()/100;
        wheelSummary_[(*hDiff).first.wheel()]->setBinContent(xBin,(*hDiff).first.sector(),qReportStatus);
 
        LogVerbatim(metname_) << "-------- layer: " << (*hDiff).first << "  " << theDiffQReport->getMessage()
                              << " ------- " << theDiffQReport->getStatus()
                              << " ------- " << setprecision(3) << theDiffQReport->getQTresult();
        vector<dqm::me_util::Channel> badChannels = theDiffQReport->getBadChannels();
        for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	                                           channel != badChannels.end(); channel++) {
           LogVerbatim(metname_) << "layer: " << (*hDiff).first << " Bad channel: " 
                                             << (*channel).getBin() << "  Contents : "
                                             << (*channel).getContents();

           //wheelSummary_[(*hDiff).first.wheel()]->Fill(xBin,(*hDiff).first.sector());
        }
     }
      
  }

}

void DTt0DBValidation::endJob() {
  // Write the histos on a file
  if(outputMEsInRootFile_) dbe_->save(outputFileName_); 
}

  // Book a set of histograms for a given Layer
void DTt0DBValidation::bookHistos(DTLayerId lId, int firstWire, int lastWire) {
  
  LogTrace(metname_)<< "   Booking histos for L: " << lId;

  // Compose the chamber name
  stringstream wheel; wheel << lId.superlayerId().chamberId().wheel();	
  stringstream station; station << lId.superlayerId().chamberId().station();	
  stringstream sector; sector << lId.superlayerId().chamberId().sector();	
  stringstream superLayer; superLayer << lId.superlayerId().superlayer();	
  stringstream layer; layer << lId.layer();

  string lHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str() +
    "_SL" + superLayer.str()+
    "_L" + layer.str();
  
  dbe_->setCurrentFolder("DT/DtCalib/InterChannelSynchDBValidation/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str() +
			   "/SuperLayer" +superLayer.str());
  // Create the monitor elements
  MonitorElement * hDifference;
  hDifference = dbe_->book1D("T0Difference"+lHistoName, "difference between the two t0 values",lastWire-firstWire+1, firstWire-0.5, lastWire+0.5);
  
  t0DiffHistos_[lId] = hDifference;
}

// Book the summary histos
void DTt0DBValidation::bookHistos(int wheel) {
  dbe_->setCurrentFolder("DT/DtCalib/InterChannelSynchDBValidation");
  stringstream wh; wh << wheel;
    wheelSummary_[wheel]= dbe_->book2D("SummaryWrongT0_W"+wh.str(), "W"+wh.str()+": summary of wrong t0 differences",44,1,45,14,1,15);
    wheelSummary_[wheel]->setBinLabel(1,"M1L1",1);
    wheelSummary_[wheel]->setBinLabel(2,"M1L2",1);
    wheelSummary_[wheel]->setBinLabel(3,"M1L3",1);
    wheelSummary_[wheel]->setBinLabel(4,"M1L4",1);
    wheelSummary_[wheel]->setBinLabel(5,"M1L5",1);
    wheelSummary_[wheel]->setBinLabel(6,"M1L6",1);
    wheelSummary_[wheel]->setBinLabel(7,"M1L7",1);
    wheelSummary_[wheel]->setBinLabel(8,"M1L8",1);
    wheelSummary_[wheel]->setBinLabel(9,"M1L9",1);
    wheelSummary_[wheel]->setBinLabel(10,"M1L10",1);
    wheelSummary_[wheel]->setBinLabel(11,"M1L11",1);
    wheelSummary_[wheel]->setBinLabel(12,"M1L12",1);
    wheelSummary_[wheel]->setBinLabel(13,"M2L1",1);
    wheelSummary_[wheel]->setBinLabel(14,"M2L2",1);
    wheelSummary_[wheel]->setBinLabel(15,"M2L3",1);
    wheelSummary_[wheel]->setBinLabel(16,"M2L4",1);
    wheelSummary_[wheel]->setBinLabel(17,"M2L5",1);
    wheelSummary_[wheel]->setBinLabel(18,"M2L6",1);
    wheelSummary_[wheel]->setBinLabel(19,"M2L7",1);
    wheelSummary_[wheel]->setBinLabel(20,"M2L8",1);
    wheelSummary_[wheel]->setBinLabel(21,"M2L9",1);
    wheelSummary_[wheel]->setBinLabel(22,"M2L10",1);
    wheelSummary_[wheel]->setBinLabel(23,"M2L11",1);
    wheelSummary_[wheel]->setBinLabel(24,"M2L12",1);
    wheelSummary_[wheel]->setBinLabel(25,"M3L1",1);
    wheelSummary_[wheel]->setBinLabel(26,"M3L2",1);
    wheelSummary_[wheel]->setBinLabel(27,"M3L3",1);
    wheelSummary_[wheel]->setBinLabel(28,"M3L4",1);
    wheelSummary_[wheel]->setBinLabel(29,"M3L5",1);
    wheelSummary_[wheel]->setBinLabel(30,"M3L6",1);
    wheelSummary_[wheel]->setBinLabel(31,"M3L7",1);
    wheelSummary_[wheel]->setBinLabel(32,"M3L8",1);
    wheelSummary_[wheel]->setBinLabel(33,"M3L9",1);
    wheelSummary_[wheel]->setBinLabel(34,"M3L10",1);
    wheelSummary_[wheel]->setBinLabel(35,"M3L11",1);
    wheelSummary_[wheel]->setBinLabel(36,"M3L12",1);
    wheelSummary_[wheel]->setBinLabel(37,"M4L1",1);
    wheelSummary_[wheel]->setBinLabel(38,"M4L2",1);
    wheelSummary_[wheel]->setBinLabel(39,"M4L3",1);
    wheelSummary_[wheel]->setBinLabel(40,"M4L4",1);
    wheelSummary_[wheel]->setBinLabel(41,"M4L5",1);
    wheelSummary_[wheel]->setBinLabel(42,"M4L6",1);
    wheelSummary_[wheel]->setBinLabel(43,"M4L7",1);
    wheelSummary_[wheel]->setBinLabel(44,"M4L8",1);
}
