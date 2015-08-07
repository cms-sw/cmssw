/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 *  revised by G. Mila - INFN Torino
 */

#include "DTSegmentAnalysisTask.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

#include <iterator>
#include <TMath.h>

using namespace edm;
using namespace std;

DTSegmentAnalysisTask::DTSegmentAnalysisTask(const edm::ParameterSet& pset) : nevents(0) , nEventsInLS(0), hNevtPerLS(0) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "[DTSegmentAnalysisTask] Constructor called!";

  // switch for detailed analysis
  detailedAnalysis = pset.getUntrackedParameter<bool>("detailedAnalysis",false);
  // the name of the 4D rec hits collection
  recHits4DToken_ = consumes<DTRecSegment4DCollection>(
      edm::InputTag(pset.getParameter<string>("recHits4DLabel")));
  // Get the map of noisy channels
  checkNoisyChannels = pset.getUntrackedParameter<bool>("checkNoisyChannels",false);
  // # of bins in the time histos
  nTimeBins = pset.getUntrackedParameter<int>("nTimeBins",100);
  // # of LS per bin in the time histos
  nLSTimeBin = pset.getUntrackedParameter<int>("nLSTimeBin",2);
  // switch on/off sliding bins in time histos
  slideTimeBins = pset.getUntrackedParameter<bool>("slideTimeBins",true);
  phiSegmCut = pset.getUntrackedParameter<double>("phiSegmCut",30.);
  nhitsCut = pset.getUntrackedParameter<int>("nhitsCut",12);

  // top folder for the histograms in DQMStore
  topHistoFolder = pset.getUntrackedParameter<string>("topHistoFolder","DT/02-Segments");
  // hlt DQM mode
  hltDQMMode = pset.getUntrackedParameter<bool>("hltDQMMode",false);

}


DTSegmentAnalysisTask::~DTSegmentAnalysisTask(){
  //FR moved fron endjob
  delete hNevtPerLS;
  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "[DTSegmentAnalysisTask] Destructor called!";
}


void DTSegmentAnalysisTask::dqmBeginRun(const Run& run, const edm::EventSetup& context){

  // Get the DT Geometry
  context.get<MuonGeometryRecord>().get(dtGeom);

}

void DTSegmentAnalysisTask::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & context) {

  if (!hltDQMMode) {
    ibooker.setCurrentFolder("DT/EventInfo/Counters");
    nEventMonitor = ibooker.bookFloat("nProcessedEventsSegment");
  }

  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    ibooker.setCurrentFolder(topHistoFolder + "/Wheel" + wheel.str());
    string histoName =  "numberOfSegments_W" + wheel.str();

    summaryHistos[wh] = ibooker.book2D(histoName.c_str(),histoName.c_str(),12,1,13,4,1,5);
    summaryHistos[wh]->setAxisTitle("Sector",1);
    summaryHistos[wh]->setBinLabel(1,"1",1);
    summaryHistos[wh]->setBinLabel(2,"2",1);
    summaryHistos[wh]->setBinLabel(3,"3",1);
    summaryHistos[wh]->setBinLabel(4,"4",1);
    summaryHistos[wh]->setBinLabel(5,"5",1);
    summaryHistos[wh]->setBinLabel(6,"6",1);
    summaryHistos[wh]->setBinLabel(7,"7",1);
    summaryHistos[wh]->setBinLabel(8,"8",1);
    summaryHistos[wh]->setBinLabel(9,"9",1);
    summaryHistos[wh]->setBinLabel(10,"10",1);
    summaryHistos[wh]->setBinLabel(11,"11",1);
    summaryHistos[wh]->setBinLabel(12,"12",1);
    summaryHistos[wh]->setBinLabel(1,"MB1",2);
    summaryHistos[wh]->setBinLabel(2,"MB2",2);
    summaryHistos[wh]->setBinLabel(3,"MB3",2);
    summaryHistos[wh]->setBinLabel(4,"MB4",2);
  }

  // loop over all the DT chambers & book the histos
  const vector<const DTChamber*>& chambers = dtGeom->chambers();
  vector<const DTChamber*>::const_iterator ch_it = chambers.begin();
  vector<const DTChamber*>::const_iterator ch_end = chambers.end();
  for (; ch_it != ch_end; ++ch_it) {
    bookHistos(ibooker,(*ch_it)->id());
  }

  // book sector time-evolution histos
  int modeTimeHisto = 0;
  if(!slideTimeBins) modeTimeHisto = 1;
  for(int wheel = -2; wheel != 3; ++wheel) { // loop over wheels
    for(int sector = 1; sector <= 12; ++sector) { // loop over sectors

      stringstream wheelstr; wheelstr << wheel;
      stringstream sectorstr; sectorstr << sector;
      string sectorHistoName = "NSegmPerEvent_W" + wheelstr.str()
	+ "_Sec" + sectorstr.str();
      string sectorHistoTitle = "# segm. W" + wheelstr.str() + " Sect." + sectorstr.str();

      ibooker.setCurrentFolder(topHistoFolder + "/Wheel" + wheelstr.str() +
	  "/Sector" + sectorstr.str());

      histoTimeEvol[wheel][sector] = new DTTimeEvolutionHisto(ibooker,sectorHistoName,sectorHistoTitle,
	  nTimeBins,nLSTimeBin,slideTimeBins,modeTimeHisto);

    }
  }

  if(hltDQMMode) ibooker.setCurrentFolder(topHistoFolder);
  else ibooker.setCurrentFolder("DT/EventInfo/");

  hNevtPerLS = new DTTimeEvolutionHisto(ibooker,"NevtPerLS","# evt.",nTimeBins,nLSTimeBin,slideTimeBins,2);

}

void DTSegmentAnalysisTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {

  nevents++;
  nEventMonitor->Fill(nevents);

  nEventsInLS++;
  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "[DTSegmentAnalysisTask] Analyze #Run: " << event.id().run()
    << " #Event: " << event.id().event();
  if(!(event.id().event()%1000))
    edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "[DTSegmentAnalysisTask] Analyze #Run: " << event.id().run()
      << " #Event: " << event.id().event();

  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    setup.get<DTStatusFlagRcd>().get(statusMap);
  }


  // -- 4D segment analysis  -----------------------------------------------------

  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByToken(recHits4DToken_, all4DSegments);

  if(!all4DSegments.isValid()) return;

  // Loop over all chambers containing a segment
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = all4DSegments->id_begin();
      chamberId != all4DSegments->id_end();
      ++chamberId){
    // Get the range for the corresponding ChamerId
    DTRecSegment4DCollection::range  range = all4DSegments->get(*chamberId);

    edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "   Chamber: " << *chamberId << " has " << distance(range.first, range.second)
      << " 4D segments";

    // Loop over the rechits of this ChamerId
    for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
	segment4D!=range.second;
	++segment4D){

      //FOR NOISY CHANNELS////////////////////////////////
      bool segmNoisy = false;
      if(checkNoisyChannels) {

	if((*segment4D).hasPhi()){
	  const DTChamberRecSegment2D* phiSeg = (*segment4D).phiSegment();
	  vector<DTRecHit1D> phiHits = phiSeg->specificRecHits();
	  map<DTSuperLayerId,vector<DTRecHit1D> > hitsBySLMap;
	  for(vector<DTRecHit1D>::const_iterator hit = phiHits.begin();
	      hit != phiHits.end(); ++hit) {
	    DTWireId wireId = (*hit).wireId();

	    // Check for noisy channels to skip them
	    bool isNoisy = false;
	    bool isFEMasked = false;
	    bool isTDCMasked = false;
	    bool isTrigMask = false;
	    bool isDead = false;
	    bool isNohv = false;
	    statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	    if(isNoisy) {
	      edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "Wire: " << wireId << " is noisy, skipping!";
	      segmNoisy = true;
	    }
	  }
	}

	if((*segment4D).hasZed()) {
	  const DTSLRecSegment2D* zSeg = (*segment4D).zSegment();  // zSeg lives in the SL RF
	  // Check for noisy channels to skip them
	  vector<DTRecHit1D> zHits = zSeg->specificRecHits();
	  for(vector<DTRecHit1D>::const_iterator hit = zHits.begin();
	      hit != zHits.end(); ++hit) {
	    DTWireId wireId = (*hit).wireId();
	    bool isNoisy = false;
	    bool isFEMasked = false;
	    bool isTDCMasked = false;
	    bool isTrigMask = false;
	    bool isDead = false;
	    bool isNohv = false;
	    statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	    if(isNoisy) {
	      edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "Wire: " << wireId << " is noisy, skipping!";
	      segmNoisy = true;
	    }
	  }
	}

      } // end of switch on noisy channels
      if (segmNoisy) {
	edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask")<<"skipping the segment: it contains noisy cells";
	continue;
      }
      //END FOR NOISY CHANNELS////////////////////////////////

      int nHits=0;
      if((*segment4D).hasPhi())
	nHits = (((*segment4D).phiSegment())->specificRecHits()).size();
      if((*segment4D).hasZed())
	nHits = nHits + ((((*segment4D).zSegment())->specificRecHits()).size());

      double anglePhiSegm(0.);
      if( (*segment4D).hasPhi() ) {
	double xdir = (*segment4D).phiSegment()->localDirection().x();
	double zdir = (*segment4D).phiSegment()->localDirection().z();

	anglePhiSegm = atan(xdir/zdir)*180./TMath::Pi();
      }
      if( fabs(anglePhiSegm) > phiSegmCut ) continue;
      // If the segment is in Wh+-2/SecX/MB1, get the DT chambers just above and check if there is a segment
      // to validate the segment present in MB1
      if( fabs((*chamberId).wheel()) == 2 && (*chamberId).station() == 1 ) {

	bool segmOk=false;
	int mb(2);
	while( mb < 4 ) {
	  DTChamberId checkMB((*chamberId).wheel(),mb,(*chamberId).sector());
	  DTRecSegment4DCollection::range  ckrange = all4DSegments->get(checkMB);

	  for (DTRecSegment4DCollection::const_iterator cksegment4D = ckrange.first;
	      cksegment4D!=ckrange.second;
	      ++cksegment4D){

	    int nHits=0;
	    if((*cksegment4D).hasPhi())
	      nHits = (((*cksegment4D).phiSegment())->specificRecHits()).size();
	    if((*cksegment4D).hasZed())
	      nHits = nHits + ((((*cksegment4D).zSegment())->specificRecHits()).size());

	    if( nHits >= nhitsCut ) segmOk=true;

	  }
	  mb++;
	}

	if( !segmOk ) continue;

      }
      fillHistos(*chamberId,
	  nHits,
	  (*segment4D).chi2()/(*segment4D).degreesOfFreedom());
    }
  }

  // -----------------------------------------------------------------------------
}


// Book a set of histograms for a give chamber
void DTSegmentAnalysisTask::bookHistos(DQMStore::IBooker & ibooker, DTChamberId chamberId) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTSegmentAnalysisTask") << "   Booking histos for chamber: " << chamberId;


  // Compose the chamber name
  stringstream wheel; wheel << chamberId.wheel();
  stringstream station; station << chamberId.station();
  stringstream sector; sector << chamberId.sector();

  string chamberHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();

  ibooker.setCurrentFolder(topHistoFolder + "/Wheel" + wheel.str() +
      "/Sector" + sector.str() +
      "/Station" + station.str());

  // Create the monitor elements
  vector<MonitorElement *> histos;
  histos.push_back(ibooker.book1D("h4DSegmNHits"+chamberHistoName,
	"# of hits per segment",
	16, 0.5, 16.5));
  if(detailedAnalysis){
    histos.push_back(ibooker.book1D("h4DChi2"+chamberHistoName,
	  "4D Segment reduced Chi2",
	  20, 0, 20));
  }
  histosPerCh[chamberId] = histos;
}


// Fill a set of histograms for a give chamber
void DTSegmentAnalysisTask::fillHistos(DTChamberId chamberId,
    int nHits,
    float chi2) {
  int sector = chamberId.sector();
  if(chamberId.sector()==13) {
    sector = 4;
  } else if(chamberId.sector()==14) {
    sector = 10;
  }

  summaryHistos[chamberId.wheel()]->Fill(sector,chamberId.station());
  histoTimeEvol[chamberId.wheel()][sector]->accumulateValueTimeSlot(1);

  vector<MonitorElement *> histos =  histosPerCh[chamberId];
  histos[0]->Fill(nHits);
  if(detailedAnalysis){
    histos[1]->Fill(chi2);
  }

}


void DTSegmentAnalysisTask::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& eSetup) {

  hNevtPerLS->updateTimeSlot(lumiSeg.luminosityBlock(), nEventsInLS);
  // book sector time-evolution histos
  for(int wheel = -2; wheel != 3; ++wheel) {
    for(int sector = 1; sector <= 12; ++sector) {
      histoTimeEvol[wheel][sector]->updateTimeSlot(lumiSeg.luminosityBlock(), nEventsInLS);
    }
  }
}


void DTSegmentAnalysisTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& eSetup) {
  nEventsInLS = 0;
}



// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
