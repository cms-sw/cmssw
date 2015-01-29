/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */



#include "DTChamberEfficiencyTask.h"

//Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"


#include <iterator>
#include <iostream>
#include <cmath>
using namespace edm;
using namespace std;



DTChamberEfficiencyTask::DTChamberEfficiencyTask(const ParameterSet& pset) {

  debug = pset.getUntrackedParameter<bool>("debug",false);

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTChamberEfficiencyTask") << "[DTChamberEfficiencyTask] Constructor called!";

  parameters = pset;

  // the name of the 4D rec hits collection
  recHits4DToken_ = consumes<DTRecSegment4DCollection>(
      edm::InputTag(parameters.getParameter<string>("recHits4DLabel")));

  // parameters to use for the segment quality check
  theMinHitsSegment = static_cast<unsigned int>(parameters.getParameter<int>("minHitsSegment"));
  theMinChi2NormSegment = parameters.getParameter<double>("minChi2NormSegment");
  // parameter to use for the exstrapolated segment check
  theMinCloseDist = parameters.getParameter<double>("minCloseDist");

  // the running modality
  onlineMonitor = parameters.getUntrackedParameter<bool>("onlineMonitor");

  // the analysis mode
  detailedAnalysis = parameters.getUntrackedParameter<bool>("detailedAnalysis");

}


DTChamberEfficiencyTask::~DTChamberEfficiencyTask(){

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTChamberEfficiencyTask") << "[DTChamberEfficiencyTask] Destructor called!";
}

void DTChamberEfficiencyTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTChamberEfficiencyTask")<<"[DTChamberEfficiencyTask]: Begin of LS transition";

  if(lumiSeg.id().luminosityBlock()%parameters.getUntrackedParameter<int>("ResetCycle", 3) == 0 && onlineMonitor) {
    for(map<DTChamberId, vector<MonitorElement*> > ::const_iterator histo = histosPerCh.begin();
	histo != histosPerCh.end();
	histo++) {
      int size = (*histo).second.size();
      for(int i=0; i<size; i++){
	(*histo).second[i]->Reset();
      }
    }
  }

}


void DTChamberEfficiencyTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& setup){

  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

}

void DTChamberEfficiencyTask::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & context) {

  ibooker.setCurrentFolder("DT/DTChamberEfficiencyTask");

  // Loop over all the chambers
  vector<const DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = dtGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    // histo booking
    bookHistos(ibooker, (*ch_it)->id());
  }

}

// Book a set of histograms for a given Layer
void DTChamberEfficiencyTask::bookHistos(DQMStore::IBooker & ibooker, DTChamberId chId) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTChamberEfficiencyTask") << "   Booking histos for CH : " << chId;

  // Compose the chamber name
  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();

  string HistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();

  ibooker.setCurrentFolder("DT/01-DTChamberEfficiency/Task/Wheel" + wheel.str() +
			   "/Sector" + sector.str() +
                           "/Station" + station.str());

  // Create the monitor elements
  vector<MonitorElement *> histos;

  //efficiency selection cuts
  // a- number of segments of the top chamber > 0 && number of segments of the bottom chamber > 0
  // b- number of segments of the middle chamber > 0
  // c- check of the top and bottom segment quality
  // d- check if interpolation falls inside the middle chamber
  // e- check of the middle segment quality
  // f- check if the distance between the reconstructed and the exstrapolated segments is ok


  // histo for efficiency with cuts a-/c-/d-
  histos.push_back(ibooker.book2D("hEffGoodSegVsPosDen"+HistoName,"Eff vs local position (good) ",25,-250.,250., 25,-250.,250.));
  // histo for efficiency with cuts a-/b-/c-/d-/e-/f-
  histos.push_back(ibooker.book2D("hEffGoodCloseSegVsPosNum"+HistoName, "Eff vs local position (good and close segs) ", 25,-250.,250., 25,-250.,250.));
  if(detailedAnalysis){
    histos.push_back(ibooker.book1D("hDistSegFromExtrap"+HistoName, "Distance segments from extrap position ",200,0.,200.));
    // histo for efficiency from segment counting
    histos.push_back(ibooker.book1D("hNaiveEffSeg"+HistoName, "Naive eff ",10,0.,10.));
    // histo for efficiency with cuts a-/c-
  histos.push_back(ibooker.book2D("hEffSegVsPosDen"+HistoName,"Eff vs local position (all) ",25,-250.,250., 25,-250.,250.));
    // histo for efficiency with cuts a-/b-/c-/d-
    histos.push_back(ibooker.book2D("hEffSegVsPosNum"+HistoName, "Eff vs local position ",25,-250.,250., 25,-250.,250.));
    // histo for efficiency with cuts a-/b-/c-/d-/e-
    histos.push_back(ibooker.book2D("hEffGoodSegVsPosNum"+HistoName, "Eff vs local position (good segs) ", 25,-250.,250., 25,-250.,250.));
  }
  histosPerCh[chId] = histos;
}


void DTChamberEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTChamberEfficiencyTask") << "[DTChamberEfficiencyTask] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event();

  // Get the 4D rechit collection from the event
  event.getByToken(recHits4DToken_, segs);

  int bottom=0, top=0;


  // Loop over all the chambers
  vector<const DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = dtGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {

    DTChamberId ch = (*ch_it)->id();
    int wheel =  ch.wheel();
    int sector = ch.sector();
    int station = ch.station();


    DTChamberId MidId(wheel, station, sector);

    // get efficiency for MB1 using MB2 and MB3
    if( station == 1 ) {
      bottom = 2;
      top = 3;
    }

    // get efficiency for MB2 using MB1 and MB3
    if( station == 2 ) {
      bottom = 1;
      top = 3;
    }

    // get efficiency for MB2 using MB2 and MB4
    if( station == 3 ) {
      bottom = 2;
      top = 4;
    }

    // get efficiency for MB4 using MB2 and MB3
    if( station == 4 ) {
      bottom = 2;
      top = 3;
    }

    // Select events with (good) segments in Bot and Top
    DTChamberId BotId(wheel, bottom, sector);
    DTChamberId TopId(wheel, top, sector);

    // Get segments in the bottom chambers (if any)
    DTRecSegment4DCollection::range segsBot= segs->get(BotId);
    int nSegsBot=segsBot.second-segsBot.first;
    // check if any segments is there
    if (nSegsBot==0) continue;

    vector<MonitorElement *> histos =  histosPerCh[MidId];

    // Get segments in the top chambers (if any)
    DTRecSegment4DCollection::range segsTop= segs->get(TopId);
    int nSegsTop=segsTop.second-segsTop.first;

    // Select one segment for the bottom chamber
    const DTRecSegment4D& bestBotSeg= getBestSegment(segsBot);

    // Select one segment for the top chamber
    DTRecSegment4D* pBestTopSeg=0;
    if (nSegsTop>0)
      pBestTopSeg = const_cast<DTRecSegment4D*>(&getBestSegment(segsTop));
    //if top chamber is MB4 sector 10, consider also sector 14
    if (TopId.station() == 4 && TopId.sector() == 10) {
      DTChamberId TopId14(wheel, top, 14);
      DTRecSegment4DCollection::range segsTop14= segs->get(TopId14);
      int nSegsTop14=segsTop14.second-segsTop14.first;
      nSegsTop+=nSegsTop14;
      if (nSegsTop14) {
	DTRecSegment4D* pBestTopSeg14 = const_cast<DTRecSegment4D*>(&getBestSegment(segsTop14));
	// get best between sector 10 and 14
	pBestTopSeg = const_cast<DTRecSegment4D*>(getBestSegment(pBestTopSeg, pBestTopSeg14));
      }
    }
    if (!pBestTopSeg) continue;
    const DTRecSegment4D& bestTopSeg= *pBestTopSeg;

    DTRecSegment4DCollection::range segsMid= segs->get(MidId);
    int nSegsMid=segsMid.second-segsMid.first;

    if(detailedAnalysis){
      // very trivial efficiency, just count segments
      histos[3]->Fill(0);
      if (nSegsMid>0) histos[3]->Fill(1);
    }

    // get position at Mid by interpolating the position (not direction) of best
    // segment in Bot and Top to Mid surface
    LocalPoint posAtMid = interpolate(bestBotSeg, bestTopSeg, MidId);

    // is best segment good enough?
    if (isGoodSegment(bestBotSeg) && isGoodSegment(bestTopSeg)) {
      if(detailedAnalysis)
	histos[4]->Fill(posAtMid.x(),posAtMid.y());
      //check if interpolation fall inside middle chamber
      if ((dtGeom->chamber(MidId))->surface().bounds().inside(posAtMid)) {
	histos[0]->Fill(posAtMid.x(),posAtMid.y());
	if (nSegsMid>0) {

	  if(detailedAnalysis){
	    histos[3]->Fill(2);
	    histos[5]->Fill(posAtMid.x(),posAtMid.y());
	  }

	  const DTRecSegment4D& bestMidSeg= getBestSegment(segsMid);
	  // check if middle segments is good enough
	  if (isGoodSegment(bestMidSeg)) {

	    if(detailedAnalysis)
	      histos[6]->Fill(posAtMid.x(),posAtMid.y());
	    LocalPoint midSegPos=bestMidSeg.localPosition();

	    // check if middle segments is also close enough
	    double dist;
	    if (bestMidSeg.hasPhi()) {
	      if (bestTopSeg.hasZed() && bestBotSeg.hasZed() && bestMidSeg.hasZed()) {
		dist = (midSegPos-posAtMid).mag();
	      } else {
		    dist = fabs((midSegPos-posAtMid).x());
	      }
	    } else {
	      dist = fabs((midSegPos-posAtMid).y());
	    }
	    if (dist < theMinCloseDist ) {
	      histos[1]->Fill(posAtMid.x(),posAtMid.y());
	    }
	    if(detailedAnalysis)
	      histos[2]->Fill(dist);
	  }
	}
      }
    }
  }// loop over stations

}




// requirements : max number of hits and min chi2
const DTRecSegment4D& DTChamberEfficiencyTask::getBestSegment(const DTRecSegment4DCollection::range& segs) const{
  DTRecSegment4DCollection::const_iterator bestIter;
  unsigned int nHitBest=0;
  double chi2Best=99999.;
  for (DTRecSegment4DCollection::const_iterator seg=segs.first ;
       seg!=segs.second ; ++seg ) {
    unsigned int nHits= ((*seg).hasPhi() ? (*seg).phiSegment()->recHits().size() : 0 ) ;
    nHits+= ((*seg).hasZed() ?  (*seg).zSegment()->recHits().size() : 0 );

    if (nHits==nHitBest) {
      if ((*seg).chi2()/(*seg).degreesOfFreedom() < chi2Best ) {
        chi2Best=(*seg).chi2()/(*seg).degreesOfFreedom();
        bestIter = seg;
      }
    }
    else if (nHits>nHitBest) {
      nHitBest=nHits;
      bestIter = seg;
    }
  }
  return *bestIter;
}

const DTRecSegment4D* DTChamberEfficiencyTask::getBestSegment(const DTRecSegment4D* s1,
                                                    const DTRecSegment4D* s2) const{

  if (!s1) return s2;
  if (!s2) return s1;
  unsigned int nHits1= (s1->hasPhi() ? s1->phiSegment()->recHits().size() : 0 ) ;
  nHits1+= (s1->hasZed() ?  s1->zSegment()->recHits().size() : 0 );

  unsigned int nHits2= (s2->hasPhi() ? s2->phiSegment()->recHits().size() : 0 ) ;
  nHits2+= (s2->hasZed() ?  s2->zSegment()->recHits().size() : 0 );

  if (nHits1==nHits2) {
    if (s1->chi2()/s1->degreesOfFreedom() < s2->chi2()/s2->degreesOfFreedom() )
      return s1;
    else
      return s2;
  }
  else if (nHits1>nHits2) return s1;
  return s2;
}


LocalPoint DTChamberEfficiencyTask::interpolate(const DTRecSegment4D& seg1,
					    const DTRecSegment4D& seg3,
					    const DTChamberId& id2) const {
  // Get GlobalPoition of Seg in MB1
  GlobalPoint gpos1=(dtGeom->chamber(seg1.chamberId()))->toGlobal(seg1.localPosition());

  // Get GlobalPoition of Seg in MB3
  GlobalPoint gpos3=(dtGeom->chamber(seg3.chamberId()))->toGlobal(seg3.localPosition());

  // interpolate
  // get all in MB2 frame
  LocalPoint pos1=(dtGeom->chamber(id2))->toLocal(gpos1);
  LocalPoint pos3=(dtGeom->chamber(id2))->toLocal(gpos3);

  // case 1: 1 and 3 has both projection. No problem

  // case 2: one projection is missing for one of the segments. Keep the other's segment position
  if (!seg1.hasZed()) pos1=LocalPoint(pos1.x(),pos3.y(),pos1.z());
  if (!seg3.hasZed()) pos3=LocalPoint(pos3.x(),pos1.y(),pos3.z());

  if (!seg1.hasPhi()) pos1=LocalPoint(pos3.x(),pos1.y(),pos1.z());
  if (!seg3.hasPhi()) pos3=LocalPoint(pos1.x(),pos3.y(),pos3.z());

  // direction
  LocalVector dir = (pos3-pos1).unit(); // z points inward!
  LocalPoint pos2 = pos1+dir*pos1.z()/(-dir.z());

  return pos2;
}


bool DTChamberEfficiencyTask::isGoodSegment(const DTRecSegment4D& seg) const {
  if (seg.chamberId().station()!=4 && !seg.hasZed()) return false;
  unsigned int nHits= (seg.hasPhi() ? seg.phiSegment()->recHits().size() : 0 ) ;
  nHits+= (seg.hasZed() ?  seg.zSegment()->recHits().size() : 0 );
  return ( nHits >= theMinHitsSegment &&
	   seg.chi2()/seg.degreesOfFreedom() < theMinChi2NormSegment );
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
