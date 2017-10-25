/*
 * \file DTLocalLutTriggerTask.cc
 *
 * \author D. Fasanella - INFN Bologna
 *
 */

#include "DQM/DTMonitorModule/src/DTLocalTriggerLutTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// DT trigger
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

//Root
#include"TH1.h"
#include"TAxis.h"

#include <sstream>
#include <iostream>
#include <cmath>


using namespace edm;
using namespace std;

DTLocalTriggerLutTask::DTLocalTriggerLutTask(const edm::ParameterSet& ps) : trigGeomUtils(nullptr) {

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerLutTask") << "[DTLocalTriggerLutTask]: Constructor"<<endl;

  tm_TokenIn_   = consumes<L1MuDTChambPhContainer>(
      ps.getUntrackedParameter<InputTag>("inputTagTMin"));
  tm_TokenOut_   = consumes<L1MuDTChambPhContainer>(
      ps.getUntrackedParameter<InputTag>("inputTagTMout"));
  seg_Token_   = consumes<DTRecSegment4DCollection>(
      ps.getUntrackedParameter<InputTag>("inputTagSEG"));

  overUnderIn      = ps.getUntrackedParameter<bool>("rebinOutFlowsInGraph");
  detailedAnalysis = ps.getUntrackedParameter<bool>("detailedAnalysis");
  theGeomLabel     = ps.getUntrackedParameter<string>("geomLabel");

  if (detailedAnalysis){
    nPhiBins  = 401;
    rangePhi  = 10.025;
    nPhibBins = 401;
    rangePhiB = 10.025;
  } else {
    nPhiBins  = 51;
    rangePhi  = 5.1;
    nPhibBins = 51;
    rangePhiB = 10.2;
  }

  baseFolder = "DT/03-LocalTrigger-TM/";
  parameters = ps;

  nEvents = 0;
  nLumis  = 0;

}


DTLocalTriggerLutTask::~DTLocalTriggerLutTask() {

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerLutTask") << "[DTLocalTriggerLutTask]: analyzed " << nEvents << " events" << endl;
  if (trigGeomUtils) { delete trigGeomUtils; }

}

void DTLocalTriggerLutTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) {

  // Get the geometry
  context.get<MuonGeometryRecord>().get(theGeomLabel,muonGeom);
  trigGeomUtils = new DTTrigGeomUtils(muonGeom);
}

void DTLocalTriggerLutTask::bookHistos(DQMStore::IBooker & ibooker, DTChamberId chId) {

  stringstream wheel; wheel << chId.wheel();
  stringstream sector; sector << chId.sector();
  stringstream station; station << chId.station();

  ibooker.setCurrentFolder(topFolder() + "Wheel" + wheel.str() + "/Sector" + sector.str() +
			"/Station" + station.str() + "/Segment");

  string chTag = "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
  std::map<std::string, MonitorElement*> &chambMap = chHistos[chId.rawId()];

  string hName = "TM_PhiResidualIn";

  chambMap[hName] = ibooker.book1D(hName+chTag,"Trigger local position In - Segment local position (correlated triggers)",nPhiBins,-rangePhi,rangePhi);

  hName = "TM_PhiResidualOut";

  chambMap[hName] = ibooker.book1D(hName+chTag,"Trigger local position Out - Segment local position (correlated triggers)",nPhiBins,-rangePhi,rangePhi);

  hName = "TM_PhibResidualIn";

  chambMap[hName] = ibooker.book1D(hName+chTag,"Trigger local direction In - Segment local direction (correlated triggers)",nPhibBins,-rangePhiB,rangePhiB);

  hName = "TM_PhibResidualOut";

  chambMap[hName] = ibooker.book1D(hName+chTag,"Trigger local direction Out - Segment local direction (correlated triggers)",nPhibBins,-rangePhiB,rangePhiB);


  if (detailedAnalysis) {

    hName = "TM_PhitkvsPhitrig";

    chambMap[hName] = ibooker.book2D(hName+chTag,"Local position: segment vs trigger",100,-500.,500.,100,-500.,500.);
    hName = "TM_PhibtkvsPhibtrig";

    chambMap[hName] =ibooker.book2D(hName+chTag,"Local direction : segment vs trigger",200,-40.,40.,200,-40.,40.);
    hName = "TM_PhibResidualvsTkPos";

    chambMap[hName] =ibooker.book2D(hName+chTag,"Local direction residual vs Segment Position",100,-500.,500.,200,-10.,10.);
    hName = "TM_PhiResidualvsTkPos";

    chambMap[hName] =ibooker.book2D(hName+chTag,"Local Position residual vs Segment Position",100,-500.,500.,200,-10.,10.);

  }

}

void DTLocalTriggerLutTask::bookHistograms(DQMStore::IBooker & ibooker,
                                             edm::Run const & run,
                                             edm::EventSetup const & context) {

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerLutTask") << "[DTLocalTriggerLutTask]: bookHistograms" << endl;

  std::vector<const DTChamber*>::const_iterator chambIt  = muonGeom->chambers().begin();
  std::vector<const DTChamber*>::const_iterator chambEnd = muonGeom->chambers().end();

  for (; chambIt!=chambEnd; ++chambIt)
    bookHistos(ibooker,(*chambIt)->id());

}


void DTLocalTriggerLutTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  nLumis++;
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerLutTask") << "[DTLocalTriggerLutTask]: Begin of LS transition" << endl;

  if(nLumis%parameters.getUntrackedParameter<int>("ResetCycle") == 0) {

    LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerLutTask") << "[DTLocalTriggerLutTask]: Cleaning histos" << endl;
    map<uint32_t, map<string, MonitorElement*> > ::const_iterator chambIt  = chHistos.begin();
    map<uint32_t, map<string, MonitorElement*> > ::const_iterator chambEnd = chHistos.end();

    for(; chambIt!=chambEnd; ++chambIt) {
      map<string, MonitorElement*> ::const_iterator histoIt  = chambIt->second.begin();
      map<string, MonitorElement*> ::const_iterator histoEnd = chambIt->second.end();
      for(; histoIt!=histoEnd; ++ histoIt) {
	histoIt->second->Reset();
      }
    }

  }

}



void DTLocalTriggerLutTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nEvents++;

  edm::Handle<L1MuDTChambPhContainer> trigHandleIn;
  e.getByToken(tm_TokenIn_, trigHandleIn);
  edm::Handle<L1MuDTChambPhContainer> trigHandleOut;
  e.getByToken(tm_TokenOut_, trigHandleOut);

  vector<L1MuDTChambPhDigi> const* trigsIn = trigHandleIn->getContainer();
  searchTMBestIn(trigsIn);
  vector<L1MuDTChambPhDigi> const* trigsOut = trigHandleOut->getContainer();
  searchTMBestOut(trigsOut);

  Handle<DTRecSegment4DCollection> segments4D;
  e.getByToken(seg_Token_, segments4D);
  DTRecSegment4DCollection::id_iterator chamberId;

  // Preliminary loop finds best 4D Segment and high quality ones
  vector<const DTRecSegment4D*> best4DSegments;

  for (chamberId = segments4D->id_begin(); chamberId != segments4D->id_end(); ++chamberId){

    DTRecSegment4DCollection::range  rangeInCh = segments4D->get(*chamberId);
    DTRecSegment4DCollection::const_iterator trackIt  = rangeInCh.first;
    DTRecSegment4DCollection::const_iterator trackEnd = rangeInCh.second;

    const DTRecSegment4D* tmpBest = nullptr;
    int tmpdof = 0;
    int dof = 0;

    for (; trackIt!=trackEnd; ++trackIt){

      if(trackIt->hasPhi()) {
	dof = trackIt->phiSegment()->degreesOfFreedom();
	if (dof>tmpdof) {
	  tmpBest = &(*trackIt);
	  tmpdof = dof;
	}
      }

    }

    if (tmpBest) best4DSegments.push_back(tmpBest);

  }

  vector<const DTRecSegment4D*>::const_iterator bestTrackIt  = best4DSegments.begin();
  vector<const DTRecSegment4D*>::const_iterator bestTrackEnd = best4DSegments.end();

  for (; bestTrackIt!=bestTrackEnd; ++bestTrackIt) {

    if((*bestTrackIt)->hasPhi()) {

      DTChamberId chId = (*bestTrackIt)->chamberId();
      int nHitsPhi = (*bestTrackIt)->phiSegment()->degreesOfFreedom()+2;

      int wheel    = chId.wheel();
      int station  = chId.station();
      int scsector = 0;
      float trackPosPhi, trackPosEta, trackDirPhi, trackDirEta;
      trigGeomUtils->computeSCCoordinates((*bestTrackIt),scsector,trackPosPhi,trackDirPhi,trackPosEta,trackDirEta);

      map<string, MonitorElement*> &chMap = chHistos[chId.rawId()];
      // In part
      if (trigQualBestIn[wheel+3][station][scsector] > 3 &&  // residuals only for correlate triggers
	  trigQualBestIn[wheel+3][station][scsector] < 7 &&
	  nHitsPhi>=7 ) {

	float trigPos = trigGeomUtils->trigPos(trigBestIn[wheel+3][station][scsector]);
	float trigDir = trigGeomUtils->trigDir(trigBestIn[wheel+3][station][scsector]);
	trigGeomUtils->trigToSeg(station,trigPos,trackDirPhi);

	double deltaPos = trigPos-trackPosPhi;
	deltaPos = overUnderIn ? max(min(deltaPos,rangePhi-0.01),-rangePhi+0.01) : deltaPos;
	double deltaDir = trigDir-trackDirPhi;
	deltaDir = overUnderIn ? max(min(deltaDir,rangePhiB-0.01),-rangePhiB+0.01) : deltaDir;
	chMap.find("TM_PhiResidualIn")->second->Fill(deltaPos);
	chMap.find("TM_PhibResidualIn")->second->Fill(deltaDir);

	if (detailedAnalysis){
	  chMap.find("TM_PhitkvsPhitrig")->second->Fill(trigPos,trackPosPhi);
	  chMap.find("TM_PhibtkvsPhibtrig")->second->Fill(trigDir,trackDirPhi);
	  chMap.find("TM_PhibResidualvsTkPos")->second->Fill(trackPosPhi,trigDir-trackDirPhi);
	  chMap.find("TM_PhiResidualvsTkPos")->second->Fill(trackPosPhi,trigPos-trackPosPhi);
	}
      }

      // Out part
      if (trigQualBestOut[wheel+3][station][scsector] > 3 &&  // residuals only for correlate triggers
          trigQualBestOut[wheel+3][station][scsector] < 7 &&
          nHitsPhi>=7 ) {

        float trigPos = trigGeomUtils->trigPos(trigBestOut[wheel+3][station][scsector]);
        float trigDir = trigGeomUtils->trigDir(trigBestOut[wheel+3][station][scsector]);
        trigGeomUtils->trigToSeg(station,trigPos,trackDirPhi);

        double deltaPos = trigPos-trackPosPhi;
        deltaPos = overUnderIn ? max(min(deltaPos,rangePhi-0.01),-rangePhi+0.01) : deltaPos;
        double deltaDir = trigDir-trackDirPhi;
        deltaDir = overUnderIn ? max(min(deltaDir,rangePhiB-0.01),-rangePhiB+0.01) : deltaDir;
        chMap.find("TM_PhiResidualOut")->second->Fill(deltaPos);
        chMap.find("TM_PhibResidualOut")->second->Fill(deltaDir);
     }

    }
  }
}

void DTLocalTriggerLutTask::searchTMBestIn( std::vector<L1MuDTChambPhDigi> const* trigs) {

  string histoType ;
  string histoTag ;

  // define best quality trigger segment
  // start from 1 and zero is kept empty
  for (int st=0;st<=4;++st)
    for (int wh=0;wh<=5;++wh)
      for (int sec=0;sec<=12;++sec)
	trigQualBestIn[wh][st][sec] = -1;

  vector<L1MuDTChambPhDigi>::const_iterator trigIt  = trigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator trigEnd = trigs->end();
  for(; trigIt!=trigEnd; ++trigIt) {

    int wh   = trigIt->whNum();
    int sec  = trigIt->scNum() + 1; // DTTF -> DT sector range transform
    int st   = trigIt->stNum();
    int qual = trigIt->code();

    if(qual>trigQualBestIn[wh+wheelArrayShift][st][sec] && qual<7) {
      trigQualBestIn[wh+wheelArrayShift][st][sec]=qual;
      trigBestIn[wh+wheelArrayShift][st][sec] = &(*trigIt);
    }

  }

}


void DTLocalTriggerLutTask::searchTMBestOut( std::vector<L1MuDTChambPhDigi> const* trigs) {

  string histoType ;
  string histoTag ;

// define best quality trigger segment
//   // start from 1 and zero is kept empty
  for (int st=0;st<=4;++st)
    for (int wh=0;wh<=5;++wh)
      for (int sec=0;sec<=12;++sec)
        trigQualBestOut[wh][st][sec] = -1;

  vector<L1MuDTChambPhDigi>::const_iterator trigIt  = trigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator trigEnd = trigs->end();
  for(; trigIt!=trigEnd; ++trigIt) {

    int wh   = trigIt->whNum();
    int sec  = trigIt->scNum() + 1; // DTTF -> DT sector range transform
    int st   = trigIt->stNum();
    int qual = trigIt->code();

    if(qual>trigQualBestOut[wh+wheelArrayShift][st][sec] && qual<7) {
      trigQualBestOut[wh+wheelArrayShift][st][sec]=qual;
      trigBestOut[wh+wheelArrayShift][st][sec] = &(*trigIt);
    }

  }

}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
