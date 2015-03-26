/*
 * \file DTTriggerEfficiencyTask.cc
 *
 * \author C.Battilana - CIEMAT
 *
 */

#include "DQM/DTMonitorModule/src/DTTriggerEfficiencyTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// DT trigger
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// DT Digi
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

// Muon tracks
#include <DataFormats/MuonReco/interface/Muon.h>

//Root
#include"TH1.h"
#include"TAxis.h"

#include <sstream>
#include <iostream>
#include <fstream>


using namespace edm;
using namespace std;

DTTriggerEfficiencyTask::DTTriggerEfficiencyTask(const edm::ParameterSet& ps) : trigGeomUtils(0) {

  LogTrace ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask")  << "[DTTriggerEfficiencyTask]: Constructor" << endl;

  parameters = ps;

  muons_Token_ = consumes<reco::MuonCollection>(
      parameters.getUntrackedParameter<edm::InputTag>("inputTagMuons"));
  dcc_Token_   = consumes<L1MuDTChambPhContainer>(
      parameters.getUntrackedParameter<edm::InputTag>("inputTagDCC"));
  ddu_Token_   = consumes<DTLocalTriggerCollection>(
      parameters.getUntrackedParameter<edm::InputTag>("inputTagDDU"));
  inputTagSEG  = parameters.getUntrackedParameter<edm::InputTag>("inputTagSEG");
  gmt_Token_   = consumes<L1MuGMTReadoutCollection>(
      parameters.getUntrackedParameter<edm::InputTag>("inputTagGMT"));

  SegmArbitration = parameters.getUntrackedParameter<std::string>("SegmArbitration");

  detailedPlots = parameters.getUntrackedParameter<bool>("detailedAnalysis");
  processDCC = parameters.getUntrackedParameter<bool>("processDCC");
  processDDU = parameters.getUntrackedParameter<bool>("processDDU");
  minBXDDU = parameters.getUntrackedParameter<int>("minBXDDU");
  maxBXDDU = parameters.getUntrackedParameter<int>("maxBXDDU");

  nMinHitsPhi = parameters.getUntrackedParameter<int>("nMinHitsPhi");
  phiAccRange = parameters.getUntrackedParameter<double>("phiAccRange");

  if (processDCC) processTags.push_back("DCC");
  if (processDDU) processTags.push_back("DDU");

}


DTTriggerEfficiencyTask::~DTTriggerEfficiencyTask() {

  LogTrace ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask")  << "[DTTriggerEfficiencyTask]: analyzed " << nevents << " events" << endl;

}

void DTTriggerEfficiencyTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) {

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);
  trigGeomUtils = new DTTrigGeomUtils(muonGeom);

}

void DTTriggerEfficiencyTask::bookHistograms(DQMStore::IBooker & ibooker,
                                             edm::Run const & run,
                                             edm::EventSetup const & context) {

  LogTrace ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask") << "[DTTriggerEfficiencyTask]: bookHistograms" << endl;

  nevents = 0;

  for (int wh=-2;wh<=2;++wh){
    vector<string>::const_iterator tagIt  = processTags.begin();
    vector<string>::const_iterator tagEnd = processTags.end();
    for (; tagIt!=tagEnd; ++tagIt) {

      bookWheelHistos(ibooker,wh,(*tagIt),"Task");
      if (detailedPlots) {
        for (int stat=1;stat<=4;++stat){
          for (int sect=1;sect<=12;++sect){
            bookChamberHistos(ibooker,DTChamberId(wh,stat,sect),(*tagIt),"Segment");
          }
        }
      }
    }
  }
}

void DTTriggerEfficiencyTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  LogTrace ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask") <<"[DTTriggerEfficiencyTask]: Begin of LS transition"<<endl;

}

void DTTriggerEfficiencyTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;

  if (!hasRPCTriggers(e)) { return; }

  map<DTChamberId,const L1MuDTChambPhDigi*> phBestDCC;
  map<DTChamberId,const DTLocalTrigger*>    phBestDDU;

  // Getting best DCC Stuff
  edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
  e.getByToken(dcc_Token_, l1DTTPGPh);
  vector<L1MuDTChambPhDigi> const*  phTrigs = l1DTTPGPh->getContainer();

  vector<L1MuDTChambPhDigi>::const_iterator iph  = phTrigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();
  for(; iph !=iphe ; ++iph) {

    int phwheel = iph->whNum();
    int phsec   = iph->scNum() + 1; // DTTF numbering [0:11] -> DT numbering [1:12]
    int phst    = iph->stNum();
    int phcode  = iph->code();

    DTChamberId chId(phwheel,phst,phsec);

    if( phcode < 7 && (phBestDCC.find(chId) == phBestDCC.end() ||
          phcode>phBestDCC[chId]->code()) ) phBestDCC[chId] = &(*iph);
  }

  //Getting Best DDU Stuff
  Handle<DTLocalTriggerCollection> trigsDDU;
  e.getByToken(ddu_Token_, trigsDDU);
  DTLocalTriggerCollection::DigiRangeIterator detUnitIt;

  for (detUnitIt=trigsDDU->begin();detUnitIt!=trigsDDU->end();++detUnitIt){

    const DTChamberId& id = (*detUnitIt).first;
    const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;

    DTLocalTriggerCollection::const_iterator trigIt  = range.first;
    DTLocalTriggerCollection::const_iterator trigEnd = range.second;
    for (; trigIt!= trigEnd;++trigIt){
      int quality = trigIt->quality();
      if(quality>-1 && quality<7 &&
          (phBestDDU.find(id) == phBestDDU.end() ||
           quality>phBestDDU[id]->quality()) ) phBestDDU[id] = &(*trigIt);
    }

  }

  //Getting Best Segments
  vector<const DTRecSegment4D*> best4DSegments;

  Handle<reco::MuonCollection> muons;
  e.getByToken(muons_Token_, muons);
  reco::MuonCollection::const_iterator mu;

  for( mu = muons->begin(); mu != muons->end(); ++mu ) {

    // Make sure that is standalone muon
    if( !((*mu).isStandAloneMuon()) ) {continue;}

    // Get the chambers compatible with the muon
    const vector<reco::MuonChamberMatch> matchedChambers = (*mu).matches();
    vector<reco::MuonChamberMatch>::const_iterator chamber;

    for( chamber = matchedChambers.begin(); chamber != matchedChambers.end(); ++chamber ) {

      // look only in DTs
      if( chamber->detector() != MuonSubdetId::DT ) {continue;}

      // Get the matched segments in the chamber
      const vector<reco::MuonSegmentMatch> matchedSegments = (*chamber).segmentMatches;
      vector<reco::MuonSegmentMatch>::const_iterator segment;

      for( segment = matchedSegments.begin(); segment != matchedSegments.end(); ++segment ) {

        edm::Ref<DTRecSegment4DCollection> dtSegment = segment->dtSegmentRef;

        // Segment Arbitration
        if( SegmArbitration == "SegmentArbitration"
            && !((*segment).isMask(reco::MuonSegmentMatch::BestInChamberByDR)) ) {continue;}

        if( SegmArbitration == "SegmentAndTrackArbitration"
            && (!((*segment).isMask(reco::MuonSegmentMatch::BestInChamberByDR)) ||
              !((*segment).isMask(reco::MuonSegmentMatch::BelongsToTrackByDR))) ) {continue;}

        if( SegmArbitration == "SegmentAndTrackArbitrationCleaned"
            && (!((*segment).isMask(reco::MuonSegmentMatch::BestInChamberByDR))  ||
              !((*segment).isMask(reco::MuonSegmentMatch::BelongsToTrackByDR)) ||
              !((*segment).isMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning))) ) {continue;}


        if( (*dtSegment).hasPhi() ) {
          best4DSegments.push_back(&(*dtSegment));
        }

      }// end loop on matched segments
    }// end loop on compatible chambers
  }// end loop on muons

  // Plot filling
  vector<const DTRecSegment4D*>::const_iterator btrack;
  for ( btrack = best4DSegments.begin(); btrack != best4DSegments.end(); ++btrack ){

    int wheel    = (*btrack)->chamberId().wheel();
    int station  = (*btrack)->chamberId().station();
    int scsector = 0;
    float x, xdir, y, ydir;
    trigGeomUtils->computeSCCoordinates((*btrack),scsector,x,xdir,y,ydir);
    int nHitsPhi = (*btrack)->phiSegment()->degreesOfFreedom()+2;
    DTChamberId dtChId(wheel,station,scsector);
    uint32_t indexCh = dtChId.rawId();
    map<string, MonitorElement*> &innerChME = chamberHistos[indexCh];
    map<string, MonitorElement*> &innerWhME = wheelHistos[wheel];

    if (fabs(xdir)<phiAccRange && nHitsPhi>=nMinHitsPhi){

      vector<string>::const_iterator tagIt  = processTags.begin();
      vector<string>::const_iterator tagEnd = processTags.end();

      for (; tagIt!=tagEnd; ++tagIt) {

        int qual   = (*tagIt) == "DCC" ?
          phBestDCC.find(dtChId) != phBestDCC.end() ? phBestDCC[dtChId]->code() : -1 :
          phBestDDU.find(dtChId) != phBestDDU.end() ? phBestDDU[dtChId]->quality() : -1;

        innerWhME.find((*tagIt) + "_TrigEffDenum")->second->Fill(scsector,station);
        if ( qual>=0 && qual<7 ) {
          innerWhME.find((*tagIt) + "_TrigEffNum")->second->Fill(scsector,station);
          if ( qual>=4 ) {
            innerWhME.find((*tagIt) + "_TrigEffCorrNum")->second->Fill(scsector,station);
          }
        }
        if (detailedPlots) {
          innerChME.find((*tagIt) + "_TrackPosvsAngle")->second->Fill(xdir,x);
          if ( qual>=0 && qual<7 ) {
            innerChME.find((*tagIt) + "_TrackPosvsAngleAnyQual")->second->Fill(xdir,x);
            if ( qual>=4 ) {
              innerChME.find((*tagIt) + "_TrackPosvsAngleCorr")->second->Fill(xdir,x);
            }
          }
        }
      }
    }
  }

}

bool DTTriggerEfficiencyTask::hasRPCTriggers(const edm::Event& e) {

  edm::Handle<L1MuGMTReadoutCollection> gmtrc;
  e.getByToken(gmt_Token_, gmtrc);

  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr = gmt_records.begin();
  std::vector<L1MuGMTReadoutRecord>::const_iterator egmtrr = gmt_records.end();
  for(; igmtrr!=egmtrr; igmtrr++) {

    std::vector<L1MuGMTExtendedCand> candsGMT = igmtrr->getGMTCands();
    std::vector<L1MuGMTExtendedCand>::const_iterator candGMTIt   = candsGMT.begin();
    std::vector<L1MuGMTExtendedCand>::const_iterator candGMTEnd  = candsGMT.end();

    for(; candGMTIt!=candGMTEnd; ++candGMTIt){
      if(!candGMTIt->empty()) {
        int quality = candGMTIt->quality();
        if(candGMTIt->bx()==0 &&
            (quality == 5 || quality == 7)){
          return true;
        }
      }
    }
  }

  return false;

}

void DTTriggerEfficiencyTask::bookChamberHistos(DQMStore::IBooker& ibooker,const DTChamberId& dtCh, 
                                                  string histoType, string folder) {

  int wh = dtCh.wheel();
  int sc = dtCh.sector();
  int st = dtCh.station();
  stringstream wheel; wheel << wh;
  stringstream station; station << st;
  stringstream sector; sector << sc;

  string hwFolder      = topFolder(histoType);
  string bookingFolder = hwFolder + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" + station.str() + "/" + folder;
  string histoTag      = "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  ibooker.setCurrentFolder(bookingFolder);

  LogTrace ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask")
    << "[DTTriggerEfficiencyTask]: booking histos in " << bookingFolder << endl;

  float min, max;
  int nbins;
  trigGeomUtils->phiRange(dtCh,min,max,nbins,20);

  string histoName = histoType + "_TrackPosvsAngle" +  histoTag;
  string histoLabel = "Position vs Angle (phi)";

  (chamberHistos[dtCh.rawId()])[histoType + "_TrackPosvsAngle"] =
    ibooker.book2D(histoName,histoLabel,12,-30.,30.,nbins,min,max);

  histoName = histoType + "_TrackPosvsAngleAnyQual" +  histoTag;
  histoLabel = "Position vs Angle (phi) for any qual triggers";

  (chamberHistos[dtCh.rawId()])[histoType + "_TrackPosvsAngleAnyQual"] =
    ibooker.book2D(histoName,histoLabel,12,-30.,30.,nbins,min,max);

  histoName = histoType + "_TrackPosvsAngleCorr" +  histoTag;
  histoLabel = "Position vs Angle (phi) for correlated triggers";

  (chamberHistos[dtCh.rawId()])[histoType + "_TrackPosvsAngleCorr"] =
    ibooker.book2D(histoName,histoLabel,12,-30.,30.,nbins,min,max);

}

void DTTriggerEfficiencyTask::bookWheelHistos(DQMStore::IBooker& ibooker,int wheel,string hTag,
                                                string folder) {

  stringstream wh; wh << wheel;
  string basedir;
  if (hTag.find("Summary") != string::npos ) {
    basedir = topFolder(hTag);   //Book summary histo outside folder directory
  } else {
    basedir = topFolder(hTag) + folder + "/" ;
  }

  ibooker.setCurrentFolder(basedir);

  string hTagName = "_W" + wh.str();

  LogTrace("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask")
    << "[DTTriggerEfficiencyTask]: booking histos in "<< basedir << endl;

  string hName = hTag + "_TrigEffDenum" + hTagName;

  MonitorElement* me = ibooker.book2D(hName.c_str(),hName.c_str(),12,1,13,4,1,5);

  me->setBinLabel(1,"MB1",2);
  me->setBinLabel(2,"MB2",2);
  me->setBinLabel(3,"MB3",2);
  me->setBinLabel(4,"MB4",2);
  me->setAxisTitle("Sector",1);

  wheelHistos[wheel][hTag + "_TrigEffDenum"] = me;

  hName = hTag + "_TrigEffNum" + hTagName;
  me = ibooker.book2D(hName.c_str(),hName.c_str(),12,1,13,4,1,5);

  me->setBinLabel(1,"MB1",2);
  me->setBinLabel(2,"MB2",2);
  me->setBinLabel(3,"MB3",2);
  me->setBinLabel(4,"MB4",2);
  me->setAxisTitle("Sector",1);

  wheelHistos[wheel][hTag + "_TrigEffNum"] = me;

  hName = hTag + "_TrigEffCorrNum" + hTagName;
  me = ibooker.book2D(hName.c_str(),hName.c_str(),12,1,13,4,1,5);

  me->setBinLabel(1,"MB1",2);
  me->setBinLabel(2,"MB2",2);
  me->setBinLabel(3,"MB3",2);
  me->setBinLabel(4,"MB4",2);
  me->setAxisTitle("Sector",1);

  wheelHistos[wheel][hTag + "_TrigEffCorrNum"] = me;

  return;
}


// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
