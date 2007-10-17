#include "Alignment/CommonAlignmentProducer/interface/AlignmentTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

const int kBPIX = 1; // where the hell is this '1' defined?
const int kFPIX = 2; // dito for '2'

// constructor ----------------------------------------------------------------

AlignmentTrackSelector::AlignmentTrackSelector(const edm::ParameterSet & cfg) :
  applyBasicCuts( cfg.getParameter<bool>( "applyBasicCuts" ) ),
  applyNHighestPt( cfg.getParameter<bool>( "applyNHighestPt" ) ),
  applyMultiplicityFilter( cfg.getParameter<bool>( "applyMultiplicityFilter" ) ),
  nHighestPt( cfg.getParameter<int>( "nHighestPt" ) ),
  minMultiplicity ( cfg.getParameter<int>( "minMultiplicity" ) ),
  maxMultiplicity ( cfg.getParameter<int>( "maxMultiplicity" ) ),
  multiplicityOnInput ( cfg.getParameter<bool>( "multiplicityOnInput" ) ),
  ptMin( cfg.getParameter<double>( "ptMin" ) ),
  ptMax( cfg.getParameter<double>( "ptMax" ) ),
  etaMin( cfg.getParameter<double>( "etaMin" ) ),
  etaMax( cfg.getParameter<double>( "etaMax" ) ),
  phiMin( cfg.getParameter<double>( "phiMin" ) ),
  phiMax( cfg.getParameter<double>( "phiMax" ) ),
  nHitMin( cfg.getParameter<double>( "nHitMin" ) ),
  nHitMax( cfg.getParameter<double>( "nHitMax" ) ),
  chi2nMax( cfg.getParameter<double>( "chi2nMax" ) ),
  nHitMin2D( cfg.getParameter<unsigned int>( "nHitMin2D" ) )
{

  if (applyBasicCuts)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "applying basic track cuts ..."
	  << "\nptmin,ptmax:     " << ptMin   << "," << ptMax 
	  << "\netamin,etamax:   " << etaMin  << "," << etaMax
	  << "\nphimin,phimax:   " << phiMin  << "," << phiMax
	  << "\nnhitmin,nhitmax: " << nHitMin << "," << nHitMax
          << "\nnhitmin2D:       " << nHitMin2D
	  << "\nchi2nmax:        " << chi2nMax;

  if (applyNHighestPt)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "filter N tracks with highest Pt N=" << nHighestPt;

  if (applyMultiplicityFilter)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "apply multiplicity filter N>= " << minMultiplicity << "and N<= " << maxMultiplicity
          << " on " << (multiplicityOnInput ? "input" : "output");

  edm::ParameterSet minHitsPerSubdet = cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" );
  minHitsinTIB = minHitsPerSubdet.getParameter<int>( "inTIB" );
  minHitsinTOB = minHitsPerSubdet.getParameter<int>( "inTOB" );
  minHitsinTID = minHitsPerSubdet.getParameter<int>( "inTID" );
  minHitsinTEC = minHitsPerSubdet.getParameter<int>( "inTEC" );
  minHitsinBPIX= minHitsPerSubdet.getParameter<int>( "inBPIX" );
  minHitsinFPIX= minHitsPerSubdet.getParameter<int>( "inFPIX" );
  
  edm::LogInfo("AlignmentTrackSelector") 
    << "Minimum number of hits in TIB/TID/TOB/TEC/BPIX/FPIX = " << minHitsinTIB << "/" << minHitsinTID << "/" << minHitsinTOB << "/" << minHitsinTEC << "/" << minHitsinBPIX << "/" << minHitsinFPIX;
}

// destructor -----------------------------------------------------------------

AlignmentTrackSelector::~AlignmentTrackSelector()
{}


// do selection ---------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::select(const Tracks& tracks, const edm::Event& evt) const 
{
  Tracks result;
  
  if (applyMultiplicityFilter && multiplicityOnInput && 
      (tracks.size() < static_cast<unsigned int>(minMultiplicity) 
       || tracks.size() > static_cast<unsigned int>(maxMultiplicity))) {
//     edm::LogInfo("test") << "@SUB=AlignmentTrackSelector::select" << "skip due to input size "
//                          << tracks.size() << ".";
    return result; // still empty
  }
  
  // apply basic track cuts (if selected)
  if (applyBasicCuts)  result= this->basicCuts(tracks);
  
  // filter N tracks with highest Pt (if selected)
  if (applyNHighestPt) result= this->theNHighestPtTracks(result);
  
  // apply minimum multiplicity requirement (if selected)
  if (applyMultiplicityFilter && !multiplicityOnInput) {
    if (result.size() < static_cast<unsigned int>(minMultiplicity) 
        || result.size() > static_cast<unsigned int>(maxMultiplicity) ) {
//       edm::LogInfo("test") << "@SUB=AlignmentTrackSelector::select" << "remove all due to output size "
//                            << result.size() << ".";
      result.clear();
    }
  }
  
  //edm::LogDebug("AlignmentTrackSelector") << "tracks all,kept: " << tracks.size() << "," << result.size();
  
  return result;
}

// make basic cuts ------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::basicCuts(const Tracks& tracks) const 
{
  Tracks result;

  for (Tracks::const_iterator it=tracks.begin(); it != tracks.end(); ++it) {
    const reco::Track* trackp=*it;
    float pt=trackp->pt();
    float eta=trackp->eta();
    float phi=trackp->phi();
    int nhit = trackp->numberOfValidHits(); 
    float chi2n = trackp->normalizedChi2();

    //edm::LogDebug("AlignmentTrackSelector") << " pt,eta,phi,nhit: "
    //  <<pt<<","<<eta<<","<<phi<<","<<nhit;

    if (pt>ptMin && pt<ptMax 
       && eta>etaMin && eta<etaMax 
       && phi>phiMin && phi<phiMax 
       && nhit>=nHitMin && nhit<=nHitMax
       && chi2n<chi2nMax) {
      if (this->detailedHitsCheck(trackp)) result.push_back(trackp);
//       else {
//         edm::LogInfo("test") << "@SUB=AlignmentTrackSelector::basicCuts"
//                              << "failed hit cuts.";
//       }
//     } else {
//         edm::LogInfo("test") << "@SUB=AlignmentTrackSelector::basicCuts"
//                              << "failed pt " << pt << " or eta "
//                              << eta << " or phi "
//                              << phi << " or nhit "
//                              << nhit << " or chi2n " << chi2n;
    }
  }

  return result;
}

//-----------------------------------------------------------------------------

bool AlignmentTrackSelector::detailedHitsCheck(const reco::Track *trackp) const
{
  // checking hit requirements beyond simple number of valid hits

  if (minHitsinTIB || minHitsinTOB || minHitsinTID || minHitsinTEC
      || minHitsinFPIX || minHitsinBPIX
      || nHitMin2D) { // any detailed hit cut is active, so have to check
    int nhitinTIB = 0, nhitinTOB = 0, nhitinTID = 0, nhitinTEC = 0, nhitinBPIX = 0, nhitinFPIX = 0;
    unsigned int nHit2D = 0;
    for (trackingRecHit_iterator iHit = trackp->recHitsBegin(); iHit != trackp->recHitsEnd(); ++iHit) {
      if (!(*iHit)->isValid()) continue; // only real hits count as in trackp->numberOfValidHits()
      const DetId detId((*iHit)->geographicalId());
      if (detId.det() != DetId::Tracker) {
        edm::LogError("DetectorMismatch") << "@SUB=AlignmentTrackSelector::detailedHitsCheck"
                                          << "DetId.det() != DetId::Tracker (=" << DetId::Tracker
                                          << "), but " << detId.det() << ".";
      }
      if      (StripSubdetector::TIB == detId.subdetId()) ++nhitinTIB;
      else if (StripSubdetector::TOB == detId.subdetId()) ++nhitinTOB;
      else if (StripSubdetector::TID == detId.subdetId()) ++nhitinTID;
      else if (StripSubdetector::TEC == detId.subdetId()) ++nhitinTEC;
      else if (                kBPIX == detId.subdetId()) ++nhitinBPIX;
      else if (                kFPIX == detId.subdetId()) ++nhitinFPIX;
      // Do not call isHit2D(..) if already enough 2D hits for performance reason:
      if (nHit2D < nHitMin2D && this->isHit2D(**iHit)) ++nHit2D;
    } // end loop on hits
    return (nhitinTIB >= minHitsinTIB && nhitinTOB >= minHitsinTOB 
            && nhitinTID >= minHitsinTID && nhitinTEC >= minHitsinTEC 
            && nhitinBPIX >= minHitsinBPIX && nhitinFPIX >= minHitsinFPIX 
            && nHit2D >= nHitMin2D);
  } else { // no cuts set, so we are just fine and can avoid loop on hits
    return true;
  }
}

//-----------------------------------------------------------------------------

bool AlignmentTrackSelector::isHit2D(const TrackingRecHit &hit) const
{
  if (hit.dimension() < 2) {
    return false; // some (muon...) stuff really has RecHit1D
  } else {
    const DetId detId(hit.geographicalId());
    if (detId.det() == DetId::Tracker) {
      if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) {
        return true; // pixel is always 2D
      } else { // should be SiStrip now
        if (dynamic_cast<const SiStripRecHit2D*>(&hit)) return false; // normal hit
        else if (dynamic_cast<const SiStripMatchedRecHit2D*>(&hit)) return true; // matched is 2D
        else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(&hit)) return false; // crazy hit...
        else {
          edm::LogError("UnkownType") << "@SUB=AlignmentTrackSelector::isHit2D"
                                      << "Tracker hit not in pixel and neither SiStripRecHit2D nor "
                                      << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
          return false;
        }
      }
    } else { // not tracker??
      edm::LogWarning("DetectorMismatch") << "@SUB=AlignmentTrackSelector::isHit2D"
                                          << "Hit not in tracker with 'official' dimension >=2.";
      return true; // dimension() >= 2 so accept that...
    }
  }
  // never reached...
}

//-----------------------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::theNHighestPtTracks(const Tracks& tracks) const
{
  Tracks sortedTracks=tracks;
  Tracks result;

  // sort in pt
  std::sort(sortedTracks.begin(),sortedTracks.end(),ptComparator);

  // copy theTrackMult highest pt tracks to result vector
  int n=0;
  for (Tracks::const_iterator it=sortedTracks.begin();
	   it!=sortedTracks.end(); it++) {
	if (n<nHighestPt) { result.push_back(*it); n++; }
  }

  return result;
}

