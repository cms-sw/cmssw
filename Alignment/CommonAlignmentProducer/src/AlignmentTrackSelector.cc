#include "Alignment/CommonAlignmentProducer/interface/AlignmentTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

// constructor ----------------------------------------------------------------

AlignmentTrackSelector::AlignmentTrackSelector(const edm::ParameterSet & cfg) :
  applyBasicCuts_( cfg.getParameter<bool>( "applyBasicCuts" ) ),
  applyNHighestPt_( cfg.getParameter<bool>( "applyNHighestPt" ) ),
  applyMultiplicityFilter_( cfg.getParameter<bool>( "applyMultiplicityFilter" ) ),
  nHighestPt_( cfg.getParameter<int>( "nHighestPt" ) ),
  minMultiplicity_ ( cfg.getParameter<int>( "minMultiplicity" ) ),
  maxMultiplicity_ ( cfg.getParameter<int>( "maxMultiplicity" ) ),
  multiplicityOnInput_ ( cfg.getParameter<bool>( "multiplicityOnInput" ) ),
  ptMin_( cfg.getParameter<double>( "ptMin" ) ),
  ptMax_( cfg.getParameter<double>( "ptMax" ) ),
  etaMin_( cfg.getParameter<double>( "etaMin" ) ),
  etaMax_( cfg.getParameter<double>( "etaMax" ) ),
  phiMin_( cfg.getParameter<double>( "phiMin" ) ),
  phiMax_( cfg.getParameter<double>( "phiMax" ) ),
  nHitMin_( cfg.getParameter<double>( "nHitMin" ) ),
  nHitMax_( cfg.getParameter<double>( "nHitMax" ) ),
  chi2nMax_( cfg.getParameter<double>( "chi2nMax" ) ),
  nHitMin2D_( cfg.getParameter<unsigned int>( "nHitMin2D" ) ),
  // Ugly to use the same getParameter 6 times, but this allows const cut variables...
  minHitsinTIB_(cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTIB" ) ),
  minHitsinTOB_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTOB" ) ),
  minHitsinTID_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTID" ) ),
  minHitsinTEC_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTEC" ) ),
  minHitsinBPIX_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inBPIX" ) ),
  minHitsinFPIX_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inFPIX" ) )
{

  if (applyBasicCuts_)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "applying basic track cuts ..."
	  << "\nptmin,ptmax:     " << ptMin_   << "," << ptMax_ 
	  << "\netamin,etamax:   " << etaMin_  << "," << etaMax_
	  << "\nphimin,phimax:   " << phiMin_  << "," << phiMax_
	  << "\nnhitmin,nhitmax: " << nHitMin_ << "," << nHitMax_
          << "\nnhitmin2D:       " << nHitMin2D_
	  << "\nchi2nmax:        " << chi2nMax_;

  if (applyNHighestPt_)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "filter N tracks with highest Pt N=" << nHighestPt_;

  if (applyMultiplicityFilter_)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "apply multiplicity filter N>= " << minMultiplicity_ << "and N<= " << maxMultiplicity_
          << " on " << (multiplicityOnInput_ ? "input" : "output");

  edm::LogInfo("AlignmentTrackSelector") 
    << "Minimum number of hits in TIB/TID/TOB/TEC/BPIX/FPIX = " 
    << minHitsinTIB_ << "/" << minHitsinTID_ << "/" << minHitsinTOB_
    << "/" << minHitsinTEC_ << "/" << minHitsinBPIX_ << "/" << minHitsinFPIX_;
}

// destructor -----------------------------------------------------------------

AlignmentTrackSelector::~AlignmentTrackSelector()
{}


// do selection ---------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::select(const Tracks& tracks, const edm::Event& evt) const 
{
  
  if (applyMultiplicityFilter_ && multiplicityOnInput_ && 
      (tracks.size() < static_cast<unsigned int>(minMultiplicity_) 
       || tracks.size() > static_cast<unsigned int>(maxMultiplicity_))) {
//     edm::LogInfo("test") << "@SUB=AlignmentTrackSelector::select" << "skip due to input size "
//                          << tracks.size() << ".";
    return Tracks(); // empty collection
  }
  
  Tracks result = tracks;
  // apply basic track cuts (if selected)
  if (applyBasicCuts_)  result = this->basicCuts(result);
  
  // filter N tracks with highest Pt (if selected)
  if (applyNHighestPt_) result = this->theNHighestPtTracks(result);
  
  // apply minimum multiplicity requirement (if selected)
  if (applyMultiplicityFilter_ && !multiplicityOnInput_) {
    if (result.size() < static_cast<unsigned int>(minMultiplicity_) 
        || result.size() > static_cast<unsigned int>(maxMultiplicity_) ) {
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

    if (pt>ptMin_ && pt<ptMax_ 
       && eta>etaMin_ && eta<etaMax_ 
       && phi>phiMin_ && phi<phiMax_ 
       && nhit>=nHitMin_ && nhit<=nHitMax_
       && chi2n<chi2nMax_) {
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

  if (minHitsinTIB_ || minHitsinTOB_ || minHitsinTID_ || minHitsinTEC_
      || minHitsinFPIX_ || minHitsinBPIX_
      || nHitMin2D_) { // any detailed hit cut is active, so have to check
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
      if (nHit2D < nHitMin2D_ && this->isHit2D(**iHit)) ++nHit2D;
    } // end loop on hits
    return (nhitinTIB >= minHitsinTIB_ && nhitinTOB >= minHitsinTOB_ 
            && nhitinTID >= minHitsinTID_ && nhitinTEC >= minHitsinTEC_ 
            && nhitinBPIX >= minHitsinBPIX_ && nhitinFPIX >= minHitsinFPIX_ 
            && nHit2D >= nHitMin2D_);
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
	if (n<nHighestPt_) { result.push_back(*it); n++; }
  }

  return result;
}

