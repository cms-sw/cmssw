#include "Alignment/CommonAlignmentProducer/interface/AlignmentTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"


#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

// constructor ----------------------------------------------------------------

AlignmentTrackSelector::AlignmentTrackSelector(const edm::ParameterSet & cfg) :
  applyBasicCuts_( cfg.getParameter<bool>( "applyBasicCuts" ) ),
  applyNHighestPt_( cfg.getParameter<bool>( "applyNHighestPt" ) ),
  applyMultiplicityFilter_( cfg.getParameter<bool>( "applyMultiplicityFilter" ) ),
  seedOnlyFromAbove_( cfg.getParameter<int>( "seedOnlyFrom" ) ),
  applyIsolation_( cfg.getParameter<bool>( "applyIsolationCut" ) ),
  chargeCheck_( cfg.getParameter<bool>( "applyChargeCheck" ) ),
  nHighestPt_( cfg.getParameter<int>( "nHighestPt" ) ),
  minMultiplicity_ ( cfg.getParameter<int>( "minMultiplicity" ) ),
  maxMultiplicity_ ( cfg.getParameter<int>( "maxMultiplicity" ) ),
  multiplicityOnInput_ ( cfg.getParameter<bool>( "multiplicityOnInput" ) ),
  ptMin_( cfg.getParameter<double>( "ptMin" ) ),
  ptMax_( cfg.getParameter<double>( "ptMax" ) ),
  pMin_( cfg.getParameter<double>( "pMin" ) ),
  pMax_( cfg.getParameter<double>( "pMax" ) ),
  etaMin_( cfg.getParameter<double>( "etaMin" ) ),
  etaMax_( cfg.getParameter<double>( "etaMax" ) ),
  phiMin_( cfg.getParameter<double>( "phiMin" ) ),
  phiMax_( cfg.getParameter<double>( "phiMax" ) ),
  nHitMin_( cfg.getParameter<double>( "nHitMin" ) ),
  nHitMax_( cfg.getParameter<double>( "nHitMax" ) ),
  chi2nMax_( cfg.getParameter<double>( "chi2nMax" ) ),
  minHitChargeStrip_( cfg.getParameter<double>( "minHitChargeStrip" ) ),
  minHitIsolation_( cfg.getParameter<double>( "minHitIsolation" ) ),
  rphirecHitsTag_( cfg.getParameter<edm::InputTag>("rphirecHits") ),
  matchedrecHitsTag_( cfg.getParameter<edm::InputTag>("matchedrecHits") ),
  nHitMin2D_( cfg.getParameter<unsigned int>( "nHitMin2D" ) ),
  // Ugly to use the same getParameter 6 times, but this allows const cut variables...
  minHitsinTIB_(cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTIB" ) ),
  minHitsinTOB_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTOB" ) ),
  minHitsinTID_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTID" ) ),
  minHitsinTEC_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inTEC" ) ),
  minHitsinBPIX_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inBPIX" ) ),
  minHitsinFPIX_ (cfg.getParameter<edm::ParameterSet>( "minHitsPerSubDet" ).getParameter<int>( "inFPIX" ) )
{
  if (applyBasicCuts_){
      edm::LogInfo("AlignmentTrackSelector") 
	<< "applying basic track cuts ..."
	<< "\nptmin,ptmax:     " << ptMin_   << "," << ptMax_ 
	<< "\npmin,pmax:       " << pMin_    << "," << pMax_ 
	<< "\netamin,etamax:   " << etaMin_  << "," << etaMax_
	<< "\nphimin,phimax:   " << phiMin_  << "," << phiMax_
	<< "\nnhitmin,nhitmax: " << nHitMin_ << "," << nHitMax_
	<< "\nnhitmin2D:       " << nHitMin2D_
	<< "\nchi2nmax:        " << chi2nMax_;

      if (applyIsolation_)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "only retain tracks isolated at least by " << minHitIsolation_ << 
	  " cm from other rechits"; 
      
      if (chargeCheck_)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "only retain hits with at least " << minHitChargeStrip_ <<  
	  " ADC counts of total cluster charge"; 
      
      edm::LogInfo("AlignmentTrackSelector") 
	<< "Minimum number of hits in TIB/TID/TOB/TEC/BPIX/FPIX = " 
	<< minHitsinTIB_ << "/" << minHitsinTID_ << "/" << minHitsinTOB_
	<< "/" << minHitsinTEC_ << "/" << minHitsinBPIX_ << "/" << minHitsinFPIX_;
    }
  
  if (applyNHighestPt_)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "filter N tracks with highest Pt N=" << nHighestPt_;

  if (applyMultiplicityFilter_)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "apply multiplicity filter N>= " << minMultiplicity_ << "and N<= " << maxMultiplicity_
          << " on " << (multiplicityOnInput_ ? "input" : "output");

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
    return Tracks(); // empty collection
  }
  
  Tracks result = tracks;
  // apply basic track cuts (if selected)
  if (applyBasicCuts_) result= this->basicCuts(result, evt);
  
  // filter N tracks with highest Pt (if selected)
  if (applyNHighestPt_) result = this->theNHighestPtTracks(result);
  
  // apply minimum multiplicity requirement (if selected)
  if (applyMultiplicityFilter_ && !multiplicityOnInput_) {
    if (result.size() < static_cast<unsigned int>(minMultiplicity_) 
        || result.size() > static_cast<unsigned int>(maxMultiplicity_) ) {

      result.clear();
    }
  }
  
  return result;
}

 ///returns if any of the Filters is used.
bool AlignmentTrackSelector::useThisFilter()
{
  return applyMultiplicityFilter_  || applyBasicCuts_ || applyNHighestPt_ ;
}


// make basic cuts ------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::basicCuts(const Tracks& tracks, const edm::Event& evt) const 
{
  Tracks result;

  for (Tracks::const_iterator it=tracks.begin(); it != tracks.end(); ++it) {
    const reco::Track* trackp=*it;
    float pt=trackp->pt();
    float p=trackp->p();
    float eta=trackp->eta();
    float phi=trackp->phi();
    int nhit = trackp->numberOfValidHits(); 
    float chi2n = trackp->normalizedChi2();

    // edm::LogDebug("AlignmentTrackSelector") << " pt,eta,phi,nhit: "
    //  <<pt<<","<<eta<<","<<phi<<","<<nhit;

    if (pt>ptMin_ && pt<ptMax_ 
       && p>pMin_ && p<pMax_ 
       && eta>etaMin_ && eta<etaMax_ 
       && phi>phiMin_ && phi<phiMax_ 
       && nhit>=nHitMin_ && nhit<=nHitMax_
       && chi2n<chi2nMax_) {
      if (this->detailedHitsCheck(trackp, evt)) result.push_back(trackp);
    }
  }

  return result;
}

//-----------------------------------------------------------------------------

bool AlignmentTrackSelector::detailedHitsCheck(const reco::Track *trackp, const edm::Event& evt) const
{
  // checking hit requirements beyond simple number of valid hits

  if (minHitsinTIB_ || minHitsinTOB_ || minHitsinTID_ || minHitsinTEC_
      || minHitsinFPIX_ || minHitsinBPIX_ || nHitMin2D_ || chargeCheck_
      || applyIsolation_ || (seedOnlyFromAbove_ == 1 || seedOnlyFromAbove_ == 2)) {
    // any detailed hit cut is active, so have to check
    
    int nhitinTIB = 0, nhitinTOB = 0, nhitinTID = 0;
    int nhitinTEC = 0, nhitinBPIX = 0, nhitinFPIX = 0;
    unsigned int nHit2D = 0;
    unsigned int thishit = 0;

    for (trackingRecHit_iterator iHit = trackp->recHitsBegin(); iHit != trackp->recHitsEnd(); ++iHit) {
      thishit++;
      const DetId detId((*iHit)->geographicalId());
      const int subdetId = detId.subdetId(); 

      // *** thishit == 1 means last hit in CTF *** 
      // (FIXME: assumption might change or not be valid for all tracking algorthms)
      // ==> for cosmics
      // seedOnlyFrom = 1 is TIB-TOB-TEC tracks only
      // seedOnlyFrom = 2 is TOB-TEC tracks only 
      if (seedOnlyFromAbove_ == 1 && thishit == 1 
          && (subdetId == int(StripSubdetector::TOB) || subdetId == int(StripSubdetector::TEC))){
        return false; 
      }
      if (seedOnlyFromAbove_ == 2 && thishit == 1 && subdetId == int(StripSubdetector::TIB)) {
        return false;
      }

      if (!(*iHit)->isValid()) continue; // only real hits count as in trackp->numberOfValidHits()
      if (detId.det() != DetId::Tracker) {
        edm::LogError("DetectorMismatch") << "@SUB=AlignmentTrackSelector::detailedHitsCheck"
                                          << "DetId.det() != DetId::Tracker (=" << DetId::Tracker
                                          << "), but " << detId.det() << ".";
      }
      const TrackingRecHit* therechit = (*iHit).get();
      if (chargeCheck_ && !(this->isOkCharge(therechit))) return false;
      if (applyIsolation_ && (!this->isIsolated(therechit, evt))) return false;
      if      (StripSubdetector::TIB == subdetId) ++nhitinTIB;
      else if (StripSubdetector::TOB == subdetId) ++nhitinTOB;
      else if (StripSubdetector::TID == subdetId) ++nhitinTID;
      else if (StripSubdetector::TEC == subdetId) ++nhitinTEC;
      else if (                kBPIX == subdetId) ++nhitinBPIX;
      else if (                kFPIX == subdetId) ++nhitinFPIX;
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

bool AlignmentTrackSelector::isOkCharge(const TrackingRecHit* hit) const
{
  if (!hit || !hit->isValid()) return true; 

  // check det and subdet
  const DetId id(hit->geographicalId());
  if (id.det() != DetId::Tracker) {
    edm::LogWarning("DetectorMismatch") << "@SUB=isOkCharge" << "Hit not in tracker!";
    return true;
  }
  if (id.subdetId() == kFPIX || id.subdetId() == kBPIX) {
    return true; // might add some requirement...
  }

  // We are in SiStrip now, so test normal hit:
  const SiStripRecHit2D *stripHit2D = dynamic_cast<const SiStripRecHit2D*>(hit);
  if (stripHit2D) {
    return this->isOkChargeStripHit(stripHit2D);
  }
  // or matched (should not occur anymore due to hit splitting since 20X):
  const SiStripMatchedRecHit2D *matchedHit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
  if (matchedHit) {  
    return (this->isOkChargeStripHit(matchedHit->monoHit())
            && this->isOkChargeStripHit(matchedHit->stereoHit()));
  }
  // or projected (should not occur anymore due to hit splitting since 20X):
  const ProjectedSiStripRecHit2D *projHit = dynamic_cast<const ProjectedSiStripRecHit2D*>(hit);
  if (projHit) {
    return this->isOkChargeStripHit(&projHit->originalHit());
  }

  // and now? SiTrackerMultiRecHit? Not here I guess!
  // Should we throw instead?
  edm::LogError("AlignmentTrackSelector") 
    << "@SUB=isOkCharge" << "Unknown valid tracker hit not in pixel, subdet " << id.subdetId()
    << ", SiTrackerMultiRecHit " << dynamic_cast<const SiTrackerMultiRecHit*>(hit)
    << ", BaseSiTrackerRecHit2DLocalPos " << dynamic_cast<const BaseSiTrackerRecHit2DLocalPos*>(hit); 
  
  return true;
} 

//-----------------------------------------------------------------------------

bool AlignmentTrackSelector::isOkChargeStripHit(const SiStripRecHit2D *siStripRecHit2D) const
{
  double charge = 0.;

  SiStripRecHit2D::ClusterRef cluster(siStripRecHit2D->cluster());
  const std::vector<uint8_t> &amplitudes = cluster->amplitudes();

  for (size_t ia = 0; ia < amplitudes.size(); ++ia) {
    charge += amplitudes[ia];
  }

  return (charge >= minHitChargeStrip_);
}

//-----------------------------------------------------------------------------

bool AlignmentTrackSelector::isIsolated(const TrackingRecHit* therechit, const edm::Event& evt) const
{
	   
  // edm::ESHandle<TrackerGeometry> tracker;
  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits;
  // es.get<TrackerDigiGeometryRecord>().get(tracker);
  evt.getByLabel( rphirecHitsTag_, rphirecHits );
  evt.getByLabel( matchedrecHitsTag_, matchedrecHits ); 

  SiStripRecHit2DCollection::const_iterator istripSt; 
  SiStripMatchedRecHit2DCollection::const_iterator istripStm; 
  const SiStripRecHit2DCollection& stripcollSt = *rphirecHits;
  const SiStripMatchedRecHit2DCollection& stripcollStm = *matchedrecHits;
  
  DetId idet = therechit->geographicalId(); 
  
  // FIXME: instead of looping the full hit collection, we should explore the features of 
  // SiStripRecHit2DCollection::rangeRphi = rphirecHits.get(idet) and loop
  // only from rangeRphi.first until rangeRphi.second
  for( istripSt=stripcollSt.begin(); istripSt!=stripcollSt.end(); ++istripSt ) {
    const SiStripRecHit2D *aHit = &*(istripSt);
    DetId mydet1 = aHit->geographicalId(); 
    if (idet.rawId() != mydet1.rawId()) continue; 
    float theDistance = ( therechit->localPosition() - aHit->localPosition() ).mag();
    // std::cout << "theDistance1 = " << theDistance << "\n";
    if (theDistance > 0.001 && theDistance < minHitIsolation_) return false;
  }
  
  // FIXME: see above
  for( istripStm=stripcollStm.begin(); istripStm!=stripcollStm.end(); ++istripStm ) {
    const SiStripMatchedRecHit2D *aHit = &*(istripStm);
    DetId mydet2 = aHit->geographicalId(); 
    if (idet.rawId() != mydet2.rawId()) continue;
    float theDistance = (therechit->localPosition() - aHit->localPosition()).mag();
    // std::cout << "theDistance1 = " << theDistance << "\n";
    if (theDistance > 0.001 && theDistance < minHitIsolation_) return false;
  }
  return true;
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
	   it!=sortedTracks.end(); ++it) {
	if (n<nHighestPt_) { result.push_back(*it); n++; }
  }

  return result;
}

