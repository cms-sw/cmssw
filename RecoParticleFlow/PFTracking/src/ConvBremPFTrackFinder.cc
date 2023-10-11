#include "RecoParticleFlow/PFTracking/interface/ConvBremPFTrackFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "TMath.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "TMVA/MethodBDT.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace edm;
using namespace std;
using namespace reco;

ConvBremPFTrackFinder::ConvBremPFTrackFinder(const TransientTrackBuilder& builder,
                                             double mvaBremConvCutBarrelLowPt,
                                             double mvaBremConvCutBarrelHighPt,
                                             double mvaBremConvCutEndcapsLowPt,
                                             double mvaBremConvCutEndcapsHighPt)
    : builder_(builder),
      mvaBremConvCutBarrelLowPt_(mvaBremConvCutBarrelLowPt),
      mvaBremConvCutBarrelHighPt_(mvaBremConvCutBarrelHighPt),
      mvaBremConvCutEndcapsLowPt_(mvaBremConvCutEndcapsLowPt),
      mvaBremConvCutEndcapsHighPt_(mvaBremConvCutEndcapsHighPt) {}
ConvBremPFTrackFinder::~ConvBremPFTrackFinder() {}

void ConvBremPFTrackFinder::runConvBremFinder(const Handle<PFRecTrackCollection>& thePfRecTrackCol,
                                              const Handle<VertexCollection>& primaryVertex,
                                              const edm::Handle<reco::PFDisplacedTrackerVertexCollection>& pfNuclears,
                                              const edm::Handle<reco::PFConversionCollection>& pfConversions,
                                              const edm::Handle<reco::PFV0Collection>& pfV0,
                                              const convbremhelpers::HeavyObjectCache* cache,
                                              bool useNuclear,
                                              bool useConversions,
                                              bool useV0,
                                              const reco::PFClusterCollection& theEClus,
                                              const reco::GsfPFRecTrack& gsfpfrectk) {
  found_ = false;

  LogDebug("ConvBremPFTrackFinder")  << "runConvBremFinder:: Entering ";

  const reco::GsfTrackRef& refGsf = gsfpfrectk.gsfTrackRef(); 
  float refGsfEta = refGsf->eta();
  float refGsfPt = refGsf->pt();
  float refGsfPhi = refGsf->phi();
  float gsfR = sqrt(refGsf->innerPosition().x() * refGsf->innerPosition().x() +
		    refGsf->innerPosition().y() * refGsf->innerPosition().y());
  // direction of the Gsf track
  GlobalVector direction(refGsf->innerMomentum().x(), refGsf->innerMomentum().y(), refGsf->innerMomentum().z());   
  float refGsfPtMode = refGsf->ptMode();
 
  const reco::PFRecTrackRef& pfTrackRef = gsfpfrectk.kfPFRecTrackRef();
  vector<PFBrem> primPFBrem = gsfpfrectk.PFRecBrem();

  const PFRecTrackCollection& PfRTkColl = *(thePfRecTrackCol.product());
  reco::PFRecTrackCollection::const_iterator pft = PfRTkColl.begin();
  reco::PFRecTrackCollection::const_iterator pftend = PfRTkColl.end();

  vector<PFRecTrackRef> AllPFRecTracks;
  AllPFRecTracks.clear();
  unsigned int ipft = 0;

  for (; pft != pftend; ++pft, ipft++) {
    // do not consider the kf track already associated to the seed
    if (pfTrackRef.isNonnull())
      if (pfTrackRef->trackRef() == pft->trackRef())
        continue;

    PFRecTrackRef pfRecTrRef(thePfRecTrackCol, ipft);
    TrackRef trackRef = pfRecTrRef->trackRef();
    reco::TrackBaseRef selTrackBaseRef(trackRef);
    
    LogDebug("ConvBremPFTrackFinder") << "runConvBremFinder:: pushing_back High Purity " << pft->trackRef()->pt() << " eta,phi "
				      << pft->trackRef()->eta() << ", " << pft->trackRef()->phi() << " Memory Address Ref  " << &*trackRef
				      << " Memory Address BaseRef  " << &*selTrackBaseRef;
    AllPFRecTracks.push_back(pfRecTrRef);
  }

  if (useConversions) {
    const PFConversionCollection& PfConvColl = *(pfConversions.product());
    for (unsigned i = 0; i < PfConvColl.size(); i++) {
      reco::PFConversionRef convRef(pfConversions, i);

      unsigned int trackSize = (convRef->pfTracks()).size();
      if (convRef->pfTracks().size() < 2)
        continue;
      for (unsigned iTk = 0; iTk < trackSize; iTk++) {
        PFRecTrackRef compPFTkRef = convRef->pfTracks()[iTk];
        reco::TrackBaseRef newTrackBaseRef(compPFTkRef->trackRef());
        // do not consider the kf track already associated to the seed
        if (pfTrackRef.isNonnull()) {
          reco::TrackBaseRef primaryTrackBaseRef(pfTrackRef->trackRef());
          if (primaryTrackBaseRef == newTrackBaseRef)
            continue;
        }
        bool notFound = true;
        for (unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
          reco::TrackBaseRef selTrackBaseRef(AllPFRecTracks[iPF]->trackRef());
	  
	  LogDebug("ConvBremPFTrackFinder") << "## Track 1 HP pt " << AllPFRecTracks[iPF]->trackRef()->pt() << " eta, phi "
					    << AllPFRecTracks[iPF]->trackRef()->eta() << ", " << AllPFRecTracks[iPF]->trackRef()->phi()
					    << " Memory Address Ref  " << &(*AllPFRecTracks[iPF]->trackRef()) << " Memory Address BaseRef  "
					    << &*selTrackBaseRef;
	  LogDebug("ConvBremPFTrackFinder") << "** Track 2 CONV pt " << compPFTkRef->trackRef()->pt() << " eta, phi "
					    << compPFTkRef->trackRef()->eta() << ", " << compPFTkRef->trackRef()->phi() << " Memory Address Ref "
					    << &*compPFTkRef->trackRef() << " Memory Address BaseRef " << &*newTrackBaseRef;
          //if(selTrackBaseRef == newTrackBaseRef ||  AllPFRecTracks[iPF]->trackRef()== compPFTkRef->trackRef()) {
          if (AllPFRecTracks[iPF]->trackRef() == compPFTkRef->trackRef()) {
	    LogDebug("ConvBremPFTrackFinder") << "  SAME BREM REF " << endl;
            notFound = false;
          }
        }
        if (notFound) {
	  LogDebug("ConvBremPFTrackFinder") << "runConvBremFinder:: pushing_back Conversions " << compPFTkRef->trackRef()->pt() << " eta,phi "
					    << compPFTkRef->trackRef()->eta() << " phi " << compPFTkRef->trackRef()->phi();
          AllPFRecTracks.push_back(compPFTkRef);
        }
      }
    }
  }

  if (useNuclear) {
    const PFDisplacedTrackerVertexCollection& PfNuclColl = *(pfNuclears.product());
    for (unsigned i = 0; i < PfNuclColl.size(); i++) {
      const reco::PFDisplacedTrackerVertexRef dispacedVertexRef(pfNuclears, i);
      unsigned int trackSize = dispacedVertexRef->pfRecTracks().size();
      for (unsigned iTk = 0; iTk < trackSize; iTk++) {
        reco::PFRecTrackRef newPFRecTrackRef = dispacedVertexRef->pfRecTracks()[iTk];
        reco::TrackBaseRef newTrackBaseRef(newPFRecTrackRef->trackRef());
        // do not consider the kf track already associated to the seed
        if (pfTrackRef.isNonnull()) {
          reco::TrackBaseRef primaryTrackBaseRef(pfTrackRef->trackRef());
          if (primaryTrackBaseRef == newTrackBaseRef)
            continue;
        }
        bool notFound = true;
        for (unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
          reco::TrackBaseRef selTrackBaseRef(AllPFRecTracks[iPF]->trackRef());
          if (selTrackBaseRef == newTrackBaseRef)
            notFound = false;
        }
        if (notFound) {
	  LogDebug("ConvBremPFTrackFinder") << "runConvBremFinder:: pushing_back displaced Vertex pt " << newPFRecTrackRef->trackRef()->pt()
					    << " eta,phi " << newPFRecTrackRef->trackRef()->eta() << ", " << newPFRecTrackRef->trackRef()->phi();
          AllPFRecTracks.push_back(newPFRecTrackRef);
        }
      }
    }
  }

  if (useV0) {
    const PFV0Collection& PfV0Coll = *(pfV0.product());
    for (unsigned i = 0; i < PfV0Coll.size(); i++) {
      reco::PFV0Ref v0Ref(pfV0, i);
      unsigned int trackSize = (v0Ref->pfTracks()).size();
      for (unsigned iTk = 0; iTk < trackSize; iTk++) {
        reco::PFRecTrackRef newPFRecTrackRef = (v0Ref->pfTracks())[iTk];
        reco::TrackBaseRef newTrackBaseRef(newPFRecTrackRef->trackRef());
        // do not consider the kf track already associated to the seed
        if (pfTrackRef.isNonnull()) {
          reco::TrackBaseRef primaryTrackBaseRef(pfTrackRef->trackRef());
          if (primaryTrackBaseRef == newTrackBaseRef)
            continue;
        }
        bool notFound = true;
        for (unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
          reco::TrackBaseRef selTrackBaseRef(AllPFRecTracks[iPF]->trackRef());
          if (selTrackBaseRef == newTrackBaseRef)
            notFound = false;
        }
        if (notFound) {
	  LogDebug("ConvBremPFTrackFinder") << "runConvBremFinder:: pushing_back V0 " << newPFRecTrackRef->trackRef()->pt() << " eta,phi "
					    << newPFRecTrackRef->trackRef()->eta() << ", " << newPFRecTrackRef->trackRef()->phi();
          AllPFRecTracks.push_back(newPFRecTrackRef);
        }
      }
    }
  }

  pfRecTrRef_vec_.clear();

  for (unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
    double dphi = std::abs(::deltaPhi(AllPFRecTracks[iPF]->trackRef()->phi(), refGsfPhi));
    double deta = std::abs(AllPFRecTracks[iPF]->trackRef()->eta() - refGsfEta);

    // limiting the phase space (just for saving cpu-time)
    if (std::abs(dphi) > 1.0 || std::abs(deta) > 0.4)
      continue;

    double minDEtaBremKF = std::numeric_limits<double>::max();
    double minDPhiBremKF = std::numeric_limits<double>::max();
    double minDRBremKF2 = std::numeric_limits<double>::max();
    double minDEtaBremKFPos = std::numeric_limits<double>::max();
    double minDPhiBremKFPos = std::numeric_limits<double>::max();
    double minDRBremKFPos2 = std::numeric_limits<double>::max();
    reco::TrackRef trkRef = AllPFRecTracks[iPF]->trackRef();

    double secEta = trkRef->innerMomentum().eta();
    double secPhi = trkRef->innerMomentum().phi();

    double posEta = trkRef->innerPosition().eta();
    double posPhi = trkRef->innerPosition().phi();

    for (unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
      if (primPFBrem[ipbrem].indTrajPoint() == 99)
        continue;
      const reco::PFTrajectoryPoint& atPrimECAL =
          primPFBrem[ipbrem].extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
      if (!atPrimECAL.isValid())
        continue;
      double bremEta = atPrimECAL.momentum().Eta();
      double bremPhi = atPrimECAL.momentum().Phi();
            
      double deta = std::abs(bremEta - secEta);
      double dphi = std::abs(::deltaPhi(bremPhi,secPhi));
      double DR2 = deta * deta + dphi * dphi;

      double detaPos = std::abs(bremEta - posEta);
      double dphiPos = std::abs(::deltaPhi(bremPhi, posPhi));
      double DRPos2 = detaPos * detaPos + dphiPos * dphiPos;

      // find the closest track tangent
      if (DR2 < minDRBremKF2) {
        minDRBremKF2 = DR2;
        minDEtaBremKF = deta;
        minDPhiBremKF = std::abs(dphi);
      }

      if (DRPos2 < minDRBremKFPos2) {
        minDRBremKFPos2 = DRPos2;
        minDEtaBremKFPos = detaPos;
        minDPhiBremKFPos = std::abs(dphiPos);
      }
    }
  
    // secR
    secR = sqrt(trkRef->innerPosition().x() * trkRef->innerPosition().x() +
                trkRef->innerPosition().y() * trkRef->innerPosition().y());

    // apply loose selection (to be parallel) between the secondary track and brem-tangents.
    // Moreover if the secR is internal with respect to the GSF track by two pixel layers discard it.
    if ((minDPhiBremKF < 0.1 || minDPhiBremKFPos < 0.1) && (minDEtaBremKF < 0.02 || minDEtaBremKFPos < 0.02) &&
        secR > (gsfR - 8)) {
  
      LogDebug("ConvBremPFTrackFinder") << "runConvBremFinder:: OK Find track and BREM close "
					<< " MinDphi " << minDPhiBremKF << " MinDeta " << minDEtaBremKF;

      float MinDist = std::numeric_limits<float>::max();
      float EE_calib = 0.;
      PFRecTrack pfrectrack = *AllPFRecTracks[iPF];
      // Find and ECAL associated cluster
      for (PFClusterCollection::const_iterator clus = theEClus.begin(); clus != theEClus.end(); clus++) {
        // Removed unusd variable, left this in case it has side effects
        clus->position();
        double dist = -1.;
        PFCluster clust = *clus;
        clust.calculatePositionREP();
        dist = pfrectrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax).isValid()
                   ? LinkByRecHit::testTrackAndClusterByRecHit(pfrectrack, clust)
                   : -1.;

        if (dist > 0. && dist < MinDist) {
          MinDist = dist;
          EE_calib = cache->pfcalib_->energyEm(*clus, 0.0, 0.0, false);
        }
      } 
      if (MinDist > 0. && MinDist < 100000.) {
        // compute all the input variables for conv brem selection

        secPout = sqrt(trkRef->outerMomentum().x() * trkRef->outerMomentum().x() +
                       trkRef->outerMomentum().y() * trkRef->outerMomentum().y() +
                       trkRef->outerMomentum().z() * trkRef->outerMomentum().z());

        secPin = sqrt(trkRef->innerMomentum().x() * trkRef->innerMomentum().x() +
                      trkRef->innerMomentum().y() * trkRef->innerMomentum().y() +
                      trkRef->innerMomentum().z() * trkRef->innerMomentum().z());

        // maybe put innter momentum pt?
        ptRatioGsfKF = trkRef->pt() / refGsfPtMode;

        Vertex dummy;
        const Vertex* pv = &dummy;
        edm::Ref<VertexCollection> pvRef;
        if (!primaryVertex->empty()) {
          pv = &*primaryVertex->begin();
          // we always use the first vertex (at the moment)
          pvRef = edm::Ref<VertexCollection>(primaryVertex, 0);
        } else {  // create a dummy PV
          Vertex::Error e;
          e(0, 0) = 0.0015 * 0.0015;
          e(1, 1) = 0.0015 * 0.0015;
          e(2, 2) = 15. * 15.;
          Vertex::Point p(0, 0, 0);
          dummy = Vertex(p, e, 0, 0, 0);
        }

        TransientTrack transientTrack = builder_.build(*trkRef);
        sTIP = IPTools::signedTransverseImpactParameter(transientTrack, direction, *pv).second.significance();

        Epout = EE_calib / secPout;

        // eta distance brem-secondary kf track
        detaBremKF = minDEtaBremKF;

        // Number of commont hits
        unsigned int tmpSh = 0;
        //uint ish=0;
        int kfhitcounter = 0;
        for (auto const& nhit : refGsf->recHits())
          if (nhit->isValid()) {
            kfhitcounter = 0;
            for (auto const& ihit : trkRef->recHits())
              if (ihit->isValid()) {
                // method 1
                if (nhit->sharesInput(ihit, TrackingRecHit::all))
                  tmpSh++;
                kfhitcounter++;
                // method 2 to switch in case of problem with rechit collections
                // if((ihit->geographicalId()==nhit->geographicalId())&&
                //     ((nhit->localPosition()-ihit->localPosition()).mag()<0.01)) ish++;
              }
          }

        nHITS1 = tmpSh;

        double mvaValue = -10;
        double cutvalue = -10;

        float vars[6] = {secR, sTIP, nHITS1, Epout, detaBremKF, ptRatioGsfKF};
        if (refGsfPt < 20 && std::abs(refGsfEta) < 1.479) {
          mvaValue = cache->gbrBarrelLowPt_->GetClassifier(vars);
          cutvalue = mvaBremConvCutBarrelLowPt_;
        }
        if (refGsfPt > 20 && std::abs(refGsfEta) < 1.479) {
          mvaValue = cache->gbrBarrelHighPt_->GetClassifier(vars);
          cutvalue = mvaBremConvCutBarrelHighPt_;
        }
        if (refGsfPt < 20 && std::abs(refGsfEta) > 1.479) {
          mvaValue = cache->gbrEndcapsLowPt_->GetClassifier(vars);
          cutvalue = mvaBremConvCutEndcapsLowPt_;
        }
        if (refGsfPt > 20 && std::abs(refGsfEta) > 1.479) {
          mvaValue = cache->gbrEndcapsHighPt_->GetClassifier(vars);
          cutvalue = mvaBremConvCutEndcapsHighPt_;
        }
	
	LogDebug("ConvBremPFTrackFinder") << "Gsf track Pt, Eta " << refGsfPt << " " << refGsfEta;
	LogDebug("ConvBremPFTrackFinder") << "Cutvalue " << cutvalue;

        if ((kfhitcounter - nHITS1) <= 3 && nHITS1 > 3)
          mvaValue = 2;  // this means duplicates tracks, very likely not physical

        LogDebug("ConvBremPFTrackFinder") << " The input variables for conv brem tracks identification "
					  << " secR          " << secR << " gsfR " << gsfR
					  << " N shared hits " << nHITS1 
					  << " sTIP          " << sTIP
					  << " detaBremKF    " << detaBremKF
					  << " E/pout        " << Epout
					  << " ptRatioKFGsf  " << ptRatioGsfKF
					  << " ***** MVA ***** " << mvaValue;

        if (mvaValue > cutvalue) {
          found_ = true;
          pfRecTrRef_vec_.push_back(AllPFRecTracks[iPF]);
        }
      }  // end MinDIST
    }    // end selection kf - brem tangents
  }      // loop on the kf tracks
}
