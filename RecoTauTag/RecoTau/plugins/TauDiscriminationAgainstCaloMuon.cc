
/* 
 * class TauDiscriminationAgainstCaloMuon
 * created : Nov 20 2010
 * revised : 
 * Authors : Christian Veelken (UC Davis)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <TVector3.h>
#include <TMath.h>

#include <string>

using namespace reco;

// define acess to lead. track for CaloTaus
template <typename T>
class TauLeadTrackExtractor
{
 public:
  reco::TrackRef getLeadTrack(const T& tau) const
  {
    return tau.leadTrack();
  }
  double getTrackPtSum(const T& tau) const
  {
    double trackPtSum = 0.;
    for ( TrackRefVector::const_iterator signalTrack = tau.signalTracks().begin();
	  signalTrack != tau.signalTracks().end(); ++signalTrack ) {
      trackPtSum += (*signalTrack)->pt();
    }
    return trackPtSum;
  }
};

// define acess to lead. track for PFTaus
template <>
class TauLeadTrackExtractor<reco::PFTau>
{
 public:
  reco::TrackRef getLeadTrack(const reco::PFTau& tau) const
  {
    return tau.leadPFChargedHadrCand()->trackRef();
  }
  double getTrackPtSum(const reco::PFTau& tau) const
  {
    double trackPtSum = 0.;
    for ( std::vector<PFCandidatePtr>::const_iterator signalTrack = tau.signalPFChargedHadrCands().begin();
	  signalTrack != tau.signalPFChargedHadrCands().end(); ++signalTrack ) {
      trackPtSum += (*signalTrack)->pt();
    }
    return trackPtSum;
  }
};

template<class TauType, class TauDiscriminator>
class TauDiscriminationAgainstCaloMuon : public TauDiscriminationProducerBase<TauType, TauDiscriminator>
{
 public:
  // setup framework types for this tautype
  typedef std::vector<TauType>    TauCollection; 
  typedef edm::Ref<TauCollection> TauRef;    

  explicit TauDiscriminationAgainstCaloMuon(const edm::ParameterSet&);
  ~TauDiscriminationAgainstCaloMuon() {} 

  // called at the beginning of every event
  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double discriminate(const TauRef&);

 private:  
  edm::InputTag srcEcalRecHitsBarrel_;
  edm::Handle<EcalRecHitCollection> ebRecHits_;
  edm::InputTag srcEcalRecHitsEndcap_;
  edm::Handle<EcalRecHitCollection> eeRecHits_;
  edm::InputTag srcHcalRecHits_;
  edm::Handle<HBHERecHitCollection> hbheRecHits_;

  edm::InputTag srcVertex_;
  GlobalPoint eventVertexPosition_;

  const TransientTrackBuilder* trackBuilder_;
  const CaloGeometry* caloGeometry_;

  TauLeadTrackExtractor<TauType> leadTrackExtractor_;

  double minLeadTrackPt_;
  double minLeadTrackPtFraction_;

  double drEcal_;
  double drHcal_;

  double maxEnEcal_;
  double maxEnHcal_;

  double maxEnToTrackRatio_;
};

template<class TauType, class TauDiscriminator>
TauDiscriminationAgainstCaloMuon<TauType, TauDiscriminator>::TauDiscriminationAgainstCaloMuon(const edm::ParameterSet& cfg)
  : TauDiscriminationProducerBase<TauType, TauDiscriminator>(cfg) 
{
  srcEcalRecHitsBarrel_ = cfg.getParameter<edm::InputTag>("srcEcalRecHitsBarrel");
  srcEcalRecHitsEndcap_ = cfg.getParameter<edm::InputTag>("srcEcalRecHitsEndcap");
  srcHcalRecHits_ = cfg.getParameter<edm::InputTag>("srcHcalRecHits");

  srcVertex_ = cfg.getParameter<edm::InputTag>("srcVertex");

  minLeadTrackPt_ = cfg.getParameter<double>("minLeadTrackPt");
  minLeadTrackPtFraction_ = cfg.getParameter<double>("minLeadTrackPtFraction");

  drEcal_ = cfg.getParameter<double>("dRecal");
  drHcal_ = cfg.getParameter<double>("dRhcal");

  maxEnEcal_ = cfg.getParameter<double>("maxEnEcal");
  maxEnHcal_ = cfg.getParameter<double>("maxEnHcal");

  maxEnToTrackRatio_ = cfg.getParameter<double>("maxEnToTrackRatio");
}

template<class TauType, class TauDiscriminator>
void TauDiscriminationAgainstCaloMuon<TauType, TauDiscriminator>::beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  evt.getByLabel(srcEcalRecHitsBarrel_, ebRecHits_);
  evt.getByLabel(srcEcalRecHitsEndcap_, eeRecHits_);
  evt.getByLabel(srcHcalRecHits_, hbheRecHits_);
  
  edm::ESHandle<TransientTrackBuilder> trackBuilderHandle;
  evtSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", trackBuilderHandle);
  trackBuilder_ = trackBuilderHandle.product();
  if ( !trackBuilder_ ) {
    edm::LogError ("TauDiscriminationAgainstCaloMuon::discriminate")
      << " Failed to access TransientTrackBuilder !!";
  }

  edm::ESHandle<CaloGeometry> caloGeometryHandle;
  evtSetup.get<CaloGeometryRecord>().get(caloGeometryHandle);
  caloGeometry_ = caloGeometryHandle.product();
  if ( !caloGeometry_ ) {
    edm::LogError ("TauDiscriminationAgainstCaloMuon::discriminate")
      << " Failed to access CaloGeometry !!";
  }

  edm::Handle<reco::VertexCollection> vertices;
  evt.getByLabel(srcVertex_, vertices);
  eventVertexPosition_ = GlobalPoint(0., 0., 0.);
  if ( vertices->size() >= 1 ) {
    const reco::Vertex& thePrimaryEventVertex = (*vertices->begin());
    eventVertexPosition_ = GlobalPoint(thePrimaryEventVertex.x(), thePrimaryEventVertex.y(), thePrimaryEventVertex.z());
  }
}		

double compEcalEnergySum(const EcalRecHitCollection& ecalRecHits, 
			 const CaloSubdetectorGeometry* detGeometry, 
			 const reco::TransientTrack& transientTrack, double dR, 
			 const GlobalPoint& eventVertexPosition)
{
  double ecalEnergySum = 0.;
  for ( EcalRecHitCollection::const_iterator ecalRecHit = ecalRecHits.begin();
	ecalRecHit != ecalRecHits.end(); ++ecalRecHit ) {
    const CaloCellGeometry* cellGeometry = detGeometry->getGeometry(ecalRecHit->detid());
    
    if ( !cellGeometry ) {
      edm::LogError ("compEcalEnergySum") 
	<< " Failed to access ECAL geometry for detId = " << ecalRecHit->detid().rawId()
	<< " --> skipping !!";
      continue;
    }

    const GlobalPoint& cellPosition = cellGeometry->getPosition();

//--- CV: speed up computation by requiring eta-phi distance
//        between cell position and track direction to be dR < 0.5
    Vector3DBase<float, GlobalTag> cellPositionRelVertex = (cellPosition) - eventVertexPosition;
    if ( deltaR(cellPositionRelVertex.eta(), cellPositionRelVertex.phi(), 
		transientTrack.track().eta(), transientTrack.track().phi()) > 0.5 ) continue;

    TrajectoryStateClosestToPoint dcaPosition = transientTrack.trajectoryStateClosestToPoint(cellPosition);
    
    Vector3DBase<float, GlobalTag> d = (cellPosition - dcaPosition.position());

    TVector3 d3(d.x(), d.y(), d.z());
    TVector3 dir(transientTrack.track().px(), transientTrack.track().py(), transientTrack.track().pz());

    double dPerp = d3.Cross(dir.Unit()).Mag();
    double dParl = TVector3(cellPosition.x(), cellPosition.y(), cellPosition.z()).Dot(dir.Unit());
    
    if ( dPerp < dR && dParl > 100. ) {
      ecalEnergySum += ecalRecHit->energy();
    }
  }
  
  return ecalEnergySum;
}

double compHcalEnergySum(const HBHERecHitCollection& hcalRecHits, 
			 const CaloSubdetectorGeometry* hbGeometry, const CaloSubdetectorGeometry* heGeometry, 
			 const reco::TransientTrack& transientTrack, double dR, 
			 const GlobalPoint& eventVertexPosition)
{
  double hcalEnergySum = 0.;
  for ( HBHERecHitCollection::const_iterator hcalRecHit = hcalRecHits.begin();
	hcalRecHit != hcalRecHits.end(); ++hcalRecHit ) {
    const CaloCellGeometry* hbCellGeometry = hbGeometry->getGeometry(hcalRecHit->detid());
    const CaloCellGeometry* heCellGeometry = heGeometry->getGeometry(hcalRecHit->detid());

    const GlobalPoint* cellPosition = 0;
    if ( hbCellGeometry ) cellPosition = &(hbCellGeometry->getPosition());
    if ( heCellGeometry ) cellPosition = &(heCellGeometry->getPosition());

    if ( !cellPosition ) {
      edm::LogError ("compHcalEnergySum") 
	<< " Failed to access HCAL geometry for detId = " << hcalRecHit->detid().rawId()
	<< " --> skipping !!";
      continue;
    }

//--- CV: speed up computation by requiring eta-phi distance
//        between cell position and track direction to be dR < 0.5
    Vector3DBase<float, GlobalTag> cellPositionRelVertex = (*cellPosition) - eventVertexPosition;
    if ( deltaR(cellPositionRelVertex.eta(), cellPositionRelVertex.phi(), 
		transientTrack.track().eta(), transientTrack.track().phi()) > 0.5 ) continue;

    TrajectoryStateClosestToPoint dcaPosition = transientTrack.trajectoryStateClosestToPoint(*cellPosition);
    
    Vector3DBase<float, GlobalTag> d = ((*cellPosition) - dcaPosition.position());

    TVector3 d3(d.x(), d.y(), d.z());
    TVector3 dir(transientTrack.track().px(), transientTrack.track().py(), transientTrack.track().pz());

    double dPerp = d3.Cross(dir.Unit()).Mag();
    double dParl = TVector3(cellPosition->x(), cellPosition->y(), cellPosition->z()).Dot(dir.Unit());

    if ( dPerp < dR && dParl > 100. ) {
      hcalEnergySum += hcalRecHit->energy();
    }
  }
  
  return hcalEnergySum;
}

template<class TauType, class TauDiscriminator>
double TauDiscriminationAgainstCaloMuon<TauType, TauDiscriminator>::discriminate(const TauRef& tau)
{
  if ( !(trackBuilder_ && caloGeometry_) ) return 0.;

  const CaloSubdetectorGeometry* ebGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorGeometry* eeGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  const CaloSubdetectorGeometry* hbGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* heGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  TrackRef leadTrackRef = leadTrackExtractor_.getLeadTrack(*tau);

  if ( (leadTrackRef.isAvailable() || leadTrackRef.isTransient()) && leadTrackRef.isNonnull() ) {
    double leadTrackPt = leadTrackRef->pt();
    double trackPtSum = leadTrackExtractor_.getTrackPtSum(*tau);
    
    double leadTrackPtFraction = ( trackPtSum > 0. ) ? (leadTrackPt/trackPtSum) : -1.;
    
    if ( leadTrackPt > minLeadTrackPt_ && leadTrackPtFraction > minLeadTrackPtFraction_ ) {
      reco::TransientTrack transientTrack = trackBuilder_->build(leadTrackRef);

      double ebEnergySum = compEcalEnergySum(*ebRecHits_, ebGeometry, transientTrack, drEcal_, eventVertexPosition_);
      double eeEnergySum = compEcalEnergySum(*eeRecHits_, eeGeometry, transientTrack, drEcal_, eventVertexPosition_);
      double ecalEnergySum = ebEnergySum + eeEnergySum;
      
      double hbheEnergySum = compHcalEnergySum(*hbheRecHits_, hbGeometry, heGeometry, transientTrack, drHcal_, eventVertexPosition_);
      
      double caloEnergySum = ecalEnergySum + hbheEnergySum;

      if ( ecalEnergySum <  maxEnEcal_ &&
	   hbheEnergySum <  maxEnHcal_ &&
	   caloEnergySum < (maxEnToTrackRatio_*leadTrackPt) ) return 0.;
    }
  }

  return 1.;
}

#include "FWCore/Framework/interface/MakerMacros.h"

typedef TauDiscriminationAgainstCaloMuon<PFTau, PFTauDiscriminator> PFRecoTauDiscriminationAgainstCaloMuon;
typedef TauDiscriminationAgainstCaloMuon<CaloTau, CaloTauDiscriminator> CaloRecoTauDiscriminationAgainstCaloMuon;

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstCaloMuon);
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationAgainstCaloMuon);
