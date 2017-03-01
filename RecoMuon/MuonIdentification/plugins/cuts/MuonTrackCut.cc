#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonTrackCut : public CutApplicatorBase
{
public:
  MuonTrackCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }
  double value(const reco::CandidatePtr&) const override final;

private:
  // inner track selection cuts
  int minTrackerLayersWithMeasurement_, minPixelLayersWithMeasurement_, minNumberOfValidPixelHits_;
  double minValidFraction_;
  reco::Track::TrackQuality trackQuality_;
  int minNumberOfValidMuonHits_;

  bool doInnerTrack_, doGlobalTrack_;

};
DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  MuonTrackCut, "MuonTrackCut");

// Define constructors and initialization routines
MuonTrackCut::MuonTrackCut(const edm::ParameterSet& c):
  CutApplicatorBase(c),
  minTrackerLayersWithMeasurement_(-1),
  minPixelLayersWithMeasurement_(-1),
  minNumberOfValidPixelHits_(-1),
  minValidFraction_(-1),
  trackQuality_(reco::Track::undefQuality),
  minNumberOfValidMuonHits_(-1),
  doInnerTrack_(false), doGlobalTrack_(false)
{
  
  if ( c.existsAs<edm::ParameterSet>("innerTrack") )
  {
    doInnerTrack_ = true;

    const edm::ParameterSet cc = c.getParameter<edm::ParameterSet>("innerTrack");
    if ( cc.exists("minTrackerLayersWithMeasurement") ) minTrackerLayersWithMeasurement_ = cc.getParameter<int>("minTrackerLayersWithMeasurement");
    if ( cc.exists("minPixelLayersWithMeasurement") ) minPixelLayersWithMeasurement_ = cc.getParameter<int>("minPixelLayersWithMeasurement");
    if ( cc.exists("minNumberOfValidPixelHits") ) minNumberOfValidPixelHits_ = cc.getParameter<int>("minNumberOfValidPixelHits");
    if ( cc.exists("minValidFraction") ) minValidFraction_ = cc.getParameter<double>("minValidFraction");
    const std::string trackQualityStr = cc.exists("trackQuality") ? cc.getParameter<std::string>("trackQuality") : "";
    trackQuality_ = reco::Track::qualityByName(trackQualityStr);
  }
  if ( c.existsAs<edm::ParameterSet>("globalTrack") )
  {
    doGlobalTrack_ = true;

    const edm::ParameterSet cc = c.getParameter<edm::ParameterSet>("globalTrack");
    //if ( cc.exists("maxNormalizedChi2") ) maxNormalizedChi2_ = cc.getParameter<double>("maxNormalizedChi2");
    if ( cc.exists("minNumberOfValidMuonHits") ) minNumberOfValidMuonHits_ = cc.getParameter<int>("minNumberOfValidMuonHits");
  }

}

// Functors for evaluation
CutApplicatorBase::result_type MuonTrackCut::operator()(const reco::MuonPtr& muon) const
{
  if ( doInnerTrack_ )
  {
    const reco::TrackRef t = muon->innerTrack();
    if ( t.isNull() ) return false;
    const auto& h = t->hitPattern();
    if ( trackQuality_ != reco::Track::undefQuality and !t->quality(trackQuality_) ) return false;
    if ( h.trackerLayersWithMeasurement() < minTrackerLayersWithMeasurement_ ) return false;
    if ( h.pixelLayersWithMeasurement() < minPixelLayersWithMeasurement_ ) return false;
    if ( h.numberOfValidPixelHits() < minNumberOfValidPixelHits_ ) return false;
    if ( t->validFraction() <= minValidFraction_ ) return false;
  }
  if ( doGlobalTrack_ )
  {
    const reco::TrackRef t = muon->globalTrack();
    if ( t.isNull() ) return false;
    const auto& h = t->hitPattern();
    if ( h.numberOfValidMuonHits() < minNumberOfValidMuonHits_ ) return false;
    // if ( t->normalizedChi2() > maxNormalizedChi2_ ) return false; Not used for 
  }

  return true;
}

double MuonTrackCut::value(const reco::CandidatePtr& cand) const
{
  const reco::MuonPtr muon(cand);
  if ( doInnerTrack_ )
  {
    const reco::TrackRef t = muon->innerTrack();
    if ( t.isNull() ) return 0;
    const auto& h = t->hitPattern();
    if ( trackQuality_ != reco::Track::undefQuality and !t->quality(trackQuality_) ) return t->quality(trackQuality_);
    if ( h.trackerLayersWithMeasurement() < minTrackerLayersWithMeasurement_ ) return h.trackerLayersWithMeasurement();
    if ( h.pixelLayersWithMeasurement() < minPixelLayersWithMeasurement_ ) return h.pixelLayersWithMeasurement();
    if ( h.numberOfValidPixelHits() < minNumberOfValidPixelHits_ ) return h.numberOfValidPixelHits();
    if ( t->validFraction() <= minValidFraction_ ) return t->validFraction();

    return t->validFraction();
  }
  if ( doGlobalTrack_ )
  {
    const reco::TrackRef t = muon->globalTrack();
    if ( t.isNull() ) return 0;
    const auto& h = t->hitPattern();
    if ( h.numberOfValidMuonHits() < minNumberOfValidMuonHits_ ) return h.numberOfValidMuonHits();
    // if ( t->normalizedChi2() > maxNormalizedChi2_ ) return false; Not used for 

    return h.numberOfValidMuonHits();
  }

  return 0;
}
