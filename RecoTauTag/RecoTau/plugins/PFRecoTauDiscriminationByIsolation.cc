#include <functional>
#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "RecoTauTag/TauTagTools/interface/PFTauQualityCutWrapper.h"
#include "RecoTauTag/RecoTau/interface/ConeTools.h"

/* class PFRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Thu Aug 13 14:44:40 PDT 2009
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg),
 * Christian Veelken (veelken@fnal.gov ; UC Davis),
 *                Evan K. Friis (friis@physics.ucdavis.edu ; UC Davis)
 */

using namespace reco;
using namespace std;

class PFRecoTauDiscriminationByIsolation :
  public PFTauDiscriminationProducerBase  {
  public:
    explicit PFRecoTauDiscriminationByIsolation(
        const edm::ParameterSet& pset):
      PFTauDiscriminationProducerBase(pset),
      qualityCuts_(pset.getParameter<edm::ParameterSet>("qualityCuts"))  {
        includeTracks_ = pset.getParameter<bool>(
            "ApplyDiscriminationByTrackerIsolation");
        includeGammas_ = pset.getParameter<bool>(
            "ApplyDiscriminationByECALIsolation");

        applyOccupancyCut_ = pset.getParameter<bool>("applyOccupancyCut");
        maximumOccupancy_ = pset.getParameter<uint32_t>("maximumOccupancy");

        applySumPtCut_ = pset.getParameter<bool>("applySumPtCut");
        maximumSumPt_ = pset.getParameter<double>("maximumSumPtCut");

        applyRelativeSumPtCut_ = pset.getParameter<bool>(
            "applyRelativeSumPtCut");
        maximumRelativeSumPt_ = pset.getParameter<double>(
            "relativeSumPtCut");

        pvProducer_ = pset.getParameter<edm::InputTag>("PVProducer");

        if (pset.exists("customOuterCone")) {
          customIsoCone_ = pset.getParameter<double>("customOuterCone");
        } else {
          customIsoCone_ = -1;
        }
      }

    ~PFRecoTauDiscriminationByIsolation(){}

    void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup);
    double discriminate(const PFTauRef& pfTau);

  private:
    PFTauQualityCutWrapper qualityCuts_;

    bool includeTracks_;
    bool includeGammas_;

    bool applyOccupancyCut_;
    uint32_t maximumOccupancy_;

    bool applySumPtCut_;
    double maximumSumPt_;

    bool applyRelativeSumPtCut_;
    double maximumRelativeSumPt_;

    double customIsoCone_;

    edm::InputTag pvProducer_;

    Vertex currentPV_;
};

void PFRecoTauDiscriminationByIsolation::beginEvent(const edm::Event& event,
    const edm::EventSetup& eventSetup) {

  // NB: The use of the PV in this context is necessitated by its use in
  // applying quality cuts to the different objects in the isolation cone

  // get the PV for this event
  edm::Handle<VertexCollection> primaryVertices;
  event.getByLabel(pvProducer_, primaryVertices);

  // take the highest pt primary vertex in the event
  if( primaryVertices->size() ) {
    currentPV_ = *(primaryVertices->begin());
  } else {
    const double smearedPVsigmaY = 0.0015;
    const double smearedPVsigmaX = 0.0015;
    const double smearedPVsigmaZ = 0.005;
    Vertex::Error SimPVError;
    SimPVError(0,0) = smearedPVsigmaX*smearedPVsigmaX;
    SimPVError(1,1) = smearedPVsigmaY*smearedPVsigmaY;
    SimPVError(2,2) = smearedPVsigmaZ*smearedPVsigmaZ;
    Vertex::Point blankVertex(0, 0, 0);

    // note that the PFTau has its vertex set as the associated PV.  So if it
    // doesn't exist, a fake vertex has already been created (about 0, 0, 0) w/
    // the above width (gaussian)
    currentPV_ = Vertex(blankVertex, SimPVError,1,1,1);
  }
}

double PFRecoTauDiscriminationByIsolation::discriminate(const PFTauRef& pfTau) {
  // collect the objects we are working with (ie tracks, tracks+gammas, etc)
  std::vector<LeafCandidate> isoObjects;

  if (includeTracks_) {
    qualityCuts_.isolationChargedObjects(*pfTau, currentPV_, isoObjects);
  }

  if (includeGammas_) {
    qualityCuts_.isolationGammaObjects(*pfTau, isoObjects);
  }

  typedef reco::tau::cone::DeltaRFilter<LeafCandidate> DRFilter;

  // Check if we want a custom iso cone
  if (customIsoCone_ >= 0.) {
    DRFilter filter(pfTau->p4(), 0, customIsoCone_);
    // Remove all the objects not in our iso cone
    std::remove_if(isoObjects.begin(), isoObjects.end(), std::not1(filter));
  }

  bool failsOccupancyCut     = false;
  bool failsSumPtCut         = false;
  bool failsRelativeSumPtCut = false;

  //--- nObjects requirement
  failsOccupancyCut = ( isoObjects.size() > maximumOccupancy_ );

  //--- Sum PT requirement
  if( applySumPtCut_ || applyRelativeSumPtCut_ ) {
    reco::Particle::LorentzVector totalP4;
    BOOST_FOREACH(const LeafCandidate& isoObject, isoObjects) {
      totalP4 += isoObject.p4();
    }

    failsSumPtCut = (totalP4.pt() > maximumSumPt_);

    //--- Relative Sum PT requirement
    failsRelativeSumPtCut = (
        (pfTau->pt() > 0 ? totalP4.pt()/pfTau->pt() : 0 )
        > maximumRelativeSumPt_ );
  }

  bool fails = (applyOccupancyCut_ && failsOccupancyCut) ||
    (applySumPtCut_ && failsSumPtCut) ||
    (applyRelativeSumPtCut_ && failsRelativeSumPtCut) ;

  return (fails ? 0. : 1.);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolation);
