#ifndef TrackingRegionsFromBeamSpotAndL2Tau_h
#define TrackingRegionsFromBeamSpotAndL2Tau_h

//
// Class:           TrackingRegionsFromBeamSpotAndL2Tau
//
// $Id:$


#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"


/** class TrackingRegionsFromBeamSpotAndL2Tau
 * plugin for creating eta-phi TrackingRegions in directions of L2 taus
 */
class  TrackingRegionsFromBeamSpotAndL2Tau : public TrackingRegionProducer
{
public:
    
  explicit TrackingRegionsFromBeamSpotAndL2Tau(const edm::ParameterSet& conf)
  {
    edm::LogInfo ("TrackingRegionsFromBeamSpotAndL2Tau") << "Enter the TrackingRegionsFromBeamSpotAndL2Tau";

    edm::ParameterSet regionPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

    m_ptMin            = regionPSet.getParameter<double>("ptMin");
    m_originRadius     = regionPSet.getParameter<double>("originRadius");
    m_originHalfLength = regionPSet.getParameter<double>("originHalfLength");
    m_deltaEta         = regionPSet.getParameter<double>("deltaEta");
    m_deltaPhi         = regionPSet.getParameter<double>("deltaPhi");
    m_jetSrc           = regionPSet.getParameter<edm::InputTag>("JetSrc");
    m_jetMinPt         = regionPSet.getParameter<double>("JetMinPt");
    m_jetMaxEta        = regionPSet.getParameter<double>("JetMaxEta");
    m_jetMaxN          = regionPSet.getParameter<int>("JetMaxN");
    m_beamSpotTag      = regionPSet.getParameter<edm::InputTag>("beamSpot");
    m_precise          = regionPSet.getParameter<bool>("precise");

    if (regionPSet.exists("searchOpt")) m_searchOpt = regionPSet.getParameter<bool>("searchOpt");
    else                                m_searchOpt = false;

    m_measurementTrackerName ="";
    m_whereToUseMeasurementTracker=0;
    if (regionPSet.exists("measurementTrackerName"))
    {
      m_measurementTrackerName = regionPSet.getParameter<std::string>("measurementTrackerName");
      if (regionPSet.exists("whereToUseMeasurementTracker"))
        m_whereToUseMeasurementTracker = regionPSet.getParameter<double>("whereToUseMeasurementTracker");
    }
  }
  
  virtual ~TrackingRegionsFromBeamSpotAndL2Tau() {}
    

  virtual std::vector<TrackingRegion* > regions(const edm::Event& e, const edm::EventSetup& es) const
  {
    std::vector<TrackingRegion* > result;

    // use beam spot to pick up the origin
    edm::Handle<reco::BeamSpot> bsHandle;
    e.getByLabel( m_beamSpotTag, bsHandle);
    if(!bsHandle.isValid()) return result;
    const reco::BeamSpot & bs = *bsHandle;
    GlobalPoint origin(bs.x0(), bs.y0(), bs.z0());

    // get L2 object refs by the label
    edm::Handle<trigger::TriggerFilterObjectWithRefs> coll_l2;
    e.getByLabel(m_jetSrc, coll_l2);

    // get hold of L2 tau jets
    std::vector<reco::CaloJetRef> coll_l2_tau_jets;
    coll_l2->getObjects(trigger::TriggerTau, coll_l2_tau_jets);
    const size_t n_jets(coll_l2_tau_jets.size());
    
    // create maximum JetMaxN tracking regions in directions of 
    // highest pt jets that are above threshold and are within allowed eta
    // (we expect that jet collection was sorted in decreasing pt order)
    int n_regions = 0;
    for (unsigned int i =0; i < n_jets && n_regions < m_jetMaxN; ++i)
    {
      reco::CaloJetRef jet = coll_l2_tau_jets[i];
      if ( jet->pt() < m_jetMinPt || std::abs(jet->eta()) > m_jetMaxEta ) continue;
      
      GlobalVector direction(jet->momentum().x(), jet->momentum().y(), jet->momentum().z());
      RectangularEtaPhiTrackingRegion* etaphiRegion = 
        new RectangularEtaPhiTrackingRegion(
          direction,
          origin,
          m_ptMin,
          m_originRadius,
          m_originHalfLength,
          m_deltaEta,
          m_deltaPhi,
          m_whereToUseMeasurementTracker,
          m_precise,
          m_measurementTrackerName,
          m_searchOpt
        );
      result.push_back(etaphiRegion);
      ++n_regions;
    }
    //std::cout<<"nregions = "<<n_regions<<std::endl;
    return result;
  }
  
private:

  float m_ptMin;
  float m_originRadius;
  float m_originHalfLength;
  float m_deltaEta;
  float m_deltaPhi;
  edm::InputTag m_jetSrc;
  float m_jetMinPt;
  float m_jetMaxEta;
  int   m_jetMaxN;
  std::string m_measurementTrackerName;
  float m_whereToUseMeasurementTracker;
  bool m_searchOpt;
  edm::InputTag m_beamSpotTag;
  bool m_precise;
};

#endif
