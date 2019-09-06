// -*- C++ -*-
//
// Package:    AlignmentMonitorAsAnalyzer
// Class:      AlignmentMonitorAsAnalyzer
//
/**\class AlignmentMonitorAsAnalyzer AlignmentMonitorAsAnalyzer.cc Dummy/AlignmentMonitorAsAnalyzer/src/AlignmentMonitorAsAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Sat Apr 26 12:36:13 CDT 2008
// $Id: AlignmentMonitorAsAnalyzer.cc,v 1.9 2012/07/13 09:18:40 yana Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

//
// class decleration
//

class AlignmentMonitorAsAnalyzer : public edm::EDAnalyzer {
public:
  explicit AlignmentMonitorAsAnalyzer(const edm::ParameterSet&);
  ~AlignmentMonitorAsAnalyzer() override = default;

  typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair;
  typedef std::vector<ConstTrajTrackPair> ConstTrajTrackPairCollection;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag m_tjTag;
  edm::ParameterSet m_aliParamStoreCfg;

  std::unique_ptr<AlignableTracker> m_alignableTracker;
  std::unique_ptr<AlignableMuon> m_alignableMuon;
  std::unique_ptr<AlignmentParameterStore> m_alignmentParameterStore;

  std::vector<std::unique_ptr<AlignmentMonitorBase>> m_monitors;

  bool m_firstEvent;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AlignmentMonitorAsAnalyzer::AlignmentMonitorAsAnalyzer(const edm::ParameterSet& iConfig)
    : m_tjTag(iConfig.getParameter<edm::InputTag>("tjTkAssociationMapTag")),
      m_aliParamStoreCfg(iConfig.getParameter<edm::ParameterSet>("ParameterStore")) {
  std::vector<std::string> monitors = iConfig.getUntrackedParameter<std::vector<std::string>>("monitors");

  for (auto const& mon : monitors) {
    m_monitors.emplace_back(
        AlignmentMonitorPluginFactory::get()->create(mon, iConfig.getUntrackedParameter<edm::ParameterSet>(mon)));
  }
}

//
// member functions
//

// ------------ method called to for each event  ------------
void AlignmentMonitorAsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  if (m_firstEvent) {
    GeometryAligner aligner;

    edm::ESTransientHandle<DDCompactView> cpv;
    iSetup.get<IdealGeometryRecord>().get(cpv);

    edm::ESHandle<GeometricDet> theGeometricDet;
    iSetup.get<IdealGeometryRecord>().get(theGeometricDet);
    edm::ESHandle<PTrackerParameters> ptp;
    iSetup.get<PTrackerParametersRcd>().get(ptp);
    TrackerGeomBuilderFromGeometricDet trackerBuilder;
    std::shared_ptr<TrackerGeometry> theTracker(trackerBuilder.build(&(*theGeometricDet), *ptp, tTopo));

    edm::ESHandle<MuonDDDConstants> mdc;
    iSetup.get<MuonNumberingRecord>().get(mdc);
    DTGeometryBuilderFromDDD DTGeometryBuilder;
    CSCGeometryBuilderFromDDD CSCGeometryBuilder;
    auto theMuonDT = std::make_shared<DTGeometry>();
    DTGeometryBuilder.build(*theMuonDT, &(*cpv), *mdc);
    auto theMuonCSC = std::make_shared<CSCGeometry>();
    CSCGeometryBuilder.build(*theMuonCSC, &(*cpv), *mdc);

    edm::ESHandle<Alignments> globalPositionRcd;
    iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);

    edm::ESHandle<Alignments> alignments;
    iSetup.get<TrackerAlignmentRcd>().get(alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    iSetup.get<TrackerAlignmentErrorExtendedRcd>().get(alignmentErrors);
    aligner.applyAlignments<TrackerGeometry>(&(*theTracker),
                                             &(*alignments),
                                             &(*alignmentErrors),
                                             align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Tracker)));

    edm::ESHandle<Alignments> dtAlignments;
    iSetup.get<DTAlignmentRcd>().get(dtAlignments);
    edm::ESHandle<AlignmentErrorsExtended> dtAlignmentErrorsExtended;
    iSetup.get<DTAlignmentErrorExtendedRcd>().get(dtAlignmentErrorsExtended);
    aligner.applyAlignments<DTGeometry>(&(*theMuonDT),
                                        &(*dtAlignments),
                                        &(*dtAlignmentErrorsExtended),
                                        align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));

    edm::ESHandle<Alignments> cscAlignments;
    iSetup.get<CSCAlignmentRcd>().get(cscAlignments);
    edm::ESHandle<AlignmentErrorsExtended> cscAlignmentErrorsExtended;
    iSetup.get<CSCAlignmentErrorExtendedRcd>().get(cscAlignmentErrorsExtended);
    aligner.applyAlignments<CSCGeometry>(&(*theMuonCSC),
                                         &(*cscAlignments),
                                         &(*cscAlignmentErrorsExtended),
                                         align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));

    // within an analyzer, modules can't expect to see any selected alignables!
    align::Alignables empty_alignables;

    m_alignableTracker = std::make_unique<AlignableTracker>(&(*theTracker), tTopo);
    m_alignableMuon = std::make_unique<AlignableMuon>(&(*theMuonDT), &(*theMuonCSC));
    m_alignmentParameterStore = std::make_unique<AlignmentParameterStore>(empty_alignables, m_aliParamStoreCfg);

    for (auto const& monitor : m_monitors) {
      monitor->beginOfJob(m_alignableTracker.get(), m_alignableMuon.get(), m_alignmentParameterStore.get());
    }
    for (auto const& monitor : m_monitors) {
      monitor->startingNewLoop();
    }

    m_firstEvent = false;
  }

  // Retrieve trajectories and tracks from the event
  edm::Handle<TrajTrackAssociationCollection> trajTracksMap;
  iEvent.getByLabel(m_tjTag, trajTracksMap);

  // Form pairs of trajectories and tracks
  ConstTrajTrackPairCollection trajTracks;
  for (const auto& iPair : *trajTracksMap) {
    trajTracks.push_back(ConstTrajTrackPair(&(*iPair.key), &(*iPair.val)));
  }

  // Run the monitors
  for (const auto& monitor : m_monitors) {
    monitor->duringLoop(iEvent, iSetup, trajTracks);
  }
}

// ------------ method called once each job just before starting event loop  ------------
void AlignmentMonitorAsAnalyzer::beginJob() { m_firstEvent = true; }

// ------------ method called once each job just after ending the event loop  ------------
void AlignmentMonitorAsAnalyzer::endJob() {
  for (auto const& monitor : m_monitors) {
    monitor->endOfLoop();
  }
  for (auto const& monitor : m_monitors) {
    monitor->endOfJob();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignmentMonitorAsAnalyzer);
