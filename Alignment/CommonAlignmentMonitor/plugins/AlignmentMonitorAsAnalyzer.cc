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
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
//
// class decleration
//

class AlignmentMonitorAsAnalyzer : public edm::EDAnalyzer {
   public:
      explicit AlignmentMonitorAsAnalyzer(const edm::ParameterSet&);
      ~AlignmentMonitorAsAnalyzer();

      typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair; 
      typedef std::vector<ConstTrajTrackPair> ConstTrajTrackPairCollection;

   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      edm::InputTag m_tjTag;
      edm::ParameterSet m_aliParamStoreCfg;
  const edm::ParameterSet m_pSet;

      AlignableTracker *m_alignableTracker;
      AlignableMuon *m_alignableMuon;
      AlignmentParameterStore *m_alignmentParameterStore;

      std::vector<AlignmentMonitorBase*> m_monitors;
      const edm::EventSetup *m_lastSetup;

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
   : m_tjTag(iConfig.getParameter<edm::InputTag>("tjTkAssociationMapTag"))
     , m_aliParamStoreCfg(iConfig.getParameter<edm::ParameterSet>("ParameterStore")),
     m_pSet(iConfig)
   , m_alignableTracker(NULL)
   , m_alignableMuon(NULL)
   , m_alignmentParameterStore(NULL)
{
   std::vector<std::string> monitors = iConfig.getUntrackedParameter<std::vector<std::string> >( "monitors" );

   for (std::vector<std::string>::const_iterator miter = monitors.begin();  miter != monitors.end();  ++miter) {
      AlignmentMonitorBase* newMonitor = AlignmentMonitorPluginFactory::get()->create(*miter, iConfig.getUntrackedParameter<edm::ParameterSet>(*miter));

      if (!newMonitor) throw cms::Exception("BadConfig") << "Couldn't find monitor named " << *miter;

      m_monitors.push_back(newMonitor);
   }
}


AlignmentMonitorAsAnalyzer::~AlignmentMonitorAsAnalyzer()
{
   delete m_alignableTracker;
   delete m_alignableMuon;
   delete m_alignmentParameterStore;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
AlignmentMonitorAsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
   const TrackerTopology* const tTopo = tTopoHandle.product();

   if (m_firstEvent) {
      GeometryAligner aligner;
    
      edm::ESTransientHandle<DDCompactView> cpv;
      iSetup.get<IdealGeometryRecord>().get( cpv );
      
      edm::ESHandle<GeometricDet> theGeometricDet;
      iSetup.get<IdealGeometryRecord>().get( theGeometricDet );
      TrackerGeomBuilderFromGeometricDet trackerBuilder;
      boost::shared_ptr<TrackerGeometry> theTracker(trackerBuilder.build(&(*theGeometricDet), m_pSet ));
      
      edm::ESHandle<MuonDDDConstants> mdc;
      iSetup.get<MuonNumberingRecord>().get(mdc);
      DTGeometryBuilderFromDDD DTGeometryBuilder;
      CSCGeometryBuilderFromDDD CSCGeometryBuilder;
      boost::shared_ptr<DTGeometry> theMuonDT(new DTGeometry);
      DTGeometryBuilder.build(theMuonDT, &(*cpv), *mdc);
      boost::shared_ptr<CSCGeometry> theMuonCSC(new CSCGeometry);
      CSCGeometryBuilder.build(theMuonCSC, &(*cpv), *mdc);
      
      edm::ESHandle<Alignments> globalPositionRcd;
      iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);
      
      edm::ESHandle<Alignments> alignments;
      iSetup.get<TrackerAlignmentRcd>().get( alignments );
      edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
      iSetup.get<TrackerAlignmentErrorExtendedRcd>().get( alignmentErrors );
      aligner.applyAlignments<TrackerGeometry>( &(*theTracker), &(*alignments), &(*alignmentErrors),
						align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Tracker)) );
      
      edm::ESHandle<Alignments> dtAlignments;
      iSetup.get<DTAlignmentRcd>().get( dtAlignments );
      edm::ESHandle<AlignmentErrorsExtended> dtAlignmentErrorsExtended;
      iSetup.get<DTAlignmentErrorExtendedRcd>().get( dtAlignmentErrorsExtended );
      aligner.applyAlignments<DTGeometry>( &(*theMuonDT), &(*dtAlignments), &(*dtAlignmentErrorsExtended),
					   align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)) );
      
      edm::ESHandle<Alignments> cscAlignments;
      iSetup.get<CSCAlignmentRcd>().get( cscAlignments );
      edm::ESHandle<AlignmentErrorsExtended> cscAlignmentErrorsExtended;
      iSetup.get<CSCAlignmentErrorExtendedRcd>().get( cscAlignmentErrorsExtended );
      aligner.applyAlignments<CSCGeometry>( &(*theMuonCSC), &(*cscAlignments), &(*cscAlignmentErrorsExtended),
					    align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)) );
      
      // within an analyzer, modules can't expect to see any selected alignables!
      std::vector<Alignable*> empty_alignables;
      
      m_alignableTracker = new AlignableTracker( &(*theTracker), tTopo );
      m_alignableMuon = new AlignableMuon( &(*theMuonDT), &(*theMuonCSC) );
      m_alignmentParameterStore = new AlignmentParameterStore(empty_alignables, m_aliParamStoreCfg);
      
      for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = m_monitors.begin();  monitor != m_monitors.end();  ++monitor) {
	(*monitor)->beginOfJob(m_alignableTracker, m_alignableMuon, m_alignmentParameterStore);
      }
      for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = m_monitors.begin();  monitor != m_monitors.end();  ++monitor) {
	(*monitor)->startingNewLoop();
      }
      
      m_firstEvent = false;
   }

   // Retrieve trajectories and tracks from the event
   edm::Handle<TrajTrackAssociationCollection> trajTracksMap;
   iEvent.getByLabel(m_tjTag, trajTracksMap);
   
   // Form pairs of trajectories and tracks
   ConstTrajTrackPairCollection trajTracks;
   for (TrajTrackAssociationCollection::const_iterator iPair = trajTracksMap->begin();  iPair != trajTracksMap->end();  ++iPair) {
      trajTracks.push_back(ConstTrajTrackPair(&(*(*iPair).key), &(*(*iPair).val)));
   }
   
   // Run the monitors
   for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = m_monitors.begin();  monitor != m_monitors.end();  ++monitor) {
      (*monitor)->duringLoop(iEvent, iSetup, trajTracks);
   }

   // Keep this for endOfLoop (why does endOfLoop want iSetup???)
   m_lastSetup = &iSetup;
}


// ------------ method called once each job just before starting event loop  ------------
void 
AlignmentMonitorAsAnalyzer::beginJob()
{
   m_firstEvent = true;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
AlignmentMonitorAsAnalyzer::endJob() 
{
   for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = m_monitors.begin();  monitor != m_monitors.end();  ++monitor) {
      (*monitor)->endOfLoop(*m_lastSetup);
   }
   for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = m_monitors.begin();  monitor != m_monitors.end();  ++monitor) {
      (*monitor)->endOfJob();
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignmentMonitorAsAnalyzer);
