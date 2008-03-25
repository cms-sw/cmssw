// -*- C++ -*-
//
// Package:    StraightMuonCutProducer
// Class:      StraightMuonCutProducer
// 
/**\class StraightMuonCutProducer StraightMuonCutProducer.cc Alignment/CommonAlignmentProducer/plugins/StraightMuonCutProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Feb 20 10:56:46 CST 2008
// $Id: StraightMuonCutProducer.cc,v 1.1 2008/02/20 22:51:36 pivarski Exp $
//
//


// system include files
#include <memory>
#include <map>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

//
// class decleration
//

class StraightMuonCutProducer : public edm::EDProducer {
   public:
      explicit StraightMuonCutProducer(const edm::ParameterSet&);
      ~StraightMuonCutProducer();

   private:
      virtual void beginJob(const edm::EventSetup&);
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      
      void make_chambers();
      void write();

      // ---------- member data --------------------------------
      std::map<int, std::string> m_chamber;
      edm::InputTag m_muonInput;
      TrackTransformer *m_trackTransformer;
      std::string m_muonPropagator;

      unsigned int m_stage;

      std::string m_outputFileName;
      unsigned int m_bins;
      double m_low, m_high;
      unsigned int m_minHits;
      std::map<int, TH1F*> m_hist;
      TH1F *m_histmeans;
      TH1F *m_histstdevs;

      std::map<int, double> m_mean, m_stdev;
      double m_cut;
      TH1F *m_histchi2;
      TH1F *m_histlog10chi2;
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
StraightMuonCutProducer::StraightMuonCutProducer(const edm::ParameterSet& iConfig)
   : m_stage(iConfig.getParameter<unsigned int>("stage"))
{
   make_chambers();
   edm::Service<TFileService> tfile;

   if (m_stage == 1) {
      m_outputFileName = iConfig.getParameter<std::string>("outputFileName");
      m_bins = iConfig.getParameter<unsigned int>("bins");
      m_low = iConfig.getParameter<double>("low");
      m_high = iConfig.getParameter<double>("high");
      m_minHits = iConfig.getParameter<unsigned int>("minHits");

      for (std::map<int, std::string>::const_iterator chamber = m_chamber.begin();  chamber != m_chamber.end();  ++chamber) {
	 m_hist[chamber->first] = tfile->make<TH1F>(chamber->second.c_str(), chamber->second.c_str(), m_bins, m_low, m_high);
      }
      m_histmeans = tfile->make<TH1F>("means", "means", m_bins, m_low, m_high);
      m_histstdevs = tfile->make<TH1F>("stdevs", "stdevs", m_bins, 0., m_high);
   }
   else if (m_stage == 2) {
      m_cut = iConfig.getParameter<double>("cut");

      for (std::map<int, std::string>::const_iterator chamber = m_chamber.begin();  chamber != m_chamber.end();  ++chamber) {
	 m_mean[chamber->first] = iConfig.getParameter<double>(chamber->second + std::string("_mean"));
	 m_stdev[chamber->first] = iConfig.getParameter<double>(chamber->second + std::string("_stdev"));
      }
      
      m_histchi2 = tfile->make<TH1F>("chi2", "chi2", 100, 0, 10.);
      m_histlog10chi2 = tfile->make<TH1F>("log10chi2", "log10chi2", 100, -3., 7.);
   }
   else {
      throw cms::Exception("StraightPropagationMuon") << "stage must be either 1 or 2 (it's " << m_stage << ")" << std::endl;
   }

   m_muonInput = iConfig.getParameter<edm::InputTag>("muonInput");
   m_trackTransformer = new TrackTransformer(iConfig.getParameter<edm::ParameterSet>("TrackerTrackTransformer"));
   m_muonPropagator = iConfig.getParameter<std::string>("muonPropagator");

   produces<reco::MuonCollection>();
}


StraightMuonCutProducer::~StraightMuonCutProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void StraightMuonCutProducer::make_chambers() {
   char name[256];

   for (int wheel = -2;  wheel <= 2;  wheel++) {
      for (int station = 1;  station <= 4;  station++) {
	 int max = (station == 4 ? 14 : 12);
	 for (int sector = 1;  sector <= max;  sector++) {
	    if (wheel < 0) sprintf(name, "MBm%d_%d_%d", -wheel, station, sector);
	    else           sprintf(name, "MBp%d_%d_%d",  wheel, station, sector);
	    
	    m_chamber[DTChamberId(wheel, station, sector).rawId()] = std::string(name);
	 }
      }
   }

   for (int disk = -4;  disk <= 4;  disk++) {
      if (disk == 0) continue;

      int max = 2;
      if (abs(disk) == 1) max = 4;
      if (abs(disk) == 4) max = 1;
      for (int ring = 1;  ring <= max;  ring++) {
	 int cmax = 36;
	 if (abs(disk) > 1  &&  ring == 1) cmax = 18;
	 for (int chamber = 1;  chamber <= cmax;  chamber++) {
	    if (disk < 0) sprintf(name, "MEm%d_%d_%d", -disk, ring, chamber);
	    else          sprintf(name, "MEp%d_%d_%d",  disk, ring, chamber);
	    
	    m_chamber[CSCDetId((disk < 0 ? 2 : 1), abs(disk), ring, chamber, 0).rawId()] = std::string(name);
	 }
      }
   }
}

// ------------ method called to produce the data  ------------
void
StraightMuonCutProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   m_trackTransformer->setServices(iSetup);

   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(m_muonInput, muons);

   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get(m_muonPropagator, propagator);

   edm::ESHandle<DTGeometry> dtGeometry;
   iSetup.get<MuonGeometryRecord>().get(dtGeometry);

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   edm::ESHandle<MagneticField> magneticField;
   iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

   // only really used for stage 2
   std::auto_ptr<reco::MuonCollection> output(new reco::MuonCollection);
   double chi2 = 0.;
   double N = 0.;

   for (reco::MuonCollection::const_iterator muon = muons->begin();  muon != muons->end();  ++muon) {
      std::vector<Trajectory> trackerTrajectories = m_trackTransformer->transform(*muon->track());

      if (trackerTrajectories.size() == 1) {
	 const Trajectory trackerTrajectory = *(trackerTrajectories.begin());

	 TrajectoryStateOnSurface tsos = trackerTrajectory.lastMeasurement().forwardPredictedState();
	 TrajectoryStateOnSurface last_tsos = tsos;

	 for (trackingRecHit_iterator hit = muon->standAloneMuon()->recHitsBegin();  hit != muon->standAloneMuon()->recHitsEnd();  ++hit) {
	    DetId id = (*hit)->geographicalId();
	    
	    int chamberId = 0;
	    if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	       DTChamberId dtid(id.rawId());
	       tsos = propagator->propagate(last_tsos, dtGeometry->idToDet(id)->surface());
	       chamberId = dtid.rawId();
	    }
	    else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	       CSCDetId cscid(id.rawId());
	       cscid = CSCDetId(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 0);
	       tsos = propagator->propagate(last_tsos, cscGeometry->idToDet(id)->surface());
	       chamberId = cscid.rawId();
	    }

	    if (chamberId != 0  &&  tsos.isValid()) {
	       double residual = (*hit)->localPosition().x() - tsos.localPosition().x();
	       double reserr2 = (*hit)->localPositionError().xx() + tsos.localError().positionError().xx();
		  
	       if (m_stage == 1) {
		  m_hist[chamberId]->Fill(residual, 1./reserr2);
	       }
	       else if (m_stage == 2) {
		  if (m_stdev[chamberId] < 999.) {
		     chi2 += pow((residual - m_mean[chamberId])/m_stdev[chamberId], 2);
		     N += 1.;
		  }
	       }

	       last_tsos = tsos;
	    }
	 } // end loop over hits
      } // endif we have a good trajectory

      if (m_stage == 2  &&  N > 0.) {
	 double redchi2 = chi2/N;

	 m_histchi2->Fill(redchi2);
	 m_histlog10chi2->Fill(log10(redchi2));

	 if (chi2/N > m_cut) output->push_back(*muon);
      }
   } // end loop over muons

   // in stage 1, output an empty list, which should be ignored
   iEvent.put(output);
}

// ------------ method called once each job just before starting event loop  ------------
void 
StraightMuonCutProducer::beginJob(const edm::EventSetup&) {}

// ------------ method called once each job just after ending the event loop  ------------
void 
StraightMuonCutProducer::endJob() {
   if (m_stage == 1) {
      std::ofstream output(m_outputFileName.c_str());
      output << "block StraightMuonCutStage2Params = {" << std::endl;
      for (std::map<int, std::string>::const_iterator chamber = m_chamber.begin();  chamber != m_chamber.end();  ++chamber) {
	 if (m_hist[chamber->first]->GetEntries() > m_minHits) {
	    output << "    double " << chamber->second << "_mean = " << m_hist[chamber->first]->GetMean() << std::endl;
	    output << "    double " << chamber->second << "_stdev = " << m_hist[chamber->first]->GetRMS() << std::endl;
	    m_histmeans->Fill(m_hist[chamber->first]->GetMean());
	    m_histstdevs->Fill(m_hist[chamber->first]->GetRMS());
	 }
	 else {
	    output << "    double " << chamber->second << "_mean = 0." << std::endl;
	    output << "    double " << chamber->second << "_stdev = 1000." << std::endl;
	 }
      }
      output << "}" << std::endl;
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(StraightMuonCutProducer);
