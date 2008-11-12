// -*- C++ -*-
//
// Package:    MuonHIPAlignmentAlgorithm
// Class:      MuonHIPAlignmentAlgorithm
// 
/**\class MuonHIPAlignmentAlgorithm MuonHIPAlignmentAlgorithm.cc Alignment/MuonHIPAlignmentAlgorithm/interface/MuonHIPAlignmentAlgorithm.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski,,,
//         Created:  Wed Nov  5 21:47:33 CET 2008
// $Id$
//
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/HIPAlignmentAlgorithm/interface/MuonHIPAlignmentAlgorithm.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"  
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"  
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"  
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"  
#include "Alignment/MuonAlignment/interface/AlignableDTSuperLayer.h"
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCRing.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include <sstream>

MuonHIPAlignmentAlgorithm::MuonHIPAlignmentAlgorithm(const edm::ParameterSet& iConfig)
   : AlignmentAlgorithmBase(iConfig)
   , m_minTrackerHits(iConfig.getParameter<int>("minTrackerHits"))
   , m_maxRedChi2(iConfig.getParameter<double>("maxRedChi2"))
   , m_minStations(iConfig.getParameter<int>("minStations"))
   , m_minHitsPerDT(iConfig.getParameter<int>("minHitsPerDT"))
   , m_minHitsPerDT4(iConfig.getParameter<int>("minHitsPerDT4"))
   , m_minHitsPerCSC(iConfig.getParameter<int>("minHitsPerCSC"))
   , m_maxResidualDT13(iConfig.getParameter<double>("maxResidualDT13"))
   , m_maxResidualDT2(iConfig.getParameter<double>("maxResidualDT2"))
   , m_maxResidualCSC(iConfig.getParameter<double>("maxResidualCSC"))
   , m_ignoreCSCRings(iConfig.getParameter<std::vector<int> >("ignoreCSCRings"))
   , m_minTracksPerAlignable(iConfig.getParameter<int>("minTracksPerAlignable"))
   , m_useHitWeightsInDTAlignment(iConfig.getParameter<bool>("useHitWeightsInDTAlignment"))
   , m_useHitWeightsInCSCAlignment(iConfig.getParameter<bool>("useHitWeightsInCSCAlignment"))
   , m_useOneDTSuperLayerPerEntry(iConfig.getParameter<bool>("useOneDTSuperLayerPerEntry"))
   , m_useOneCSCChamberPerEntry(iConfig.getParameter<bool>("useOneCSCChamberPerEntry"))
   , m_fitRangeDTrphi(iConfig.getParameter<double>("fitRangeDTrphi"))
   , m_fitRangeDTz(iConfig.getParameter<double>("fitRangeDTz"))
   , m_fitRangeCSCrphi(iConfig.getParameter<double>("fitRangeCSCrphi"))
   , m_fitRangeCSCz(iConfig.getParameter<double>("fitRangeCSCz"))
   
   , m_align(iConfig.getParameter<bool>("align"))
   , m_collector(iConfig.getParameter<std::vector<std::string> >("collector"))
   , m_collectorDirectory(iConfig.getParameter<std::string>("collectorDirectory"))
{}

MuonHIPAlignmentAlgorithm::~MuonHIPAlignmentAlgorithm() {}

void MuonHIPAlignmentAlgorithm::initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore) {
   m_alignmentParameterStore = alignmentParameterStore;
   m_alignables = m_alignmentParameterStore->alignables();

   if (alignableMuon == NULL) {
     throw cms::Exception("MuonHIPAlignmentAlgorithm") << "doMuon must be set to True" << std::endl;
   }

   if (alignableTracker == NULL) m_alignableNavigator = new AlignableNavigator(alignableMuon);
   else m_alignableNavigator = new AlignableNavigator(alignableTracker, alignableMuon);

   // set the APE of all muon chambers to 1000.
   std::vector<Alignable*> all_DT_chambers = alignableMuon->DTChambers();
   std::vector<Alignable*> all_CSC_chambers = alignableMuon->CSCChambers();
   m_alignmentParameterStore->setAlignmentPositionError(all_DT_chambers, 1000., 0.);
   m_alignmentParameterStore->setAlignmentPositionError(all_CSC_chambers, 1000., 0.);

   edm::Service<TFileService> tfileService;
   TFileDirectory dir = tfileService->mkdir("MuonHIPAlignmentAlgorithm");

   m_hist_qoverpt = dir.make<TH1F>("qoverpt", "Charge over pT (1/GeV)", 100, -0.2, 0.2);
   m_hist_qoverpt_vs_eta = dir.make<TProfile>("qoverpt_vs_eta", "Charge over pT (1/GeV) versus eta", 100, -2.4, 2.4);
   m_hist_plusoverpt_vs_eta = dir.make<TProfile>("qoverpt_vs_eta_plus", "Positive particle 1/pT (1/GeV) versus eta", 100, -2.4, 2.4);
   m_hist_minusoverpt_vs_eta = dir.make<TProfile>("qoverpt_vs_eta_minus", "Negative particle 1/pT (1/GeV) versus eta", 100, -2.4, 2.4);
   m_hist_redChi2 = dir.make<TH1F>("redchi2", "Reduced chi^2 for tracker hits", 100, 0., 10.);

   m_all_histograms.push_back(m_hist_qoverpt);
   m_all_histograms.push_back(m_hist_qoverpt_vs_eta);
   m_all_histograms.push_back(m_hist_plusoverpt_vs_eta);
   m_all_histograms.push_back(m_hist_minusoverpt_vs_eta);
   m_all_histograms.push_back(m_hist_redChi2);

   m_hist_qoverpt_vs_eta->SetAxisRange(-0.2, 0.2, "Y");
   m_hist_plusoverpt_vs_eta->SetAxisRange(0., 0.4, "Y");
   m_hist_minusoverpt_vs_eta->SetAxisRange(0., 0.4, "Y");

   for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
      int rawId = (*ali)->id();

      std::stringstream name, title;
      name << rawId;
      if (dynamic_cast<AlignableDetUnit*>(*ali) != NULL  &&  (*ali)->geomDetId().det() == DetId::Muon  &&  (*ali)->geomDetId().subdetId() == MuonSubdetId::DT) {
	 m_nearlyGlobalCoords[*ali] = false;
	 DTLayerId id(rawId);
	 title << id;
      }
      else if (dynamic_cast<AlignableDTSuperLayer*>(*ali) != NULL) {
	 m_nearlyGlobalCoords[*ali] = false;
	 DTSuperLayerId id(rawId);
	 title << id;
      }
      else if (dynamic_cast<AlignableDTChamber*>(*ali) != NULL) {
	 m_nearlyGlobalCoords[*ali] = false;
	 DTChamberId id(rawId);
	 title << id;
      }
      else if (dynamic_cast<AlignableDTStation*>(*ali) != NULL) {
	 m_nearlyGlobalCoords[*ali] = true;
	 DTChamberId id(rawId);
	 title << "Wh:" << id.wheel() << " St:" << id.station();
      }
      else if (dynamic_cast<AlignableDTWheel*>(*ali) != NULL) {
	 m_nearlyGlobalCoords[*ali] = true;
	 DTChamberId id(rawId);
	 title << "Wh:" << id.wheel();
      }
      else if (dynamic_cast<AlignableDetUnit*>(*ali) != NULL  &&  (*ali)->geomDetId().det() == DetId::Muon  &&  (*ali)->geomDetId().subdetId() == MuonSubdetId::CSC) {
	 m_nearlyGlobalCoords[*ali] = false;
	 CSCDetId id(rawId);
	 title << id;
      }
      else if (dynamic_cast<AlignableCSCChamber*>(*ali) != NULL) {
	 m_nearlyGlobalCoords[*ali] = false;
	 CSCDetId id(rawId);
	 CSCDetId id2(id.endcap(), id.station(), id.ring(), id.chamber());
	 title << id;
      }
      else if (dynamic_cast<AlignableCSCRing*>(*ali) != NULL) {
	 m_nearlyGlobalCoords[*ali] = true;
	 CSCDetId id(rawId);
	 title << "E:" << id.endcap() << " S:" << id.station() << " R:" << id.ring();
      }
      else if (dynamic_cast<AlignableCSCStation*>(*ali) != NULL) {
	 m_nearlyGlobalCoords[*ali] = true;
	 CSCDetId id(rawId);
	 title << "E:" << id.endcap() << " S:" << id.station();
      }
      else {
	 m_nearlyGlobalCoords[*ali] = false;
	 title << rawId;
      }

      m_hist_xresidual[*ali] = dir.make<TH1F>((std::string("xresidual_") + name.str()).c_str(), (title.str() + std::string(" xresidual (mm)")).c_str(), 100, -200., 200.);
      m_hist_yresidual[*ali] = dir.make<TH1F>((std::string("yresidual_") + name.str()).c_str(), (title.str() + std::string(" yresidual (mm)")).c_str(), 100, -200., 200.);
      m_hist_xresidual10GeV[*ali] = dir.make<TH1F>((std::string("xresidual10GeV_") + name.str()).c_str(), (title.str() + std::string(" xresidual (mm) with 10 GeV cut")).c_str(), 100, -200., 200.);
      m_hist_yresidual10GeV[*ali] = dir.make<TH1F>((std::string("yresidual10GeV_") + name.str()).c_str(), (title.str() + std::string(" yresidual (mm) with 10 GeV cut")).c_str(), 100, -200., 200.);
      m_hist_xresidual20GeV[*ali] = dir.make<TH1F>((std::string("xresidual20GeV_") + name.str()).c_str(), (title.str() + std::string(" xresidual (mm) with 20 GeV cut")).c_str(), 100, -200., 200.);
      m_hist_yresidual20GeV[*ali] = dir.make<TH1F>((std::string("yresidual20GeV_") + name.str()).c_str(), (title.str() + std::string(" yresidual (mm) with 20 GeV cut")).c_str(), 100, -200., 200.);

      m_all_histograms.push_back(m_hist_xresidual[*ali]);
      m_all_histograms.push_back(m_hist_yresidual[*ali]);
      m_all_histograms.push_back(m_hist_xresidual10GeV[*ali]);
      m_all_histograms.push_back(m_hist_yresidual10GeV[*ali]);
      m_all_histograms.push_back(m_hist_xresidual20GeV[*ali]);
      m_all_histograms.push_back(m_hist_yresidual20GeV[*ali]);

      std::vector<bool> selector = (*ali)->alignmentParameters()->selector();
      int par = 0;
      for (std::vector<bool>::const_iterator sel = selector.begin();  sel != selector.end();  ++sel, ++par) {
	 if (*sel) {
	    std::string parname, units;
	    double range, profrange;
	    if (par == 0) {parname = "x";  units = " (mm)";  range = 100.;  profrange = 50.;}
	    else if (par == 1) {parname = "y";  units = " (mm)";  range = 300.;  profrange = 100.;}
	    else if (par == 2) {parname = "z";  units = " (mm)";  range = 200.;  profrange = 50.;}
	    else if (par == 3) {parname = "phix";  units = " (mrad)";  range = 100.;  profrange = 20.;}
	    else if (par == 4) {parname = "phiy";  units = " (mrad)";  range = 200.;  profrange = 100.;}
	    else if (par == 5) {parname = "phiz";  units = " (mrad)";  range = 30.;  profrange = 10.;}
	    else assert(false);

	    std::pair<Alignable*,int> index(*ali, par);

	    m_hist_parameter[index] = dir.make<TH1F>(
	       (std::string("parameter_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction") + units).c_str(), 100, -range, range);
	    m_hist_parameter10GeV[index] = dir.make<TH1F>(
	       (std::string("parameter10GeV_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction") + units + std::string(" with a 10 GeV cut")).c_str(), 100, -range, range);
	    m_hist_parameter20GeV[index] = dir.make<TH1F>(
	       (std::string("parameter20GeV_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction") + units + std::string(" with a 20 GeV cut")).c_str(), 100, -range, range);

	    m_hist_oparameter[index] = dir.make<TH1F>(
	       (std::string("oparameter_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction") + units + std::string(", one entry per ") + (DetId(rawId).subdetId() == MuonSubdetId::DT ? "superlayer" : "chamber")).c_str(), 100, -range, range);
	    m_hist_oparameter10GeV[index] = dir.make<TH1F>(
	       (std::string("oparameter10GeV_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction") + units + std::string(" with a 10 GeV cut, one entry per ") + (DetId(rawId).subdetId() == MuonSubdetId::DT ? "superlayer" : "chamber")).c_str(), 100, -range, range);
	    m_hist_oparameter20GeV[index] = dir.make<TH1F>(
	       (std::string("oparameter20GeV_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction") + units + std::string(" with a 20 GeV cut, one entry per ") + (DetId(rawId).subdetId() == MuonSubdetId::DT ? "superlayer" : "chamber")).c_str(), 100, -range, range);

	    m_hist_prof[index] = dir.make<TProfile>((std::string("prof_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction") + units).c_str(), 40, -0.2, 0.2);
	    m_hist_oprof[index] = dir.make<TProfile>((std::string("oprof_") + parname + std::string("_") + name.str()).c_str(), (title.str() + std::string(" ") + parname + std::string(" correction, one entry per ") + (DetId(rawId).subdetId() == MuonSubdetId::DT ? "superlayer" : "chamber") + units).c_str(), 40, -0.2, 0.2);

	    m_hist_prof[index]->SetAxisRange(-profrange, profrange, "Y");
	    m_hist_oprof[index]->SetAxisRange(-profrange, profrange, "Y");
	    
	    m_all_histograms.push_back(m_hist_parameter[index]);
	    m_all_histograms.push_back(m_hist_parameter10GeV[index]);
	    m_all_histograms.push_back(m_hist_parameter20GeV[index]);
	    m_all_histograms.push_back(m_hist_oparameter[index]);
	    m_all_histograms.push_back(m_hist_oparameter10GeV[index]);
	    m_all_histograms.push_back(m_hist_oparameter20GeV[index]);
	    m_all_histograms.push_back(m_hist_prof[index]);
	    m_all_histograms.push_back(m_hist_oprof[index]);

	 } // end if this is a parameter we're plotting and aligning
      } // end loop over parameters

   } // end loop over alignables
}

void MuonHIPAlignmentAlgorithm::startNewLoop() {}

void MuonHIPAlignmentAlgorithm::run(const edm::EventSetup& iSetup, const ConstTrajTrackPairCollection& trajtracks) {
   if (m_collector.size() != 0) return;

   edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
   iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

   for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack) {
      const Trajectory* traj = (*trajtrack).first;
      const reco::Track* track = (*trajtrack).second;

      std::vector<const TransientTrackingRecHit*> hitvec;
      std::vector<TrajectoryStateOnSurface> tsosvec;

//      GlobalPoint innermost(10000., 10000., 10000.);
      double qoverpt = track->charge() / track->pt();
      double eta = track->eta();

      int trackerHits = 0;
      std::map<int,int> stationsSeen;
      double chi2 = 0.;
      double ndof = 0.;

      std::vector<TrajectoryMeasurement> measurements = traj->measurements();
      for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
	 TrajectoryMeasurement meas = *im;
	 const TransientTrackingRecHit *hit = &(*meas.recHit());
	 DetId id = hit->geographicalId();

	 if (hit->isValid()) {
	    TrajectoryStateOnSurface tsos = m_tsoscomb(meas.forwardPredictedState(), meas.backwardPredictedState());
	    if (tsos.isValid()) {
	       if (m_alignableNavigator->detAndSubdetInMap(id)) {
		  hitvec.push_back(hit);
		  tsosvec.push_back(tsos);
	       }

// 	       GlobalPoint where = globalGeometry->idToDet(hit->geographicalId())->toGlobal(hit->localPosition());
// 	       if (where.mag() < innermost.mag()) {
// 		  qoverpt = tsos.charge() / tsos.globalMomentum().perp();
// 		  eta = tsos.globalMomentum().eta();
// 		  innermost = where;
// 	       }

	       if (id.det() == DetId::Tracker) {
		  trackerHits++;
		  
		  AlgebraicVector residual(2);
		  residual[0] = tsos.localPosition().x() - hit->localPosition().x();
		  residual[1] = tsos.localPosition().y() - hit->localPosition().y();

		  AlgebraicSymMatrix hitCov(2);
		  hitCov[0][0] = hit->localPositionError().xx();
		  hitCov[0][1] = hit->localPositionError().xy();
		  hitCov[1][1] = hit->localPositionError().yy();

		  AlgebraicSymMatrix tsosCov(2);
		  tsosCov[0][0] = tsos.localError().positionError().xx();
		  tsosCov[0][1] = tsos.localError().positionError().xy();
		  tsosCov[1][1] = tsos.localError().positionError().yy();

		  AlgebraicSymMatrix cov = hitCov + tsosCov;
		  int ierr;
		  cov.invert(ierr);
		  if (ierr != 0) {
		     edm::LogError("MuonHIPAlignmentAlgorithm") << "Matrix inversion failed!" << std::endl;
		     return;
		  }

		  if (hit->dimension() == 1) {
		     chi2 += pow(residual[0], 2.) * cov[0][0];
		     ndof += 1.;
		  }
		  else {
		     chi2 += pow(residual[0], 2.) * cov[0][0] + 2.*residual[0]*residual[1] * cov[0][1] + pow(residual[1], 2.) * cov[1][1];
		     ndof += 2.;
		  }
	       }

	       if (id.det() == DetId::Muon) {
		  int index = -1;
		  if (id.subdetId() == MuonSubdetId::DT) {
		     DTChamberId dtid(id.rawId());
		     index = dtid.station() + 10*(dtid.wheel() + 3);  // always-positive, index % 10 == DT station
		  }
		  else if (id.subdetId() == MuonSubdetId::CSC) {
		     CSCDetId cscid(id.rawId());
		     index = cscid.ring() * 100 + cscid.station() * 1000 + cscid.endcap() * 10000;
		  }

		  if (index > 0) {
		     if (stationsSeen.find(index) == stationsSeen.end()) {
			stationsSeen[index] = 0;
		     }
		     stationsSeen[index]++;
		  }
	       } // end if this is a muon hit

	    } // end if good TSOS
	 } // end if good hit
      } // end loop over hits/TSOSes
      ndof -= 5.;

      double redChi2 = -1.;
      if (ndof > 0.) redChi2 = chi2 / ndof;
      m_hist_redChi2->Fill(redChi2);

      int numStations = 0;
      for (std::map<int,int>::const_iterator iterStations = stationsSeen.begin();  iterStations != stationsSeen.end();  ++iterStations) {
	 int index = iterStations->first;
	 int numHits = iterStations->second;

	 if (index % 10 == 4) {
	    if (m_minHitsPerDT4 < 0  ||  numHits >= m_minHitsPerDT4) numStations++;
	 }
	 else if (index > 100) {
	    if (m_minHitsPerCSC < 0  ||  numHits >= m_minHitsPerCSC) numStations++;
	 }
	 else {
	    if (m_minHitsPerDT < 0  ||  numHits >= m_minHitsPerDT) numStations++;
	 }
      }
      
      m_hist_qoverpt->Fill(qoverpt);
      m_hist_qoverpt_vs_eta->Fill(eta, qoverpt);
      if (qoverpt > 0.) m_hist_plusoverpt_vs_eta->Fill(eta, fabs(qoverpt));
      else m_hist_minusoverpt_vs_eta->Fill(eta, fabs(qoverpt));

      std::vector<AlignableDetOrUnitPtr> alidetvec = m_alignableNavigator->alignablesFromHits(hitvec);
      CompositeAlignmentParameters aap = m_alignmentParameterStore->selectParameters(alidetvec);

      std::map<DetId,Alignable*> chamberalignable;
      std::map<DetId,AlgebraicSymMatrix> chamberjtvj;
      std::map<DetId,AlgebraicVector> chamberjtve;
      std::map<DetId,bool> chamber_goodZ, chamber_goodXY;

      std::vector<const TransientTrackingRecHit*>::const_iterator ihit = hitvec.begin();
      std::vector<TrajectoryStateOnSurface>::const_iterator itsos = tsosvec.begin();
      for (; ihit != hitvec.end()  &&  itsos != tsosvec.end();  ++ihit, ++itsos) {
	 DetId id = (*ihit)->geographicalId();

	 const GeomDet *det = (*ihit)->det();
	 AlignableDetOrUnitPtr alidet = m_alignableNavigator->alignableFromGeomDet(det);
	 
	 Alignable *ali = aap.alignableFromAlignableDet(alidet);
	 if (ali != NULL) {
	    AlgebraicVector hitPos(2);
	    hitPos[0] = (*ihit)->localPosition().x();
	    hitPos[1] = (*ihit)->localPosition().y();

	    AlgebraicSymMatrix hitCov(2);
	    hitCov[0][0] = (*ihit)->localPositionError().xx();
	    hitCov[0][1] = (*ihit)->localPositionError().xy();
	    hitCov[1][1] = (*ihit)->localPositionError().yy();

	    AlgebraicVector tsosPos(2);
	    tsosPos[0] = itsos->localPosition().x();
	    tsosPos[1] = itsos->localPosition().y();
	    
	    AlgebraicSymMatrix tsosCov(2);
	    tsosCov[0][0] = itsos->localError().positionError().xx();
	    tsosCov[0][1] = itsos->localError().positionError().xy();
	    tsosCov[1][1] = itsos->localError().positionError().yy();

	    AlgebraicVector residual(2);
	    residual = tsosPos - hitPos;

	    AlgebraicSymMatrix cov = hitCov + tsosCov;
	    int ierr;
	    cov.invert(ierr);
	    if (ierr != 0) {
	       edm::LogError("MuonHIPAlignmentAlgorithm") << "Matrix inversion failed!" << std::endl;
	       return;
	    }

	    if ((*ihit)->dimension() == 1) {
	       cov[0][1] = 0.;
	       cov[1][1] = 0.;
	    }

	    AlignmentParameters *params = ali->alignmentParameters();
	    AlgebraicMatrix derivs = params->selectedDerivatives(*itsos, alidet);

	    int npar = derivs.num_row();
	    AlgebraicSymMatrix thisjtvj(npar);
	    AlgebraicVector thisjtve(npar);
	    thisjtvj = cov.similarity(derivs);
	    thisjtve = derivs * cov * residual;

	    AlgebraicVector p(npar);
	    AlgebraicVector e(npar);
	    for (int i = 0;  i < npar;  i++) {
	       if (thisjtvj[i][i] == 0.  ||  1./thisjtvj[i][i] < 0.) {
		  edm::LogError("MuonHIPAlignmentAlgorithm") << "Parameter weight is " << thisjtvj[i][i] << std::endl;
		  return;
	       }

	       p[i] = -thisjtve[i] / thisjtvj[i][i];    // here is where the important minus sign is applied
	       e[i] = sqrt(1./thisjtvj[i][i]);
	    }

	    // You've made it far enough to know that you won't return an error; now fill histograms
	    if ((m_minTrackerHits < 0.  ||  trackerHits >= m_minTrackerHits)  &&
		(m_maxRedChi2 < 0.  ||  (redChi2 >= 0.  &&  redChi2 < m_maxRedChi2))  &&
		(m_minStations < 0.  ||  numStations >= m_minStations)  &&
		true) {

	       double xresidual = residual[0] * 10.;
	       double yresidual = residual[1] * 10.;

	       bool goodResidual = true;
	       bool goodZ = true;
	       bool goodXY = true;
	       DetId chamberId;  // in the case of DT, this "chamberId" is actually a superlayer id
	       if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
		  DTSuperLayerId dtid(id.rawId());
		  if (dtid.superlayer() == 2) {
		     m_hist_yresidual[ali]->Fill(xresidual);
		     if (fabs(1./qoverpt) > 10.) m_hist_yresidual10GeV[ali]->Fill(xresidual);
		     if (fabs(1./qoverpt) > 20.) m_hist_yresidual20GeV[ali]->Fill(xresidual);

		     if (m_maxResidualDT2 >= 0.  &&  fabs(xresidual / 10.) > m_maxResidualDT2) goodResidual = false;
		     if (m_nearlyGlobalCoords[ali]) goodXY = false;
		  }
		  else {
		     m_hist_xresidual[ali]->Fill(xresidual);
		     if (fabs(1./qoverpt) > 10.) m_hist_xresidual10GeV[ali]->Fill(xresidual);
		     if (fabs(1./qoverpt) > 20.) m_hist_xresidual20GeV[ali]->Fill(xresidual);

		     if (m_maxResidualDT13 >= 0.  &&  fabs(xresidual / 10.) > m_maxResidualDT13) goodResidual = false;
		     if (m_nearlyGlobalCoords[ali]) goodZ = false;
		  }

		  chamberId = dtid;  // DTChamberId(id.rawId());
	       }
	       else {  // if CSC
		  m_hist_xresidual[ali]->Fill(xresidual);
		  m_hist_yresidual[ali]->Fill(yresidual);
		  if (fabs(1./qoverpt) > 10.) {
		     m_hist_xresidual10GeV[ali]->Fill(xresidual);
		     m_hist_yresidual10GeV[ali]->Fill(yresidual);
		  }
		  if (fabs(1./qoverpt) > 20.) {
		     m_hist_xresidual20GeV[ali]->Fill(xresidual);
		     m_hist_yresidual20GeV[ali]->Fill(yresidual);
		  }

		  if (m_maxResidualCSC >= 0.  &&  fabs(xresidual / 10.) > m_maxResidualCSC) goodResidual = false;

		  CSCDetId cscId(id.rawId());
		  chamberId = CSCDetId(cscId.endcap(), cscId.station(), cscId.ring(), cscId.chamber());

		  for (std::vector<int>::const_iterator ring = m_ignoreCSCRings.begin();  ring != m_ignoreCSCRings.end();  ++ring) {
		     if (cscId.ring() == *ring) goodResidual = false;
		  }
	       } // end if DT/else CSC

	       if (goodResidual) {
		  if (chamberalignable.find(chamberId) == chamberalignable.end()) {
		     chamberalignable[chamberId] = ali;
		     chamberjtvj[chamberId] = AlgebraicSymMatrix(npar);
		     chamberjtve[chamberId] = AlgebraicVector(npar);
		     chamber_goodZ[chamberId] = goodZ;
		     chamber_goodXY[chamberId] = goodXY;
		  }

		  chamberjtvj[chamberId] += thisjtvj;
		  chamberjtve[chamberId] += thisjtve;
	       } // end if goodResidual

	       std::vector<bool> selector = ali->alignmentParameters()->selector();
	       int par = 0;
	       int i = 0;
	       for (std::vector<bool>::const_iterator sel = selector.begin();  sel != selector.end();  ++sel) {
		  if (*sel) {
		     double unitMultiplier = 1.;
		     bool goodParam = true;
		     if (par == 0) {unitMultiplier = 10.;  goodParam = goodXY;}
		     else if (par == 1) {unitMultiplier = 10.;  goodParam = goodXY;}
		     else if (par == 2) {unitMultiplier = 10.;  goodParam = goodZ;}
		     else if (par == 3) {unitMultiplier = 1000.;  goodParam = goodZ;}
		     else if (par == 4) {unitMultiplier = 1000.;  goodParam = goodZ;}
		     else if (par == 5) {unitMultiplier = 1000.;  goodParam = goodXY;}
		     else assert(false);

		     std::pair<Alignable*,int> index(ali, par);
		     if (goodResidual  &&  goodParam) {
			if ((id.subdetId() == MuonSubdetId::DT  &&  m_useHitWeightsInDTAlignment)  ||  (id.subdetId() == MuonSubdetId::CSC  &&  m_useHitWeightsInCSCAlignment)) {
			   m_hist_parameter[index]->Fill(p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));
			   if (fabs(1./qoverpt) > 10.) m_hist_parameter10GeV[index]->Fill(p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));
			   if (fabs(1./qoverpt) > 20.) m_hist_parameter20GeV[index]->Fill(p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));

			   m_hist_prof[index]->Fill(qoverpt, p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));
			}
			else {
			   m_hist_parameter[index]->Fill(p[i] * unitMultiplier);
			   if (fabs(1./qoverpt) > 10.) m_hist_parameter10GeV[index]->Fill(p[i] * unitMultiplier);
			   if (fabs(1./qoverpt) > 20.) m_hist_parameter20GeV[index]->Fill(p[i] * unitMultiplier);

			   m_hist_prof[index]->Fill(qoverpt, p[i] * unitMultiplier);
			}
		     }
	    
		     i++;
		  } // end if this is a parameter that we're plotting and aligning
		  par++;
	       } // end loop over parameters

	    } // end if hit passes cuts
	 } // end if this is something we're aligning
      } // end loop over hits/TSOSes

      for (std::map<DetId,Alignable*>::const_iterator chamberalignable_iter = chamberalignable.begin();  chamberalignable_iter != chamberalignable.end();  ++chamberalignable_iter) {
	 DetId chamberId = chamberalignable_iter->first;
	 Alignable *ali = chamberalignable_iter->second;
	 
	 AlgebraicSymMatrix jtvj = chamberjtvj[chamberId];
	 AlgebraicVector jtve = chamberjtve[chamberId];

	 int npar = jtvj.num_row();
	 AlgebraicVector p(npar);
	 AlgebraicVector e(npar);
	 for (int i = 0;  i < npar;  i++) {
	    if (jtvj[i][i] == 0.  ||  1./jtvj[i][i] < 0.) {
	       edm::LogError("MuonHIPAlignmentAlgorithm") << "Parameter weight is " << jtvj[i][i] << std::endl;
	       return;
	    }

	    p[i] = -jtve[i] / jtvj[i][i];    // here is where the important minus sign is applied
	    e[i] = sqrt(1./jtvj[i][i]);
	 }

	 bool goodZ = chamber_goodZ[chamberId];
	 bool goodXY = chamber_goodXY[chamberId];

	 std::vector<bool> selector = ali->alignmentParameters()->selector();
	 int par = 0;
	 int i = 0;
	 for (std::vector<bool>::const_iterator sel = selector.begin();  sel != selector.end();  ++sel) {
	    if (*sel) {
	       double unitMultiplier = 1.;
	       bool goodParam = true;
	       if (par == 0) {unitMultiplier = 10.;  goodParam = goodXY;}
	       else if (par == 1) {unitMultiplier = 10.;  goodParam = goodXY;}
	       else if (par == 2) {unitMultiplier = 10.;  goodParam = goodZ;}
	       else if (par == 3) {unitMultiplier = 1000.;  goodParam = goodZ;}
	       else if (par == 4) {unitMultiplier = 1000.;  goodParam = goodZ;}
	       else if (par == 5) {unitMultiplier = 1000.;  goodParam = goodXY;}
	       else assert(false);

	       DetId id(ali->id());
	       std::pair<Alignable*,int> index(ali, par);
	       if (goodParam) {
		  if ((id.subdetId() == MuonSubdetId::DT  &&  m_useHitWeightsInDTAlignment)  ||  (id.subdetId() == MuonSubdetId::CSC  &&  m_useHitWeightsInCSCAlignment)) {
		     m_hist_oparameter[index]->Fill(p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));
		     if (fabs(1./qoverpt) > 10.) m_hist_oparameter10GeV[index]->Fill(p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));
		     if (fabs(1./qoverpt) > 20.) m_hist_oparameter20GeV[index]->Fill(p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));
		     
		     m_hist_oprof[index]->Fill(qoverpt, p[i] * unitMultiplier, pow(e[i] * unitMultiplier, -2.));
		  }
		  else {
		     m_hist_oparameter[index]->Fill(p[i] * unitMultiplier);
		     if (fabs(1./qoverpt) > 10.) m_hist_oparameter10GeV[index]->Fill(p[i] * unitMultiplier);
		     if (fabs(1./qoverpt) > 20.) m_hist_oparameter20GeV[index]->Fill(p[i] * unitMultiplier);
		     
		     m_hist_oprof[index]->Fill(qoverpt, p[i] * unitMultiplier);
		  }
	       }
	       
	       i++;
	    } // end if this is a parameter that we're plotting and aligning
	    par++;
	 } // end loop over parameters
      } // end loop over individual chambers
      
   } // end loop over tracks/trajectories
}

void MuonHIPAlignmentAlgorithm::terminate() {
   if (m_collector.size() != 0) {
      for (std::vector<TH1*>::const_iterator histogram = m_all_histograms.begin();  histogram != m_all_histograms.end();  ++histogram) {
	 TList tlist;
	 for (std::vector<std::string>::const_iterator fileName = m_collector.begin();  fileName != m_collector.end();  ++fileName) {
	    TFile tfile(fileName->c_str(), "READ");  // do no harm!
	    TDirectory *dir = (TDirectory*)(tfile.Get(m_collectorDirectory.c_str()));
	    tlist.Add(dir->Get((*histogram)->GetName()));
	 }

	 if ((*histogram)->Merge(&tlist) == -1) {
	    throw cms::Exception("MuonHIPAlignmentAlgorithm") << "Could not merge " << (*histogram)->GetName() << " \"" << (*histogram)->GetTitle() << std::endl;
	 }
      } // end loop over all histograms, Merging them from previous files
   } // end if in collector mode

   if (m_align) {
      TF1 p2_DTrphi("p2_DTrphi", "[0] + [1]*x + [2]*x*x", -1./m_fitRangeDTrphi, 1./m_fitRangeDTrphi);
      TF1 p2_DTz("p2_DTz", "[0] + [1]*x + [2]*x*x", -1./m_fitRangeDTz, 1./m_fitRangeDTz);
      TF1 p2_CSCrphi("p2_CSCrphi", "[0] + [1]*x + [2]*x*x", -1./m_fitRangeCSCrphi, 1./m_fitRangeCSCrphi);
      TF1 p2_CSCz("p2_CSCz", "[0] + [1]*x + [2]*x*x", -1./m_fitRangeCSCz, 1./m_fitRangeCSCz);

      for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
	 AlgebraicVector params((*ali)->alignmentParameters()->numSelected());
	 AlgebraicSymMatrix cov((*ali)->alignmentParameters()->numSelected());

	 int numEntries = -1;

	 DetId id((*ali)->id());
	 std::cout << "***************************************************************** Alignable " << (*ali)->id() << std::endl;

	 std::vector<bool> selector = (*ali)->alignmentParameters()->selector();
	 int par = 0;
	 int i = 0;
	 for (std::vector<bool>::const_iterator sel = selector.begin();  sel != selector.end();  ++sel) {
	    if (*sel) {
	       double unitMultiplier = 1.;
	       if (par == 0) unitMultiplier = 10.;
	       else if (par == 1) unitMultiplier = 10.;
	       else if (par == 2) unitMultiplier = 10.;
	       else if (par == 3) unitMultiplier = 1000.;
	       else if (par == 4) unitMultiplier = 1000.;
	       else if (par == 5) unitMultiplier = 1000.;
	       else assert(false);

	       std::string function;
	       if (id.subdetId() == MuonSubdetId::DT) {
		  if (par == 0  ||  par == 1  ||  par == 5) {
		     function = std::string("p2_DTrphi");
		  }
		  else {
		     function = std::string("p2_DTz");
		  }
	       }
	       else {
		  if (par == 0  ||  par == 1  ||  par == 5) {
		     function = std::string("p2_CSCrphi");
		  }
		  else {
		     function = std::string("p2_CSCz");
		  }
	       }

	       std::pair<Alignable*,int> index(*ali, par);

	       std::cout << "Fitting " << m_hist_prof[index]->GetName() << " " << m_hist_prof[index]->GetTitle() << std::endl;
	       m_hist_prof[index]->Fit(function.c_str(), "R");
	       std::cout << std::endl;

	       std::cout << "Fitting " << m_hist_oprof[index]->GetName() << " " << m_hist_oprof[index]->GetTitle() << std::endl;
	       m_hist_oprof[index]->Fit(function.c_str(), "R");
	       std::cout << std::endl;
	    
	       long thisNumEntries;
	       if ((id.subdetId() == MuonSubdetId::DT  &&  m_useOneDTSuperLayerPerEntry)  ||  (id.subdetId() == MuonSubdetId::CSC  &&  m_useOneCSCChamberPerEntry)) {
		  params[i] = m_hist_oprof[index]->GetFunction(function.c_str())->GetParameter(0) / unitMultiplier;
		  cov[i][i] = pow(m_hist_oprof[index]->GetFunction(function.c_str())->GetParError(0) / unitMultiplier, 2.);

		  thisNumEntries = long(floor(m_hist_oparameter20GeV[index]->GetEntries()));
	       }
	       else {
		  params[i] = m_hist_prof[index]->GetFunction(function.c_str())->GetParameter(0) / unitMultiplier;
		  cov[i][i] = pow(m_hist_prof[index]->GetFunction(function.c_str())->GetParError(0) / unitMultiplier, 2.);

		  thisNumEntries = long(floor(m_hist_parameter20GeV[index]->GetEntries()));
	       }

	       if (numEntries < 0  ||  thisNumEntries < numEntries) numEntries = thisNumEntries;

	       i++;
	    } // end if this is a parameter we're plotting and aligning
	    par++;
	 } // end loop over parameters
	 if (numEntries < 0) numEntries = 0;

	 if (m_minTracksPerAlignable < 0  ||  numEntries >= m_minTracksPerAlignable) {
	    std::cout << "Applying alignment!  (Minimal number-of-track parameter had " << numEntries << " tracks) " << params;

	    AlignmentParameters *parnew = (*ali)->alignmentParameters()->cloneFromSelected(params, cov);
	    (*ali)->setAlignmentParameters(parnew);
	    m_alignmentParameterStore->applyParameters(*ali);
	    (*ali)->alignmentParameters()->setValid(true);
	 }
	 else {
	    std::cout << "Not aligning this one because it has only " << numEntries << " tracks (in the minimal number-of-tracks parameter)" << std::endl;
	 }

	 std::cout << std::endl;

      } // end loop over alignables
   } // end if align
}
