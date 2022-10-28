#include "RecoMuon/TrackingTools/test/MuonErrorMatrixAnalyzer.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
//#include <TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h>
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryParameters.h"

//#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TROOT.h"
#include "TList.h"
#include "TKey.h"

//
// constructors and destructor
//
MuonErrorMatrixAnalyzer::MuonErrorMatrixAnalyzer(const edm::ParameterSet& iConfig) {
  theCategory = "MuonErrorMatrixAnalyzer";
  //now do what ever initialization is needed
  theTrackLabel = iConfig.getParameter<edm::InputTag>("trackLabel");

  trackingParticleLabel = iConfig.getParameter<edm::InputTag>("trackingParticleLabel");

  //adding this for associator
  theAssocLabel = iConfig.getParameter<std::string>("associatorName");

  theErrorMatrixStore_Reported_pset = iConfig.getParameter<edm::ParameterSet>("errorMatrix_Reported_pset");

  theErrorMatrixStore_Residual_pset = iConfig.getParameter<edm::ParameterSet>("errorMatrix_Residual_pset");
  theErrorMatrixStore_Pull_pset = iConfig.getParameter<edm::ParameterSet>("errorMatrix_Pull_pset");

  thePlotFileName = iConfig.getParameter<std::string>("plotFileName");

  theRadius = iConfig.getParameter<double>("radius");
  if (theRadius != 0) {
    GlobalPoint O(0, 0, 0);
    Surface::RotationType R;
    refRSurface = Cylinder::build(theRadius, O, R);
    thePropagatorName = iConfig.getParameter<std::string>("propagatorName");
    thePropagatorToken = esConsumes(edm::ESInputTag("", thePropagatorName));
    theZ = iConfig.getParameter<double>("z");
    if (theZ != 0) {
      //plane can only be specified if R is specified
      GlobalPoint Opoz(0, 0, theZ);
      GlobalPoint Oneg(0, 0, -theZ);
      refZSurface[1] = Plane::build(Opoz, R);
      refZSurface[0] = Plane::build(Oneg, R);
    }
  }

  theGaussianPullFitRange = iConfig.getUntrackedParameter<double>("gaussianPullFitRange", 2.0);
}

MuonErrorMatrixAnalyzer::~MuonErrorMatrixAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void MuonErrorMatrixAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (theErrorMatrixStore_Reported) {
    analyze_from_errormatrix(iEvent, iSetup);
  }

  if (theErrorMatrixStore_Residual || theErrorMatrixStore_Pull) {
    analyze_from_pull(iEvent, iSetup);
  }
}

FreeTrajectoryState MuonErrorMatrixAnalyzer::refLocusState(const FreeTrajectoryState& fts) {
  if (theRadius == 0) {
    GlobalPoint vtx(0, 0, 0);
    TSCPBuilderNoMaterial tscpBuilder;
    FreeTrajectoryState PCAstate = tscpBuilder(fts, vtx).theState();
    return PCAstate;
  } else {
    //go to the cylinder surface, along momentum
    TrajectoryStateOnSurface onRef = thePropagator->propagate(fts, *refRSurface);

    if (!onRef.isValid()) {
      edm::LogError(theCategory) << " cannot propagate to cylinder of radius: " << theRadius;
      //try out the plane if specified
      if (theZ != 0) {
        onRef = thePropagator->propagate(fts, *(refZSurface[(fts.momentum().z() > 0)]));
        if (!onRef.isValid()) {
          edm::LogError(theCategory) << " cannot propagate to the plane of Z: "
                                     << ((fts.momentum().z() > 0) ? "+" : "-") << theZ << " either.";
          return FreeTrajectoryState();
        }  //invalid state
      }    //z plane is set
      else {
        return FreeTrajectoryState();
      }
    }  //invalid state
    else if (fabs(onRef.globalPosition().z()) > theZ && theZ != 0) {
      //try out the plane
      onRef = thePropagator->propagate(fts, *(refZSurface[(fts.momentum().z() > 0)]));
      if (!onRef.isValid()) {
        edm::LogError(theCategory) << " cannot propagate to the plane of Z: " << ((fts.momentum().z() > 0) ? "+" : "-")
                                   << theZ << " even though cylinder z indicates it should.";
      }  //invalid state
    }    //z further than the planes

    LogDebug(theCategory) << "reference state is:\n" << onRef;

    return (*onRef.freeState());
  }  // R=0
}

void MuonErrorMatrixAnalyzer::analyze_from_errormatrix(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (theRadius != 0) {
    //get a propagator
    thePropagator = iSetup.getHandle(thePropagatorToken);
  }

  //get the mag field
  theField = iSetup.getHandle(theFieldToken);

  //open a collection of track
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(theTrackLabel, tracks);

  //loop over them
  for (unsigned int it = 0; it != tracks->size(); ++it) {
    //   take their initial free state
    FreeTrajectoryState PCAstate = trajectoryStateTransform::initialFreeState((*tracks)[it], theField.product());
    if (PCAstate.position().mag() == 0) {
      edm::LogError(theCategory) << "invalid state from track initial state. skipping.\n" << PCAstate;
      continue;
    }

    FreeTrajectoryState trackRefState = refLocusState(PCAstate);
    if (trackRefState.position().mag() == 0)
      continue;

    AlgebraicSymMatrix55 errorMatrix = trackRefState.curvilinearError().matrix();

    double pT = trackRefState.momentum().perp();
    double eta = fabs(trackRefState.momentum().eta());
    double phi = trackRefState.momentum().phi();

    LogDebug(theCategory) << "error matrix:\n" << errorMatrix << "\n state: \n" << trackRefState;

    //fill the TProfile3D
    for (int i = 0; i != 5; ++i) {
      for (int j = i; j != 5; ++j) {
        //get the profile plot to fill
        TProfile3D* ij = theErrorMatrixStore_Reported->get(i, j);
        if (!ij) {
          edm::LogError(theCategory) << "cannot get profile " << i << " " << j;
          continue;
        }

        //get sigma squared or correlation factor
        double value = MuonErrorMatrix::Term(errorMatrix, i, j);
        ij->Fill(pT, eta, phi, value);
      }
    }
  }
}

void MuonErrorMatrixAnalyzer::analyze_from_pull(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //get the mag field
  theField = iSetup.getHandle(theFieldToken);

  //open a collection of track
  edm::Handle<View<reco::Track> > tracks;
  iEvent.getByLabel(theTrackLabel, tracks);

  //open a collection of Tracking particle
  edm::Handle<TrackingParticleCollection> TPtracks;
  iEvent.getByLabel(trackingParticleLabel, TPtracks);

  //get the associator
  edm::Handle<reco::TrackToTrackingParticleAssociator> associator;
  iEvent.getByLabel(theAssocLabel, associator);

  //associate
  reco::RecoToSimCollection recSimColl = associator->associateRecoToSim(tracks, TPtracks);

  LogDebug(theCategory) << "I have found: " << recSimColl.size() << " associations in total.";

  int it = 0;
  for (reco::RecoToSimCollection::const_iterator RtSit = recSimColl.begin(); RtSit != recSimColl.end(); ++RtSit) {
    //what am I loop over ?
    //the line below this has a problem
    //    const reco::TrackRef & track = RtSit->key;
    const std::vector<std::pair<TrackingParticleRef, double> >& tp = RtSit->val;

    //what do I want to get from those
    FreeTrajectoryState sim_fts;
    FreeTrajectoryState trackPCAstate;

    LogDebug(theCategory) << "I have found: " << tp.size() << " tracking particle associated with this reco::Track.";

    if (tp.size() != 0) {
      //take the match with best quality
      std::vector<std::pair<TrackingParticleRef, double> >::const_iterator vector_iterator = tp.begin();
      const std::pair<TrackingParticleRef, double>& pair_in_vector = *vector_iterator;
      //what am I looking at ?
      const TrackingParticleRef& trp = pair_in_vector.first;
      //the line below this has a syntax error
      //double quality & = pair_in_vector.second;
      //	 const TrackingParticle & best_associated_trackingparticle = trp.product();

      //	 work on the TrackingParticle
      //get reference point and momentum to make fts
      GlobalPoint position_sim_fts(trp->vx(), trp->vy(), trp->vz());
      GlobalVector momentum_sim_fts(trp->px(), trp->py(), trp->pz());
      int charge_sim = (int)trp->charge();
      GlobalTrajectoryParameters par_sim(position_sim_fts, momentum_sim_fts, charge_sim, theField.product());

      sim_fts = FreeTrajectoryState(par_sim);

      //	   work on the reco::Track
      trackPCAstate = trajectoryStateTransform::initialFreeState((*tracks)[it], theField.product());
      if (trackPCAstate.position().mag() == 0) {
        edm::LogError(theCategory) << "invalid state from track initial state. skipping.\n" << trackPCAstate;
        continue;
      }

      //get then both at the reference locus
      FreeTrajectoryState simRefState = refLocusState(sim_fts);
      FreeTrajectoryState trackRefState = refLocusState(trackPCAstate);

      if (simRefState.position().mag() == 0 || trackRefState.position().mag() == 0)
        continue;

      GlobalVector momentum_sim = simRefState.momentum();
      GlobalPoint point_sim = simRefState.position();

      GlobalVector momentum_track = trackRefState.momentum();
      GlobalPoint point_track = trackRefState.position();

      //conversion from global to curvinlinear parameters for both reco and sim track
      CurvilinearTrajectoryParameters sim_CTP(point_sim, momentum_sim, simRefState.charge());
      CurvilinearTrajectoryParameters track_CTP(point_track, momentum_track, trackRefState.charge());

      //These are the parameters for the CTP point
      AlgebraicVector5 sim_CTP_vector = sim_CTP.vector();
      AlgebraicVector5 track_CTP_vector = track_CTP.vector();
      const AlgebraicSymMatrix55& track_error = trackRefState.curvilinearError().matrix();

      double pT_sim = simRefState.momentum().perp();

      //get the momentum, eta, phi here
      double pT = trackRefState.momentum().perp();
      double eta = fabs(trackRefState.momentum().eta());
      double phi = trackRefState.momentum().phi();

      LogDebug(theCategory) << "The sim pT for this association is: " << pT_sim << " GeV"
                            << "sim state: \n"
                            << simRefState << "The track pT for this association is: " << pT << " GeV"
                            << "reco state: \n"
                            << trackRefState;

      //once have the momentum,eta,phi choose what bin it should go in
      //how do i get the correct bin
      double diff_i = 0, diff_j = 0;
      for (unsigned int i = 0; i != 5; ++i) {
        //do these if statements because if the parameter is phi it is not simply reco minus sim
        if (i != 2) {
          diff_i = sim_CTP_vector[i] - track_CTP_vector[i];
        } else {
          diff_i = deltaPhi(sim_CTP_vector[i], track_CTP_vector[i]);
        }

        for (unsigned int j = i; j < 5; ++j) {
          //do these if statements because if the parameter is phi it is not simply reco minus sim
          if (j != 2) {
            diff_j = sim_CTP_vector[j] - track_CTP_vector[j];
          } else {
            diff_j = deltaPhi(sim_CTP_vector[j], track_CTP_vector[j]);
          }

          //filling residual histograms
          if (theErrorMatrixStore_Residual) {
            unsigned int iH = theErrorMatrixStore_Residual->index(i, j);
            TProfile3D* ij = theErrorMatrixStore_Residual->get(i, j);
            if (!ij) {
              edm::LogError(theCategory) << i << " " << j << " not vali indexes. TProfile3D not found.";
              continue;
            }

            int iPt = theErrorMatrixStore_Residual->findBin(ij->GetXaxis(), pT) - 1;    // between 0 and maxbin-1;
            int iEta = theErrorMatrixStore_Residual->findBin(ij->GetYaxis(), eta) - 1;  // between 0 and maxbin-1;
            int iPhi = theErrorMatrixStore_Residual->findBin(ij->GetZaxis(), phi) - 1;  // between 0 and maxbin-1;

            TH1ptr& theH = (theHist_array_residual[iH])[index(ij, iPt, iEta, iPhi)];

            if (i == j) {
              LogDebug(theCategory) << "filling for: " << i;
              ((TH1*)theH)->Fill(diff_i);
            } else {
              LogDebug(theCategory) << "filling for: " << i << " " << j;
              ((TH2*)theH)->Fill(diff_i, diff_j);
            }
          }  //filling residual histograms

          //filling pull histograms
          if (theErrorMatrixStore_Pull) {
            unsigned int iH = theErrorMatrixStore_Pull->index(i, j);
            TProfile3D* ij = theErrorMatrixStore_Pull->get(i, j);
            if (!ij) {
              edm::LogError(theCategory) << i << " " << j << " not vali indexes. TProfile3D not found.";
              continue;
            }

            int iPt = theErrorMatrixStore_Pull->findBin(ij->GetXaxis(), pT) - 1;    // between 0 and maxbin-1;
            int iEta = theErrorMatrixStore_Pull->findBin(ij->GetYaxis(), eta) - 1;  // between 0 and maxbin-1;
            int iPhi = theErrorMatrixStore_Pull->findBin(ij->GetZaxis(), phi) - 1;  // between 0 and maxbin-1;

            TH1ptr& theH = (theHist_array_pull[iH])[index(ij, iPt, iEta, iPhi)];

            if (i == j) {
              LogDebug(theCategory) << "filling pulls for: " << i;
              ((TH1*)theH)->Fill(diff_i / sqrt(track_error(i, i)));
            } else {
              LogDebug(theCategory) << "filling pulls (2D) for: " << i << " " << j;
              ((TH2*)theH)->Fill(diff_i / sqrt(track_error(i, i)), diff_j / sqrt(track_error(j, j)));
            }
          }  // filling the pull histograms
        }
      }  //loop over all the 15 terms of the 5x5 matrix

    }  //end of if (tp.size()!=0)
    it++;
  }  //end of loop over the map
}

// ------------ method called once each job just before starting event loop  ------------

void MuonErrorMatrixAnalyzer::beginJob() {
  if (theErrorMatrixStore_Reported_pset.empty()) {
    theErrorMatrixStore_Reported = 0;
  } else {
    //create the error matrix provider, saying that you want to construct the things from here
    theErrorMatrixStore_Reported = new MuonErrorMatrix(theErrorMatrixStore_Reported_pset);
  }

  if (theErrorMatrixStore_Residual_pset.empty()) {
    theErrorMatrixStore_Residual = 0;
  } else {
    //create the error matrix provider for the alternative method
    theErrorMatrixStore_Residual = new MuonErrorMatrix(theErrorMatrixStore_Residual_pset);
  }

  if (theErrorMatrixStore_Pull_pset.empty()) {
    theErrorMatrixStore_Pull = 0;
  } else {
    //create the error matrix provider for the alternative method
    theErrorMatrixStore_Pull = new MuonErrorMatrix(theErrorMatrixStore_Pull_pset);
  }

  if (thePlotFileName == "") {
    thePlotFile = 0;
    //    thePlotDir=0;
    gROOT->cd();
  } else {
    edm::Service<TFileService> fs;
    //    thePlotDir = new TFileDirectory(fs->mkdir(thePlotFileName.c_str()));
    thePlotFile = TFile::Open(thePlotFileName.c_str(), "recreate");
    thePlotFile->cd();
    theBookKeeping = new TList;
  }

  //book histograms in this section
  //you will get them back from their name

  //Since the bin size is only specified in ErrorMatrix.cc must get bin size from the TProfile3D
  //need to choose which TProfile to get bin content info from.

  //create the 15 histograms, 5 of them TH1F, 10 of them TH2F
  if (theErrorMatrixStore_Residual) {
    for (unsigned int i = 0; i != 5; ++i) {
      for (unsigned int j = i; j < 5; ++j) {
        unsigned int iH = theErrorMatrixStore_Residual->index(i, j);
        TProfile3D* ij = theErrorMatrixStore_Residual->get(i, j);
        if (!ij) {
          edm::LogError(theCategory) << i << " " << j << " not valid indexes. TProfile3D not found.";
          continue;
        }
        unsigned int pTBin = (ij->GetNbinsX());
        unsigned int etaBin = (ij->GetNbinsY());
        unsigned int phiBin = (ij->GetNbinsZ());
        //allocate memory for all the histograms, for the given matrix term
        theHist_array_residual[iH] = new TH1ptr[maxIndex(ij)];

        //book the histograms now
        for (unsigned int iPt = 0; iPt < pTBin; ++iPt) {
          for (unsigned int iEta = 0; iEta < etaBin; iEta++) {
            for (unsigned int iPhi = 0; iPhi < phiBin; iPhi++) {
              TString hname = Form("%s_%i_%i_%i", ij->GetName(), iPt, iEta, iPhi);
              LogDebug(theCategory) << "preparing for: " << hname << "\n"
                                    << "trying to access at: " << index(ij, iPt, iEta, iPhi)
                                    << "while maxIndex is: " << maxIndex(ij);
              TH1ptr& theH = (theHist_array_residual[iH])[index(ij, iPt, iEta, iPhi)];

              const int bin[5] = {100, 100, 100, 5000, 100};
              const double min[5] = {-0.05, -0.05, -0.1, -0.005, -10.};
              const double max[5] = {0.05, 0.05, 0.1, 0.005, 10.};

              if (i == j) {
                //	      TString v(ErrorMatrix::vars[i]);
                TString htitle(Form("%s Pt:[%.1f,%.1f] Eta:[%.1f,%.1f] Phi:[%.1f,%.1f]",
                                    MuonErrorMatrix::vars[i].Data(),
                                    ij->GetXaxis()->GetBinLowEdge(iPt + 1),
                                    ij->GetXaxis()->GetBinUpEdge(iPt + 1),
                                    ij->GetYaxis()->GetBinLowEdge(iEta + 1),
                                    ij->GetYaxis()->GetBinUpEdge(iEta + 1),
                                    ij->GetZaxis()->GetBinLowEdge(iPhi + 1),
                                    ij->GetZaxis()->GetBinUpEdge(iPhi + 1)));
                //diagonal term
                thePlotFile->cd();
                theH = new TH1F(hname, htitle, bin[i], min[i], max[i]);
                //	      theH = thePlotDir->make<TH1F>(hname,htitle,bin[i],min[i],max[i]);
                theBookKeeping->Add(theH);
                theH->SetXTitle("#Delta_{reco-gen}(" + MuonErrorMatrix::vars[i] + ")");

                LogDebug(theCategory) << "creating a TH1F " << hname << " at: " << theH;
              } else {
                TString htitle(Form("%s Pt:[%.1f,%.1f] Eta:[%.1f,%.1f] Phi:[%.1f,%.1f]",
                                    ij->GetTitle(),
                                    ij->GetXaxis()->GetBinLowEdge(iPt + 1),
                                    ij->GetXaxis()->GetBinUpEdge(iPt + 1),
                                    ij->GetYaxis()->GetBinLowEdge(iEta + 1),
                                    ij->GetYaxis()->GetBinUpEdge(iEta + 1),
                                    ij->GetZaxis()->GetBinLowEdge(iPhi + 1),
                                    ij->GetZaxis()->GetBinUpEdge(iPhi + 1)));
                thePlotFile->cd();
                theH = new TH2F(hname, htitle, bin[i], min[i], max[i], 100, min[j], max[j]);
                //	      theH = thePlotDir->make<TH2F>(hname,htitle,bin[i],min[i],max[i],100,min[j],max[j]);
                theBookKeeping->Add(theH);
                theH->SetXTitle("#Delta_{reco-gen}(" + MuonErrorMatrix::vars[i] + ")");
                theH->SetYTitle("#Delta_{reco-gen}(" + MuonErrorMatrix::vars[j] + ")");

                LogDebug(theCategory) << "creating a TH2 " << hname << " at: " << theH;
              }
            }
          }
        }  //end of loop over the pt,eta,phi
      }
    }
  }

  //create the 15 histograms, 5 of them TH1F, 10 of them TH2F
  if (theErrorMatrixStore_Pull) {
    for (unsigned int i = 0; i != 5; ++i) {
      for (unsigned int j = i; j < 5; ++j) {
        unsigned int iH = theErrorMatrixStore_Pull->index(i, j);
        TProfile3D* ij = theErrorMatrixStore_Pull->get(i, j);
        if (!ij) {
          edm::LogError(theCategory) << i << " " << j << " not valid indexes. TProfile3D not found.";
          continue;
        }
        unsigned int pTBin = (ij->GetNbinsX());
        unsigned int etaBin = (ij->GetNbinsY());
        unsigned int phiBin = (ij->GetNbinsZ());
        //allocate memory for all the histograms, for the given matrix term
        theHist_array_pull[iH] = new TH1ptr[maxIndex(ij)];

        //book the histograms now
        for (unsigned int iPt = 0; iPt < pTBin; ++iPt) {
          for (unsigned int iEta = 0; iEta < etaBin; iEta++) {
            for (unsigned int iPhi = 0; iPhi < phiBin; iPhi++) {
              TString hname = Form("%s_p_%i_%i_%i", ij->GetName(), iPt, iEta, iPhi);
              LogDebug(theCategory) << "preparing for: " << hname << "\n"
                                    << "trying to access at: " << index(ij, iPt, iEta, iPhi)
                                    << "while maxIndex is: " << maxIndex(ij);
              TH1ptr& theH = (theHist_array_pull[iH])[index(ij, iPt, iEta, iPhi)];

              const int bin[5] = {200, 200, 200, 200, 200};
              const double min[5] = {-10, -10, -10, -10, -10};
              const double max[5] = {10, 10, 10, 10, 10};

              if (i == j) {
                //              TString v(ErrorMatrix::vars[i]);
                TString htitle(Form("%s Pt:[%.1f,%.1f] Eta:[%.1f,%.1f] Phi:[%.1f,%.1f]",
                                    MuonErrorMatrix::vars[i].Data(),
                                    ij->GetXaxis()->GetBinLowEdge(iPt + 1),
                                    ij->GetXaxis()->GetBinUpEdge(iPt + 1),
                                    ij->GetYaxis()->GetBinLowEdge(iEta + 1),
                                    ij->GetYaxis()->GetBinUpEdge(iEta + 1),
                                    ij->GetZaxis()->GetBinLowEdge(iPhi + 1),
                                    ij->GetZaxis()->GetBinUpEdge(iPhi + 1)));
                //diagonal term
                thePlotFile->cd();
                theH = new TH1F(hname, htitle, bin[i], min[i], max[i]);
                //	      theH = thePlotDir->make<TH1F>(hname,htitle,bin[i],min[i],max[i]);
                theBookKeeping->Add(theH);
                theH->SetXTitle("#Delta_{reco-gen}/#sigma(" + MuonErrorMatrix::vars[i] + ")");

                LogDebug(theCategory) << "creating a TH1F " << hname << " at: " << theH;
              } else {
                TString htitle(Form("%s Pt:[%.1f,%.1f] Eta:[%.1f,%.1f] Phi:[%.1f,%.1f]",
                                    ij->GetTitle(),
                                    ij->GetXaxis()->GetBinLowEdge(iPt + 1),
                                    ij->GetXaxis()->GetBinUpEdge(iPt + 1),
                                    ij->GetYaxis()->GetBinLowEdge(iEta + 1),
                                    ij->GetYaxis()->GetBinUpEdge(iEta + 1),
                                    ij->GetZaxis()->GetBinLowEdge(iPhi + 1),
                                    ij->GetZaxis()->GetBinUpEdge(iPhi + 1)));
                thePlotFile->cd();
                theH = new TH2F(hname, htitle, bin[i], min[i], max[i], 100, min[j], max[j]);
                //	      theH = thePlotDir->make<TH2F>(hname,htitle,bin[i],min[i],max[i],100,min[j],max[j]);
                theBookKeeping->Add(theH);
                theH->SetXTitle("#Delta_{reco-gen}/#sigma(" + MuonErrorMatrix::vars[i] + ")");
                theH->SetYTitle("#Delta_{reco-gen}/#sigma(" + MuonErrorMatrix::vars[j] + ")");

                LogDebug(theCategory) << "creating a TH2 " << hname << " at: " << theH;
              }
            }
          }
        }  //end of loop over the pt,eta,phi
      }
    }
  }
}

#include <TF2.h>

MuonErrorMatrixAnalyzer::extractRes MuonErrorMatrixAnalyzer::extract(TH2* h2) {
  extractRes res;

  //FIXME. no fitting procedure by default
  if (h2->GetEntries() < 1000000) {
    LogDebug(theCategory) << "basic method. not enough entries (" << h2->GetEntries() << ")";
    res.corr = h2->GetCorrelationFactor();
    res.x = h2->GetRMS(1);
    res.y = h2->GetRMS(2);
    return res;
  }

  //make a copy while rebinning
  int nX = std::max(1, h2->GetNbinsX() / 40);
  int nY = std::max(1, h2->GetNbinsY() / 40);
  LogDebug(theCategory) << "rebinning: " << h2->GetName() << " by: " << nX << " " << nY;
  TH2* h2r = h2->Rebin2D(nX, nY, "hnew");

  TString fname(h2->GetName());
  fname += +"_fit_f2";
  TF2* f2 = new TF2(
      fname, "[0]*exp(-0.5*(((x-[1])/[2])**2+((y-[3])/[4])**2 -2*[5]*(x-[1])*(y-[3])/([4]*[2])))", -10, 10, -10, 10);
  f2->SetParameters(h2->Integral(), 0, h2->GetRMS(1), 0, h2->GetRMS(2), h2->GetCorrelationFactor());
  f2->FixParameter(1, 0);
  f2->FixParameter(3, 0);

  if (fabs(h2->GetCorrelationFactor()) < 0.001) {
    LogDebug(theCategory) << "correlations neglected: " << h2->GetCorrelationFactor() << " for: " << h2->GetName();
    f2->FixParameter(5, 0);
  } else {
    f2->ReleaseParameter(5);
  }

  f2->SetParLimits(2, 0, 10 * h2->GetRMS(1));
  f2->SetParLimits(4, 0, 10 * h2->GetRMS(2));

  h2r->Fit(fname, "nqL");

  res.corr = f2->GetParameter(5);
  res.x = f2->GetParameter(2);
  res.y = f2->GetParameter(4);

  LogDebug(theCategory) << "\n variable: " << h2->GetXaxis()->GetTitle() << "\n RMS= " << h2->GetRMS(1)
                        << "\n fit= " << f2->GetParameter(2) << "\n variable: " << h2->GetYaxis()->GetTitle()
                        << "\n RMS= " << h2->GetRMS(2) << "\n fit= " << f2->GetParameter(4) << "\n correlation"
                        << "\n correlation factor= " << h2->GetCorrelationFactor()
                        << "\n fit=                " << f2->GetParameter(5);

  f2->Delete();
  h2r->Delete();

  return res;
}

// ------------ method called once each job just after ending the event loop  ------------
void MuonErrorMatrixAnalyzer::endJob() {
  LogDebug(theCategory) << "endJob begins.";
  //  std::cout<<"endJob of MuonErrorMatrixAnalyzer"<<std::endl;
  //evaluate the histograms to find the correlation factors and sigmas

  if (theErrorMatrixStore_Reported) {
    //close the error matrix method object
    theErrorMatrixStore_Reported->close();
  }

  //write the file with all the plots in it.
  TFile* thePlotFile = 0;
  if (thePlotFileName != "") {
    //    std::cout<<"trying to write in: "<<thePlotFileName<<std::endl;

    thePlotFile = TFile::Open(thePlotFileName.c_str(), "recreate");
    thePlotFile->cd();
    TListIter iter(theBookKeeping);
    //    std::cout<<"number of objects to write: "<<theBookKeeping->GetSize()<<std::endl;
    TObject* o = 0;
    while ((o = iter.Next())) {
      //	std::cout<<"writing: "<<o->GetName()<<" in file: "<<thePlotFile->GetName()<<std::endl;
      o->Write();
    }
    //thePlotFile->Write();
  }

  if (theErrorMatrixStore_Residual) {
    //extract the rms and correlation factor from the residual
    for (unsigned int i = 0; i != 5; ++i) {
      for (unsigned int j = i; j < 5; ++j) {
        unsigned int iH = theErrorMatrixStore_Residual->index(i, j);
        TProfile3D* ij = theErrorMatrixStore_Residual->get(i, j);
        TProfile3D* ii = theErrorMatrixStore_Residual->get(i, i);
        TProfile3D* jj = theErrorMatrixStore_Residual->get(j, j);
        if (!ij) {
          edm::LogError(theCategory) << i << " " << j << " not valid indexes. TProfile3D not found.";
          continue;
        }
        if (!ii) {
          edm::LogError(theCategory) << i << " " << i << " not valid indexes. TProfile3D not found.";
          continue;
        }
        if (!jj) {
          edm::LogError(theCategory) << j << " " << j << " not valid indexes. TProfile3D not found.";
          continue;
        }

        unsigned int pTBin = (ij->GetNbinsX());
        unsigned int etaBin = (ij->GetNbinsY());
        unsigned int phiBin = (ij->GetNbinsZ());

        //analyze the histograms now
        for (unsigned int iPt = 0; iPt < pTBin; ++iPt) {
          for (unsigned int iEta = 0; iEta < etaBin; iEta++) {
            for (unsigned int iPhi = 0; iPhi < phiBin; iPhi++) {
              double pt = ij->GetXaxis()->GetBinCenter(iPt + 1);
              double eta = ij->GetYaxis()->GetBinCenter(iEta + 1);
              double phi = ij->GetZaxis()->GetBinCenter(iPhi + 1);

              TH1ptr& theH = (theHist_array_residual[iH])[index(ij, iPt, iEta, iPhi)];
              LogDebug(theCategory) << "extracting for: " << pt << " " << eta << " " << phi
                                    << "at index: " << index(ij, iPt, iEta, iPhi)
                                    << "while maxIndex is: " << maxIndex(ij) << "\n named: " << theH->GetName();

              //FIXME. not using the i=j plots (TH1F)
              if (i != j) {
                extractRes r = extract((TH2*)theH);
                ii->Fill(pt, eta, phi, r.x);
                jj->Fill(pt, eta, phi, r.y);
                ij->Fill(pt, eta, phi, r.corr);
                LogDebug(theCategory) << "for: " << theH->GetName() << " rho is: " << r.corr << " sigma x= " << r.x
                                      << " sigma y= " << r.y;
              }

              //please free the memory !!!
              LogDebug(theCategory) << "freing memory of: " << theH->GetName();
              theH->Delete();
              theH = 0;
            }
          }
        }  //end of loop over the pt,eta,phi
      }
    }

    theErrorMatrixStore_Residual->close();
  }

  if (theErrorMatrixStore_Pull) {
    // extract the scale factors from the pull distribution
    TF1* f = new TF1("fit_for_theErrorMatrixStore_Pull", "gaus", -theGaussianPullFitRange, theGaussianPullFitRange);

    for (unsigned int i = 0; i != 5; ++i) {
      for (unsigned int j = i; j < 5; ++j) {
        unsigned int iH = theErrorMatrixStore_Pull->index(i, j);
        TProfile3D* ij = theErrorMatrixStore_Pull->get(i, j);
        TProfile3D* ii = theErrorMatrixStore_Pull->get(i, i);
        TProfile3D* jj = theErrorMatrixStore_Pull->get(j, j);
        if (!ij) {
          edm::LogError(theCategory) << i << " " << j << " not valid indexes. TProfile3D not found.";
          continue;
        }
        if (!ii) {
          edm::LogError(theCategory) << i << " " << i << " not valid indexes. TProfile3D not found.";
          continue;
        }
        if (!jj) {
          edm::LogError(theCategory) << j << " " << j << " not valid indexes. TProfile3D not found.";
          continue;
        }

        unsigned int pTBin = (ij->GetNbinsX());
        unsigned int etaBin = (ij->GetNbinsY());
        unsigned int phiBin = (ij->GetNbinsZ());

        //analyze the histograms now
        for (unsigned int iPt = 0; iPt < pTBin; ++iPt) {
          for (unsigned int iEta = 0; iEta < etaBin; iEta++) {
            for (unsigned int iPhi = 0; iPhi < phiBin; iPhi++) {
              double pt = ij->GetXaxis()->GetBinCenter(iPt + 1);
              double eta = ij->GetYaxis()->GetBinCenter(iEta + 1);
              double phi = ij->GetZaxis()->GetBinCenter(iPhi + 1);

              TH1ptr& theH = (theHist_array_pull[iH])[index(ij, iPt, iEta, iPhi)];
              LogDebug(theCategory) << "extracting for: " << pt << " " << eta << " " << phi
                                    << "at index: " << index(ij, iPt, iEta, iPhi)
                                    << "while maxIndex is: " << maxIndex(ij) << "\n named: " << theH->GetName();
              double value = 0;
              if (i != j) {
                //off diag term. not implemented yet. fill with ones
                ij->Fill(pt, eta, phi, 1.);
              } else {
                //diag term. perform a gaussian core fit (-2,2) to determine value
                ((TH1*)theH)->Fit(f->GetName(), "qnL", "");
                value = f->GetParameter(2);
                LogDebug(theCategory) << "scale factor is: " << value;
                ii->Fill(pt, eta, phi, value);
              }

              //please free the memory !!!
              LogDebug(theCategory) << "freing memory of: " << theH->GetName();
              theH->Delete();
              theH = 0;
            }
          }
        }  //end of loop over the pt,eta,phi
      }
    }

    theErrorMatrixStore_Pull->close();
  }

  //close the file with all the plots in it.
  if (thePlotFile) {
    thePlotFile->Close();
  }
}
