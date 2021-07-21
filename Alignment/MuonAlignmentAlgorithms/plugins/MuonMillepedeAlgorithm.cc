#include <fstream>

#include "TFile.h"
#include "TTree.h"
#include "TKey.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include "DataFormats/TrackReco/interface/Track.h"

#include "Alignment/MuonAlignmentAlgorithms/plugins/MuonMillepedeAlgorithm.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"

// Constructor ----------------------------------------------------------------

MuonMillepedeAlgorithm::MuonMillepedeAlgorithm(const edm::ParameterSet &cfg, const edm::ConsumesCollector &iC)
    : AlignmentAlgorithmBase(cfg, iC) {
  // parse parameters

  edm::LogWarning("Alignment") << "[MuonMillepedeAlgorithm] constructed.";

  collec_f = cfg.getParameter<std::string>("CollectionFile");

  isCollectionJob = cfg.getParameter<bool>("isCollectionJob");

  collec_path = cfg.getParameter<std::string>("collectionPath");

  collec_number = cfg.getParameter<int>("collectionNumber");

  outputCollName = cfg.getParameter<std::string>("outputCollName");

  ptCut = cfg.getParameter<double>("ptCut");

  chi2nCut = cfg.getParameter<double>("chi2nCut");
}

// Call at beginning of job ---------------------------------------------------

void MuonMillepedeAlgorithm::initialize(const edm::EventSetup &setup,
                                        AlignableTracker *tracker,
                                        AlignableMuon *muon,
                                        AlignableExtras *extras,
                                        AlignmentParameterStore *store) {
  edm::LogWarning("Alignment") << "[MuonMillepedeAlgorithm] Initializing...";

  // accessor Det->AlignableDet
  theAlignableDetAccessor = new AlignableNavigator(tracker, muon);

  // set alignmentParameterStore
  theAlignmentParameterStore = store;

  // get alignables
  theAlignables = theAlignmentParameterStore->alignables();
}

void MuonMillepedeAlgorithm::collect() {
  std::map<std::string, TMatrixD *> map;

  for (int c_job = 0; c_job < collec_number; ++c_job) {
    char name_f[40];
    snprintf(name_f, sizeof(name_f), "%s_%d/%s", collec_path.c_str(), c_job, collec_f.c_str());
    TFile file_it(name_f);

    if (file_it.IsZombie())
      continue;

    TList *m_list = file_it.GetListOfKeys();
    if (m_list == nullptr) {
      return;
    }
    TKey *index = (TKey *)m_list->First();
    if (index == nullptr) {
    }
    if (index != nullptr) {
      do {
        std::string objectName(index->GetName());
        TMatrixD *mat = (TMatrixD *)index->ReadObj();
        std::map<std::string, TMatrixD *>::iterator node = map.find(objectName);
        if (node == map.end()) {
          TMatrixD *n_mat = new TMatrixD(mat->GetNrows(), mat->GetNcols());
          map.insert(make_pair(objectName, n_mat));
        }
        *(map[objectName]) += *mat;
        index = (TKey *)m_list->After(index);
      } while (index != nullptr);
    }
    file_it.Close();
  }

  TFile theFile2(outputCollName.c_str(), "recreate");
  theFile2.cd();

  std::map<std::string, TMatrixD *>::iterator m_it = map.begin();
  for (; m_it != map.end(); ++m_it) {
    if (m_it->first.find("_invCov") != std::string::npos) {
      std::string id_s = m_it->first.substr(0, m_it->first.find("_invCov"));
      std::string id_w = id_s + "_weightRes";
      std::string id_n = id_s + "_N";
      std::string cov = id_s + "_cov";
      std::string sol = id_s + "_sol";

      //Covariance calculation
      TMatrixD invMat(m_it->second->GetNrows(), m_it->second->GetNcols());
      invMat = *(m_it->second);
      invMat.Invert();
      //weighted residuals
      TMatrixD weightMat(m_it->second->GetNcols(), 1);
      weightMat = *(map[id_w]);
      //Solution of the linear system
      TMatrixD solution(m_it->second->GetNrows(), 1);
      solution = invMat * weightMat;
      //Number of Tracks
      TMatrixD n(1, 1);
      n = *(map[id_n]);

      invMat.Write(cov.c_str());
      n.Write(id_n.c_str());
      solution.Write(sol.c_str());
    }
  }
  theFile2.Write();
  theFile2.Close();
}

// Call at end of job ---------------------------------------------------------

void MuonMillepedeAlgorithm::terminate(const edm::EventSetup &iSetup) {
  if (isCollectionJob) {
    this->collect();
    return;
  }

  edm::LogWarning("Alignment") << "[MuonMillepedeAlgorithm] Terminating";

  // iterate over alignment parameters
  for (const auto &ali : theAlignables) {
    // Alignment parameters
    // AlignmentParameters* par = ali->alignmentParameters();
    edm::LogInfo("Alignment") << "now apply params";
    theAlignmentParameterStore->applyParameters(ali);
    // set these parameters 'valid'
    ali->alignmentParameters()->setValid(true);
  }

  edm::LogWarning("Alignment") << "[MuonMillepedeAlgorithm] Writing aligned parameters to file: "
                               << theAlignables.size();

  TFile *theFile = new TFile(collec_f.c_str(), "recreate");
  theFile->cd();
  std::map<std::string, AlgebraicMatrix *>::iterator invCov_it = map_invCov.begin();
  std::map<std::string, AlgebraicMatrix *>::iterator weightRes_it = map_weightRes.begin();
  std::map<std::string, AlgebraicMatrix *>::iterator n_it = map_N.begin();
  for (; n_it != map_N.end(); ++invCov_it, ++weightRes_it, ++n_it) {
    TMatrixD tmat_invcov(0, 0);
    this->toTMat(invCov_it->second, &tmat_invcov);
    TMatrixD tmat_weightres(0, 0);
    this->toTMat(weightRes_it->second, &tmat_weightres);
    TMatrixD tmat_n(0, 0);
    this->toTMat(n_it->second, &tmat_n);

    tmat_invcov.Write(invCov_it->first.c_str());
    tmat_weightres.Write(weightRes_it->first.c_str());
    tmat_n.Write(n_it->first.c_str());
  }

  theFile->Write();
  theFile->Close();
}

void MuonMillepedeAlgorithm::toTMat(AlgebraicMatrix *am_mat, TMatrixD *tmat_mat) {
  tmat_mat->ResizeTo(am_mat->num_row(), am_mat->num_col());
  for (int c_i = 0; c_i < am_mat->num_row(); ++c_i) {
    for (int c_j = 0; c_j < am_mat->num_col(); ++c_j) {
      (*tmat_mat)(c_i, c_j) = (*am_mat)[c_i][c_j];
    }
  }
}

// Run the algorithm on trajectories and tracks -------------------------------

void MuonMillepedeAlgorithm::run(const edm::EventSetup &setup, const EventInfo &eventInfo) {
  if (isCollectionJob) {
    return;
  }

  // loop over tracks
  //int t_counter = 0;
  const ConstTrajTrackPairCollection &tracks = eventInfo.trajTrackPairs();
  for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin(); it != tracks.end(); it++) {
    const Trajectory *traj = (*it).first;
    const reco::Track *track = (*it).second;

    float pt = track->pt();
    float chi2n = track->normalizedChi2();
#ifdef EDM_ML_DEBUG
    float eta = track->eta();
    float phi = track->phi();
    //int   ndof = track->ndof();
    int nhit = track->numberOfValidHits();

    LogDebug("Alignment") << "New track pt,eta,phi,chi2n,hits: " << pt << "," << eta << "," << phi << "," << chi2n
                          << "," << nhit;
#endif

    //Accept or not accept the track
    if (pt > ptCut && chi2n < chi2nCut) {
      std::vector<const TransientTrackingRecHit *> hitvec;
      std::vector<TrajectoryStateOnSurface> tsosvec;

      std::vector<TrajectoryMeasurement> measurements = traj->measurements();

      //In this loop the measurements and hits are extracted and put on two vectors
      for (std::vector<TrajectoryMeasurement>::iterator im = measurements.begin(); im != measurements.end(); im++) {
        TrajectoryMeasurement meas = *im;
        const TransientTrackingRecHit *hit = &(*meas.recHit());
        //We are not very strict at this point
        if (hit->isValid() && theAlignableDetAccessor->detAndSubdetInMap(hit->geographicalId())) {
          //***Forward
          const TrajectoryStateOnSurface &tsos = meas.forwardPredictedState();
          if (tsos.isValid()) {
            hitvec.push_back(hit);
            tsosvec.push_back(tsos);
          }
        }
      }

      // transform RecHit vector to AlignableDet vector
      std::vector<AlignableDetOrUnitPtr> alidetvec = theAlignableDetAccessor->alignablesFromHits(hitvec);

      // get concatenated alignment parameters for list of alignables
      CompositeAlignmentParameters aap = theAlignmentParameterStore->selectParameters(alidetvec);

      std::vector<TrajectoryStateOnSurface>::const_iterator itsos = tsosvec.begin();
      std::vector<const TransientTrackingRecHit *>::const_iterator ihit = hitvec.begin();

      //int ch_counter = 0;

      while (itsos != tsosvec.end()) {
        // get AlignableDet for this hit
        const GeomDet *det = (*ihit)->det();
        AlignableDetOrUnitPtr alidet = theAlignableDetAccessor->alignableFromGeomDet(det);

        // get relevant Alignable
        Alignable *ali = aap.alignableFromAlignableDet(alidet);

        //To be sure that the ali is not null and that it's a DT segment
        if (ali != nullptr && (*ihit)->geographicalId().subdetId() == 1) {
          DTChamberId m_Chamber(det->geographicalId());
          //Station 4 does not contain Theta SL
          if ((*ihit)->dimension() == 4 || ((*ihit)->dimension() == 2 && m_Chamber.station() == 4))
          //if((*ihit)->dimension() == 4)
          {
            edm::LogInfo("Alignment") << "Entrando";

            AlignmentParameters *params = ali->alignmentParameters();

            edm::LogInfo("Alignment") << "Entrando";
            //(dx/dz,dy/dz,x,y) for a 4DSegment
            AlgebraicVector ihit4D = (*ihit)->parameters();

            //The innerMostState always contains the Z
            //(q/pt,dx/dz,dy/dz,x,y)
            AlgebraicVector5 alivec = (*itsos).localParameters().mixedFormatVector();

            //The covariance matrix follows the sequence
            //(q/pt,dx/dz,dy/dz,x,y) but we reorder to
            //(x,y,dx/dz,dy/dz)
            AlgebraicSymMatrix55 rawCovMat = (*itsos).localError().matrix();
            AlgebraicMatrix CovMat(4, 4);
            int m_index[] = {2, 3, 0, 1};
            for (int c_ei = 0; c_ei < 4; ++c_ei) {
              for (int c_ej = 0; c_ej < 4; ++c_ej) {
                CovMat[m_index[c_ei]][m_index[c_ej]] = rawCovMat(c_ei + 1, c_ej + 1);
              }
            }

            int inv_check;
            //printM(CovMat);
            CovMat.invert(inv_check);
            if (inv_check != 0) {
              edm::LogError("Alignment") << "Covariance Matrix inversion failed";
              return;
            }

            //The order is changed to:
            // (x,0,dx/dz,0) MB4 Chamber
            // (x,y,dx/dz,dy/dz) Not MB4 Chamber
            AlgebraicMatrix residuals(4, 1);
            if (m_Chamber.station() == 4) {
              //Filling Residuals
              residuals[0][0] = ihit4D[2] - alivec[3];
              residuals[1][0] = 0.0;
              residuals[2][0] = ihit4D[0] - alivec[1];
              residuals[3][0] = 0.0;
              //The error in the Theta coord is set to infinite
              CovMat[1][0] = 0.0;
              CovMat[1][1] = 0.0;
              CovMat[1][2] = 0.0;
              CovMat[1][3] = 0.0;
              CovMat[0][1] = 0.0;
              CovMat[2][1] = 0.0;
              CovMat[3][1] = 0.0;
              CovMat[3][0] = 0.0;
              CovMat[3][2] = 0.0;
              CovMat[3][3] = 0.0;
              CovMat[0][3] = 0.0;
              CovMat[2][3] = 0.0;
            } else {
              //Filling Residuals
              residuals[0][0] = ihit4D[2] - alivec[3];
              residuals[1][0] = ihit4D[3] - alivec[4];
              residuals[2][0] = ihit4D[0] - alivec[1];
              residuals[3][0] = ihit4D[1] - alivec[2];
            }

            // Derivatives
            AlgebraicMatrix derivsAux = params->selectedDerivatives(*itsos, alidet);

            std::vector<bool> mb4_mask;
            std::vector<bool> selection;
            //To be checked
            mb4_mask.push_back(true);
            mb4_mask.push_back(false);
            mb4_mask.push_back(true);
            mb4_mask.push_back(true);
            mb4_mask.push_back(true);
            mb4_mask.push_back(false);
            selection.push_back(true);
            selection.push_back(true);
            selection.push_back(false);
            selection.push_back(false);
            selection.push_back(false);
            selection.push_back(false);
            int nAlignParam = 0;
            if (m_Chamber.station() == 4) {
              for (int icor = 0; icor < 6; ++icor) {
                if (mb4_mask[icor] && selection[icor])
                  nAlignParam++;
              }
            } else {
              nAlignParam = derivsAux.num_row();
            }

            AlgebraicMatrix derivs(nAlignParam, 4);
            if (m_Chamber.station() == 4) {
              int der_c = 0;
              for (int icor = 0; icor < 6; ++icor) {
                if (mb4_mask[icor] && selection[icor]) {
                  for (int ccor = 0; ccor < 4; ++ccor) {
                    derivs[der_c][ccor] = derivsAux[icor][ccor];
                    ++der_c;
                  }
                }
              }
            } else {
              derivs = derivsAux;
            }

            //for(int co = 0; co < derivs.num_row(); ++co)
            //{
            //  for(int ci = 0; ci < derivs.num_col(); ++ci)
            //  {
            //     edm::LogInfo("Alignment") << "Derivatives: " << co << " " << ci << " " << derivs[co][ci] << " ";
            //  }
            //}

            AlgebraicMatrix derivsT = derivs.T();
            AlgebraicMatrix invCov = derivs * CovMat * derivsT;
            AlgebraicMatrix weightRes = derivs * CovMat * residuals;

            //this->printM(derivs);
            //this->printM(CovMat);
            //this->printM(residuals);

            char name[40];
            snprintf(
                name, sizeof(name), "Chamber_%d_%d_%d", m_Chamber.wheel(), m_Chamber.station(), m_Chamber.sector());
            std::string chamId(name);
            //MB4 need a special treatment
            /*AlgebraicMatrix invCovMB4(2,2);
		    AlgebraicMatrix weightResMB4(2,1); 
		    if( m_Chamber.station() == 4 )
		    { 
		      int m_index_2[] = {0,2};
		      for(int c_i = 0; c_i < 2; ++c_i)
		      {
			weightResMB4[c_i][0] = weightRes[m_index_2[c_i]][0];
			for(int c_j = 0; c_j < 2; ++c_j)
			{
			  invCovMB4[c_i][c_j] = invCov[m_index_2[c_i]][m_index_2[c_j]];
			}  
		      }
		      this->updateInfo(invCovMB4, weightResMB4, residuals, chamId); 
		    }
		    else
		    {
		      this->updateInfo(invCov, weightRes, residuals, chamId); 
		    }*/
            this->updateInfo(invCov, weightRes, residuals, chamId);
          }
        }
        itsos++;
        ihit++;
      }
    }
  }  // end of track loop
}

//Auxiliar
void MuonMillepedeAlgorithm::printM(const AlgebraicMatrix &m) {
  //for(int i = 0; i < m.num_row(); ++i)
  // {
  //  for(int j = 0; j < m.num_col(); ++j)
  //  {
  //    std::cout << m[i][j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
}

void MuonMillepedeAlgorithm::updateInfo(const AlgebraicMatrix &m_invCov,
                                        const AlgebraicMatrix &m_weightRes,
                                        const AlgebraicMatrix &m_res,
                                        std::string id) {
  std::string id_invCov = id + "_invCov";
  std::string id_weightRes = id + "_weightRes";
  std::string id_n = id + "_N";

  edm::LogInfo("Alignment") << "Entrando";

  std::map<std::string, AlgebraicMatrix *>::iterator node = map_invCov.find(id_invCov);
  if (node == map_invCov.end()) {
    AlgebraicMatrix *f_invCov = new AlgebraicMatrix(m_invCov.num_row(), m_invCov.num_col());
    AlgebraicMatrix *f_weightRes = new AlgebraicMatrix(m_weightRes.num_row(), m_weightRes.num_col());
    AlgebraicMatrix *f_n = new AlgebraicMatrix(1, 1);

    map_invCov.insert(make_pair(id_invCov, f_invCov));
    map_weightRes.insert(make_pair(id_weightRes, f_weightRes));
    map_N.insert(make_pair(id_n, f_n));

    for (int iCount = 0; iCount < 4; ++iCount) {
      char name[40];
      snprintf(name, sizeof(name), "%s_var_%d", id.c_str(), iCount);
      std::string idName(name);
      float range = 5.0;
      //if( iCount == 0 || iCount == 1 ) {
      //  range = 0.01;
      //}
      TH1D *histo = fs->make<TH1D>(idName.c_str(), idName.c_str(), 200, -range, range);
      histoMap.insert(make_pair(idName, histo));
    }
  }

  *map_invCov[id_invCov] = *map_invCov[id_invCov] + m_invCov;
  *map_weightRes[id_weightRes] = *map_weightRes[id_weightRes] + m_weightRes;
  (*map_N[id_n])[0][0]++;

  for (int iCount = 0; iCount < 4; ++iCount) {
    char name[40];
    snprintf(name, sizeof(name), "%s_var_%d", id.c_str(), iCount);
    std::string idName(name);
    histoMap[idName]->Fill(m_res[iCount][0]);
  }
}

DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory, MuonMillepedeAlgorithm, "MuonMillepedeAlgorithm");
