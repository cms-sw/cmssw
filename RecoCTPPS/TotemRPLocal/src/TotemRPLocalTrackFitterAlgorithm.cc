/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
* 	Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemRPLocalTrackFitterAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMatrixD.h"

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

TotemRPLocalTrackFitterAlgorithm::TotemRPLocalTrackFitterAlgorithm(const edm::ParameterSet &)
{
}

//----------------------------------------------------------------------------------------------------


void TotemRPLocalTrackFitterAlgorithm::reset()
{
  det_data_map_.clear();
}

//----------------------------------------------------------------------------------------------------

TotemRPLocalTrackFitterAlgorithm::RPDetCoordinateAlgebraObjs
TotemRPLocalTrackFitterAlgorithm::prepareReconstAlgebraData(unsigned int det_id, const CTPPSGeometry& tot_rp_geom)
{
  RPDetCoordinateAlgebraObjs det_algebra_obj;

  det_algebra_obj.centre_of_det_global_position_ = convert3vector(tot_rp_geom.getSensorTranslation(det_id));
 
  HepMC::ThreeVector rp_topology_stripaxis = rp_topology_.GetStripReadoutAxisDir();
  CLHEP::Hep3Vector rp_topology_stripaxis_clhep;
  
  rp_topology_stripaxis_clhep.setX(rp_topology_stripaxis.x());
  rp_topology_stripaxis_clhep.setY(rp_topology_stripaxis.y());
  rp_topology_stripaxis_clhep.setZ(rp_topology_stripaxis.z());
 
  TVector3 rd_dir = convert3vector( tot_rp_geom.localToGlobalDirection(det_id, rp_topology_stripaxis_clhep ) );
  
  TVector2 v(rd_dir.X(), rd_dir.Y());
  det_algebra_obj.readout_direction_ = v.Unit();
  det_algebra_obj.rec_u_0_ = 0.0;
  det_algebra_obj.available_ = true;
  det_algebra_obj.rec_u_0_ = - (det_algebra_obj.readout_direction_ * det_algebra_obj.centre_of_det_global_position_.XYvector());
  
  return det_algebra_obj;
}

//----------------------------------------------------------------------------------------------------

TotemRPLocalTrackFitterAlgorithm::RPDetCoordinateAlgebraObjs*
TotemRPLocalTrackFitterAlgorithm::getDetAlgebraData(unsigned int det_id, const CTPPSGeometry& tot_rp_geom)
{
  auto it = det_data_map_.find(det_id);
  if (it != det_data_map_.end())
  {
    return &(it->second);
  } else {
    det_data_map_[det_id] = prepareReconstAlgebraData(det_id, tot_rp_geom);
    return &det_data_map_[det_id];
  }
}

//----------------------------------------------------------------------------------------------------

bool TotemRPLocalTrackFitterAlgorithm::fitTrack(const edm::DetSetVector<TotemRPRecHit> &hits, double z_0,
    const CTPPSGeometry &tot_geom, TotemRPLocalTrack &fitted_track)
{
  fitted_track.setValid(false);

  // bind hits with their algebra objects
  struct HitWithAlg
  {
    unsigned int detId;
    const TotemRPRecHit *hit;
    RPDetCoordinateAlgebraObjs *alg;
  };
  
  vector<HitWithAlg> applicable_hits;

  for (auto &ds : hits)
  {
    unsigned int detId = ds.detId();

    for (auto &h : ds)
    {
      RPDetCoordinateAlgebraObjs * alg = getDetAlgebraData(detId, tot_geom);
      if (alg->available_)
        applicable_hits.push_back({ detId, &h, alg});
    }
  }
  
  if (applicable_hits.size() < 5)
    return false;
  
  TMatrixD H(applicable_hits.size(), 4);
  TVectorD V(applicable_hits.size());
  TVectorD V_inv(applicable_hits.size());
  TVectorD U(applicable_hits.size());
  
  for(unsigned int i = 0;  i < applicable_hits.size(); ++i)
  {
    RPDetCoordinateAlgebraObjs *alg_obj = applicable_hits[i].alg;

    H(i,0) = alg_obj->readout_direction_.X();
    H(i,1) = alg_obj->readout_direction_.Y();
    double delta_z = alg_obj->centre_of_det_global_position_.Z()-z_0;
    H(i,2) = alg_obj->readout_direction_.X()*delta_z;
    H(i,3) = alg_obj->readout_direction_.Y()*delta_z;
    double var = applicable_hits[i].hit->getSigma();
    var*=var;
    V[i] = var;
    V_inv[i] = 1.0/var;
    U[i] = applicable_hits[i].hit->getPosition() - alg_obj->rec_u_0_;
  }

  TMatrixD H_T_V_inv(TMatrixD::kTransposed, H);
  multiplyByDiagonalInPlace(H_T_V_inv, V_inv);
  TMatrixD V_a(H_T_V_inv);
  TMatrixD V_a_mult(V_a, TMatrixD::kMult, H);
  try
  {
    V_a_mult.Invert();
  }
  catch (cms::Exception &e)
  {
    LogError("TotemRPLocalTrackFitterAlgorithm") << "Error in TotemRPLocalTrackFitterAlgorithm::fitTrack > "
      << "Fit matrix is singular. Skipping.";
    return false;
  }

  TMatrixD u_to_a(V_a_mult, TMatrixD::kMult, H_T_V_inv);
  TVectorD a(U);
  a *= u_to_a;
  
  fitted_track.setZ0(z_0);
  fitted_track.setParameterVector(a);
  fitted_track.setCovarianceMatrix(V_a_mult);
  
  double Chi_2 = 0;
  for(unsigned int i=0; i<applicable_hits.size(); ++i)
  {
    RPDetCoordinateAlgebraObjs *alg_obj = applicable_hits[i].alg;
    TVector2 readout_dir = alg_obj->readout_direction_;
    double det_z = alg_obj->centre_of_det_global_position_.Z();
    double sigma_str = applicable_hits[i].hit->getSigma();
    double sigma_str_2 = sigma_str*sigma_str;
    TVector2 fited_det_xy_point = fitted_track.getTrackPoint(det_z);
    double U_readout = applicable_hits[i].hit->getPosition() - alg_obj->rec_u_0_;
    double U_fited = (readout_dir*=fited_det_xy_point);
    double residual = U_fited - U_readout;
    TMatrixD V_T_Cov_X_Y(1,2);
    V_T_Cov_X_Y(0,0) = readout_dir.X();
    V_T_Cov_X_Y(0,1) = readout_dir.Y();
    TMatrixD V_T_Cov_X_Y_mult(V_T_Cov_X_Y, TMatrixD::kMult, fitted_track.trackPointInterpolationCovariance(det_z));
    double fit_strip_var = V_T_Cov_X_Y_mult(0,0)*readout_dir.X() + V_T_Cov_X_Y_mult(0,1)*readout_dir.Y();
    double pull_normalization = sqrt(sigma_str_2 - fit_strip_var);
    double pull = residual/pull_normalization;
    
    Chi_2 += residual*residual / sigma_str_2;

    TotemRPLocalTrack::FittedRecHit hit_point(*(applicable_hits[i].hit), TVector3(fited_det_xy_point.X(),
      fited_det_xy_point.Y(), det_z), residual, pull);
    fitted_track.addHit(applicable_hits[i].detId, hit_point);
  }
  
  fitted_track.setChiSquared(Chi_2);
  fitted_track.setValid(true);
  return true;
}

//----------------------------------------------------------------------------------------------------

void TotemRPLocalTrackFitterAlgorithm::multiplyByDiagonalInPlace(TMatrixD &mt, const TVectorD &diag)
{
  for(int i=0; i<mt.GetNrows(); ++i)
  {
    for(int j=0; j<mt.GetNcols(); ++j)
    {
      mt[i][j] *= diag[j];
    }
  }
}
