/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
* 	Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemRPLocalTrackFitterAlgorithm
#define RecoCTPPS_TotemRPLocal_TotemRPLocalTrackFitterAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

#include "TVector3.h"
#include "TVector2.h"

#include <unordered_map>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Algorithm for fitting tracks through a single RP.
 **/
class TotemRPLocalTrackFitterAlgorithm
{
  public:
    TotemRPLocalTrackFitterAlgorithm(const edm::ParameterSet &conf);

    /// performs the track fit, returns true if successful
    bool fitTrack(const edm::DetSetVector<TotemRPRecHit> &hits, double z_0, const TotemRPGeometry &tot_geom, TotemRPLocalTrack &fitted_track);

    /// Resets the reconstruction-data cache.
    void reset();

  private:
    struct RPDetCoordinateAlgebraObjs
    {
      TVector3 centre_of_det_global_position_;
      double rec_u_0_;              ///< in mm, position of det. centre projected on readout direction
      TVector2 readout_direction_;  ///< non paralell projection and rot_cor included
      bool available_;              ///< if det should be included in the reconstruction
    };

    /// A cache of reconstruction data. Must be reset every time the geometry chagnges.
    std::unordered_map<unsigned int, RPDetCoordinateAlgebraObjs> det_data_map_;

    RPTopology rp_topology_;

    /// Returns the reconstruction data for the chosen detector from the cache DetReconstructionDataMap.
    /// If it is not yet in the cache, calls PrepareReconstAlgebraData to make it.
    RPDetCoordinateAlgebraObjs *getDetAlgebraData(unsigned int det_id, const TotemRPGeometry &tot_rp_geom);

    /// Build the reconstruction data.
    RPDetCoordinateAlgebraObjs prepareReconstAlgebraData(unsigned int det_id, const TotemRPGeometry &tot_rp_geom);

    /// A matrix multiplication shorthand.
    void multiplyByDiagonalInPlace(TMatrixD &mt, const TVectorD &diag);
    
    static TVector3 convert3vector(const CLHEP::Hep3Vector & v)
    {
      return TVector3(v.x(),v.y(),v.z()) ;
    }
};

#endif
