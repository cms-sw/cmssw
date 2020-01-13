#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEBase_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEBase_H 1

//-----------------------------------------------------------------------------
// \class        PixelCPEBase
// \description  Base class for pixel CPE's, with all the common code.
// Perform the position and error evaluation of pixel hits using
// the Det angle to estimate the track impact angle.
// Move geomCorrection to the concrete class. d.k. 06/06.
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008
// Change to use Generic error & Template calibration from DB - D.Fehling 11/08
//-----------------------------------------------------------------------------

#include <utility>
#include <vector>
#include "TMath.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

//--- For the configuration:
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"

// new errors
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
// old errors
//#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"

#include <unordered_map>

#include <iostream>
#ifdef EDM_ML_DEBUG
#include <atomic>
#endif

class RectangularPixelTopology;
class MagneticField;
class PixelCPEBase : public PixelClusterParameterEstimator {
public:
  struct DetParam {
    DetParam() {}
    const PixelGeomDetUnit* theDet;
    // gavril : replace RectangularPixelTopology with PixelTopology
    const PixelTopology* theTopol;
    const RectangularPixelTopology* theRecTopol;

    GeomDetType::SubDetector thePart;
    Local3DPoint theOrigin;
    float theThickness;
    float thePitchX;
    float thePitchY;

    float bz;  // local Bz
    float bx;  // local Bx
    LocalVector driftDirection;
    float widthLAFractionX;   // Width-LA to Offset-LA in X
    float widthLAFractionY;   // same in Y
    float lorentzShiftInCmX;  // a FULL shift, in cm
    float lorentzShiftInCmY;  // a FULL shift, in cm
    int detTemplateId;        // det if for templates & generic errors
    int detTemplateId2D;      // det if for 2D templates
  };

  struct ClusterParam {
    ClusterParam(const SiPixelCluster& cl) : theCluster(&cl) {}

    virtual ~ClusterParam() = default;

    const SiPixelCluster* theCluster;

    //--- Cluster-level quantities (filled in computeAnglesFrom....)
    float cotalpha;
    float cotbeta;

    // G.Giurgiu (05/14/08) track local coordinates
    // filled in computeAnglesFrom....
    float trk_lp_x;
    float trk_lp_y;

    // ggiurgiu@jhu.edu (12/01/2010) : Needed for calling topology methods
    // with track angles to handle surface deformations (bows/kinks)
    // filled in computeAnglesFrom.... (btw redundant with the 4 above)
    Topology::LocalTrackPred loc_trk_pred;

    //--- Probability  (protected by hasFilledProb_)
    float probabilityX_;
    float probabilityY_;
    float probabilityQ_;
    int qBin_;  // always filled by qbin

    bool isOnEdge_;              // filled in setTheClu
    bool hasBadPixels_ = false;  // (never used in current code)
    bool spansTwoROCs_;          // filled in setTheClu
    bool hasFilledProb_ = false;
    // ggiurgiu@jhu.edu (10/18/2008)
    bool with_track_angle;        // filled in computeAnglesFrom....
    bool filled_from_2d = false;  //

    // More detailed edge information (for CPE ClusterRepair, and elsewhere...)
    int edgeTypeX_ = 0;  // 0: not on edge, 1: low end on edge, 2: high end
    int edgeTypeY_ = 0;  // 0: not on edge, 1: low end on edge, 2: high end
  };

public:
  PixelCPEBase(edm::ParameterSet const& conf,
               const MagneticField* mag,
               const TrackerGeometry& geom,
               const TrackerTopology& ttopo,
               const SiPixelLorentzAngle* lorentzAngle,
               const SiPixelGenErrorDBObject* genErrorDBObject,
               const SiPixelTemplateDBObject* templateDBobject,
               const SiPixelLorentzAngle* lorentzAngleWidth,
               int flag = 0  // flag=0 for generic, =1 for templates
  );                         // NEW

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

  //--------------------------------------------------------------------------
  // Allow the magnetic field to be set/updated later.
  //--------------------------------------------------------------------------
  //inline void setMagField(const MagneticField *mag) const { magfield_ = mag; } // Not used, AH

  //--------------------------------------------------------------------------
  // Obtain the angles from the position of the DetUnit.
  //--------------------------------------------------------------------------

  inline ReturnType getParameters(const SiPixelCluster& cl, const GeomDetUnit& det) const override {
#ifdef EDM_ML_DEBUG
    nRecHitsTotal_++;
    //std::cout<<" in PixelCPEBase:localParameters(all) - "<<nRecHitsTotal_<<std::endl;  //dk
#endif

    DetParam const& theDetParam = detParam(det);
    std::unique_ptr<ClusterParam> theClusterParam = createClusterParam(cl);
    setTheClu(theDetParam, *theClusterParam);
    computeAnglesFromDetPosition(theDetParam, *theClusterParam);

    // localPosition( cl, det ) must be called before localError( cl, det ) !!!
    LocalPoint lp = localPosition(theDetParam, *theClusterParam);
    LocalError le = localError(theDetParam, *theClusterParam);
    SiPixelRecHitQuality::QualWordType rqw = rawQualityWord(*theClusterParam);
    auto tuple = std::make_tuple(lp, le, rqw);

    //std::cout<<" in PixelCPEBase:localParameters(all) - "<<lp.x()<<" "<<lp.y()<<std::endl;  //dk
    return tuple;
  }

  //--------------------------------------------------------------------------
  // In principle we could use the track too to obtain alpha and beta.
  //--------------------------------------------------------------------------
  inline ReturnType getParameters(const SiPixelCluster& cl,
                                  const GeomDetUnit& det,
                                  const LocalTrajectoryParameters& ltp) const override {
#ifdef EDM_ML_DEBUG
    nRecHitsTotal_++;
    //std::cout<<" in PixelCPEBase:localParameters(on track) - "<<nRecHitsTotal_<<std::endl;  //dk
#endif

    DetParam const& theDetParam = detParam(det);
    std::unique_ptr<ClusterParam> theClusterParam = createClusterParam(cl);
    setTheClu(theDetParam, *theClusterParam);
    computeAnglesFromTrajectory(theDetParam, *theClusterParam, ltp);

    // localPosition( cl, det ) must be called before localError( cl, det ) !!!
    LocalPoint lp = localPosition(theDetParam, *theClusterParam);
    LocalError le = localError(theDetParam, *theClusterParam);
    SiPixelRecHitQuality::QualWordType rqw = rawQualityWord(*theClusterParam);
    auto tuple = std::make_tuple(lp, le, rqw);

    //std::cout<<" in PixelCPEBase:localParameters(on track) - "<<lp.x()<<" "<<lp.y()<<std::endl;  //dk
    return tuple;
  }

private:
  virtual std::unique_ptr<ClusterParam> createClusterParam(const SiPixelCluster& cl) const = 0;

  //--------------------------------------------------------------------------
  // This is where the action happens.
  //--------------------------------------------------------------------------
  virtual LocalPoint localPosition(DetParam const& theDetParam, ClusterParam& theClusterParam) const = 0;
  virtual LocalError localError(DetParam const& theDetParam, ClusterParam& theClusterParam) const = 0;

  void fillDetParams();

  //-----------------------------------------------------------------------------
  //! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
  //! This way, we can keep the details of what is filled within the pixel
  //! code and not expose the Transient SiPixelRecHit to it as well.  The name
  //! of this function is chosen to match the one in SiPixelRecHit.
  //-----------------------------------------------------------------------------
  SiPixelRecHitQuality::QualWordType rawQualityWord(ClusterParam& theClusterParam) const;

protected:
  //--- All methods and data members are protected to facilitate (for now)
  //--- access from derived classes.

  typedef GloballyPositioned<double> Frame;

  //---------------------------------------------------------------------------
  //  Data members
  //---------------------------------------------------------------------------

  //--- Counters
#ifdef EDM_ML_DEBUG
  mutable std::atomic<int> nRecHitsTotal_;     //for debugging only
  mutable std::atomic<int> nRecHitsUsedEdge_;  //for debugging only
#endif

  // Added new members
  float lAOffset_;     // la used to calculate the offset from configuration (for testing)
  float lAWidthBPix_;  // la used to calculate the cluster width from conf.
  float lAWidthFPix_;  // la used to calculate the cluster width from conf.
  //bool useLAAlignmentOffsets_; // lorentz angle offsets detrmined by alignment
  bool useLAOffsetFromConfig_;  // lorentz angle used to calculate the offset
  bool useLAWidthFromConfig_;   // lorentz angle used to calculate the cluster width
  bool useLAWidthFromDB_;       // lorentz angle used to calculate the cluster width

  //--- Global quantities
  int theVerboseLevel;  // algorithm's verbosity
  int theFlag_;         // flag to recognice if we are in generic or templates

  const MagneticField* magfield_;  // magnetic field
  const TrackerGeometry& geom_;    // geometry
  const TrackerTopology& ttopo_;   // Tracker Topology

  const SiPixelLorentzAngle* lorentzAngle_;
  const SiPixelLorentzAngle* lorentzAngleWidth_;  // for the charge width (generic)

  const SiPixelGenErrorDBObject* genErrorDBObject_;  // NEW
  //const SiPixelCPEGenericErrorParm * genErrorParm_;  // OLD

  const SiPixelTemplateDBObject* templateDBobject_;
  bool alpha2Order;  // switch on/off E.B effect.

  bool DoLorentz_;
  bool LoadTemplatesFromDB_;

  //---------------------------------------------------------------------------
  //  Geometrical services to subclasses.
  //---------------------------------------------------------------------------
protected:
  void computeAnglesFromDetPosition(DetParam const& theDetParam, ClusterParam& theClusterParam) const;

  void computeAnglesFromTrajectory(DetParam const& theDetParam,
                                   ClusterParam& theClusterParam,
                                   const LocalTrajectoryParameters& ltp) const;

  void setTheClu(DetParam const&, ClusterParam& theClusterParam) const;

  LocalVector driftDirection(DetParam& theDetParam, GlobalVector bfield) const;
  LocalVector driftDirection(DetParam& theDetParam, LocalVector bfield) const;
  void computeLorentzShifts(DetParam&) const;

  //---------------------------------------------------------------------------
  //  Cluster-level services.
  //---------------------------------------------------------------------------

  DetParam const& detParam(const GeomDetUnit& det) const;

  using DetParams = std::vector<DetParam>;

  DetParams m_DetParams = DetParams(1440);
};

#endif
