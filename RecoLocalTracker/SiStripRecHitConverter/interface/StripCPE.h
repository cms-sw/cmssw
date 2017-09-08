#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPE_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPE_H

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

class StripTopology;

class StripCPE : public StripClusterParameterEstimator 
{
public:
  using StripClusterParameterEstimator::localParameters;

  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster& cl, const GeomDetUnit&) const override;
  
  StripCPE( edm::ParameterSet & conf, 
	    const MagneticField&, 
	    const TrackerGeometry&, 
	    const SiStripLorentzAngle&,
	    const SiStripBackPlaneCorrection&,
	    const SiStripConfObject&,
	    const SiStripLatency&);    
  LocalVector driftDirection(const StripGeomDetUnit* det) const override;

 struct Param {
    Param() : topology(nullptr) {}
    StripTopology const * topology;
    LocalVector drift;
    float thickness, invThickness,pitch_rel_err2, maxLength;
    int nstrips;
    float backplanecorrection;
    SiStripDetId::ModuleGeometry moduleGeom;
    float coveredStrips(const LocalVector&, const LocalPoint&) const;
  };

  
  struct AlgoParam {
    Param const & p; const LocalTrajectoryParameters & ltp;
    SiStripDetId::SubDetector loc; float afullProjection; float corr;
  };


  virtual StripClusterParameterEstimator::LocalValues
  localParameters( const SiStripCluster& cl, AlgoParam const & ap) const {
    return std::make_pair(LocalPoint(), LocalError());
  }
  
  AlgoParam getAlgoParam(const GeomDetUnit& det, const LocalTrajectoryParameters & ltp) const {

    StripCPE::Param const & p = param(det);
    SiStripDetId::SubDetector loc = SiStripDetId( det.geographicalId() ).subDetector();  
 
    LocalVector track = ltp.directionNotNormalized();
    track *= -p.thickness;

    const float fullProjection = p.coveredStrips( track+p.drift, ltp.position());

    auto const corr = -  0.5f*(1.f-p.backplanecorrection) * fullProjection
      + 0.5f*p.coveredStrips(track, ltp.position());

    return  AlgoParam{p,ltp,loc,std::abs(fullProjection),corr};
  }
  
 protected:  

  const bool peakMode_;
  const TrackerGeometry & geom_;
  const MagneticField& magfield_ ;
  const SiStripLorentzAngle& LorentzAngleMap_;
  const SiStripBackPlaneCorrection& BackPlaneCorrectionMap_;
  std::vector<float> xtalk1;
  std::vector<float> xtalk2;

  Param const & param(const GeomDetUnit& det) const {
    return m_Params[det.index()-m_off];
  }

private:

  void fillParams();
  typedef  std::vector<Param> Params;  
  Params m_Params;
  unsigned int m_off;

};
#endif
