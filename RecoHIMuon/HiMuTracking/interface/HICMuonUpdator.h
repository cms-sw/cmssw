#ifndef _TRACKER_HICMUUPDATOR_H_
#define _TRACKER_HICMUUPDATOR_H_
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <vector>

class HICMuonUpdator{

public:
  HICMuonUpdator(double&la1,double&la2, const MagneticField * mf){HICConst hc; zvert=hc.zvert;
                                        thePhiWin=la1; theZWin=la2; field = mf;};
  virtual ~HICMuonUpdator(){}
  TrajectoryStateOnSurface update(const Trajectory& mt,
                                  const TrajectoryStateOnSurface&, 
                                  const TrajectoryMeasurement&, 
				  const DetLayer*, 
				  double&, double&) const;
      

  TrajectoryStateOnSurface updateBarrel(std::vector<double>& rhit, std::vector<double>& zhit, 
                                        std::vector<double>& dphihit, std::vector<double>& drhit, 
	                                std::vector<double>& ehitstrip, std::vector<double>& dehitphi,
                                        const TransientTrackingRecHit::ConstRecHitPointer& pRecHit, const TransientTrackingRecHit::ConstRecHitPointer& nRecHit, 
			                const TrajectoryStateOnSurface& nTsos, double&, double&, int&) const;
						 
  TrajectoryStateOnSurface updateEndcap(std::vector<double>& rhit, std::vector<double>& zhit, 
                                        std::vector<double>& dphihit, std::vector<double>& drhit, 
	                                std::vector<double>& ehitstrip, std::vector<double>& dehitphi,
                                        const TransientTrackingRecHit::ConstRecHitPointer& pRecHit, const TransientTrackingRecHit::ConstRecHitPointer& nRecHit, 
			                const TrajectoryStateOnSurface& nTsos, double&, double&, int& ) const;

private:

  double findPhiInVertex(const FreeTrajectoryState& fts, const double& rc, const GeomDetEnumerators::Location) const;

  bool linefit1(const std::vector <double>& x, const std::vector <double>& y, const std::vector <double>& err,
                double& a, double& chi ) const;
  bool linefit2(const std::vector <double>& x, const std::vector <double>& y, const std::vector <double>& err,
                double& a, double& b, double& chi ) const;
  double zvert;
  double          thePhiWin;
  double          theZWin;
  const MagneticField * field;		
};

#endif
