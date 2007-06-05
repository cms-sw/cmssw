#ifndef _TRACKER_HICMUUPDATOR_H_
#define _TRACKER_HICMUUPDATOR_H_
#include "Utilities/Configuration/interface/Architecture.h"

#include "CommonDet/PatternPrimitives/interface/TrajectoryStateOnSurface.h"
#include "CommonDet/BasicDet/interface/SimHit.h"
#include "CommonDet/BasicDet/interface/SimDet.h"
#include "CommonDet/BasicDet/interface/DetUnit.h"
#include "CommonDet/BasicDet/interface/DetType.h"
#include "CommonDet/BasicDet/interface/RecHit.h"
#include "Tracker/HICPattern/interface/HICConst.h"
#include "CommonDet/DetLayout/interface/DetLayer.h"
#include "CommonReco/PatternTools/interface/Trajectory.h"

#include <CLHEP/Geometry/Point3D.h>
#include <vector>

class HICMuonUpdator{

public:
  HICMuonUpdator(double&la1,double&la2){HICConst hc; zvert=hc.zvert;
                                        thePhiWin=la1; theZWin=la2;};
  virtual ~HICMuonUpdator(){}
  TrajectoryStateOnSurface update(const Trajectory& mt,
                                  const TrajectoryStateOnSurface&, 
                                  const RecHit&, 
				  const DetLayer*, 
				  double&, double&) const;
      

  TrajectoryStateOnSurface updateBarrel(vector<double>& rhit, vector<double>& zhit, 
                                        vector<double>& dphihit, vector<double>& drhit, 
	                                vector<double>& ehitstrip, vector<double>& dehitphi,
                                        const RecHit& pRecHit, const RecHit& nRecHit, 
			                const TrajectoryStateOnSurface& nTsos, double&, double&, int&) const;
						 
  TrajectoryStateOnSurface updateEndcap(vector<double>& rhit, vector<double>& zhit, 
                                        vector<double>& dphihit, vector<double>& drhit, 
	                                vector<double>& ehitstrip, vector<double>& dehitphi,
                                        const RecHit& pRecHit, const RecHit& nRecHit, 
			                const TrajectoryStateOnSurface& nTsos, double&, double&, int& ) const;

private:

  double findPhiInVertex(const FreeTrajectoryState& fts, const double& rc, const Det* det) const;

  bool linefit1(const vector <double>& x, const vector <double>& y, const vector <double>& err,
                double& a, double& chi ) const;
  bool linefit2(const vector <double>& x, const vector <double>& y, const vector <double>& err,
                double& a, double& b, double& chi ) const;
  double zvert;
  double          thePhiWin;
  double          theZWin;		
};

#endif
