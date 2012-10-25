/** \file
 *
 * $Date: 2012/08/06 08:35:23 $
 * $Revision: 1.49 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 * \       A.Meneguzzo - Padova University  <anna.meneguzzo@pd.infn.it>
 * \       M.Pelliccioni - INFN TO <pellicci@cern.ch>
 * \       M.Meneghelli - INFN BO <marco.meneghelli@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"

/* Collaborating Class Header */

//mene
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "RecoLocalMuon/DTSegment/src/DTLinearFit.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* C++ Headers */
#include <string>

using namespace std;
using namespace edm;

/* ====================================================================== */

/// Constructor
DTSegmentUpdator::DTSegmentUpdator(const ParameterSet& config) :
  theFitter(new DTLinearFit()) ,
  vdrift_4parfit(config.getParameter<bool>("performT0_vdriftSegCorrection")),
  T0_hit_resolution(config.getParameter<double>("hit_afterT0_resolution")),
  perform_delta_rejecting(config.getParameter<bool>("perform_delta_rejecting")),
  debug(config.getUntrackedParameter<bool>("debug",false)) 
{  
  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = DTRecHitAlgoFactory::get()->create(theAlgoName, 
                                               config.getParameter<ParameterSet>("recAlgoConfig"));

  if(debug)
    cout << "[DTSegmentUpdator] Constructor called" << endl;
}

/// Destructor
DTSegmentUpdator::~DTSegmentUpdator() {
  delete theFitter;
}

/* Operations */ 

void DTSegmentUpdator::setES(const EventSetup& setup){
  setup.get<MuonGeometryRecord>().get(theGeom);
  theAlgo->setES(setup);
}

void DTSegmentUpdator::update(DTRecSegment4D* seg, const bool calcT0) const {

  if(debug) cout << "[DTSegmentUpdator] Starting to update the segment" << endl;

  const bool hasPhi = seg->hasPhi();
  const bool hasZed = seg->hasZed();

  //reject the bad hits (due to delta rays)
  if(perform_delta_rejecting && hasPhi) rejectBadHits(seg->phiSegment());

  int step = (hasPhi && hasZed) ? 3 : 2;
  if(calcT0) step = 4;

  GlobalPoint  pos = theGeom->idToDet(seg->geographicalId())->toGlobal(seg->localPosition());
  GlobalVector dir = theGeom->idToDet(seg->geographicalId())->toGlobal(seg->localDirection());

  if(calcT0) calculateT0corr(seg);

  if(hasPhi) updateHits(seg->phiSegment(),pos,dir,step);
  if(hasZed) updateHits(seg->zSegment()  ,pos,dir,step);

  fit(seg);
}

void DTSegmentUpdator::update(DTRecSegment2D* seg) const {
  GlobalPoint pos = (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localPosition());
  GlobalVector dir = (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localDirection());

  updateHits(seg,pos,dir);
  fit(seg);
}

void DTSegmentUpdator::fit(DTRecSegment4D* seg)  const {
  // after the update must refit the segments
  if(seg->hasPhi()) fit(seg->phiSegment());
  if(seg->hasZed()) fit(seg->zSegment());

  const DTChamber* theChamber = theGeom->chamber(seg->chamberId());

  if(seg->hasPhi() && seg->hasZed() ) {

    DTChamberRecSegment2D *segPhi=seg->phiSegment();
    DTSLRecSegment2D *segZed=seg->zSegment();

    // NB Phi seg is already in chamber ref
    LocalPoint posPhiInCh = segPhi->localPosition();
    LocalVector dirPhiInCh= segPhi->localDirection();

    // Zed seg is in SL one
    const DTSuperLayer* zSL = theChamber->superLayer(segZed->superLayerId());
    LocalPoint zPos(segZed->localPosition().x(), 
		    (zSL->toLocal(theChamber->toGlobal(segPhi->localPosition()))).y(),
		    0.);

    LocalPoint posZInCh = theChamber->toLocal(zSL->toGlobal(zPos));

    LocalVector dirZInCh = theChamber->toLocal(zSL->toGlobal(segZed->localDirection()));

    LocalPoint posZAt0 = posZInCh + dirZInCh*(-posZInCh.z())/cos(dirZInCh.theta());

    // given the actual definition of chamber refFrame, (with z poiniting to IP),
    // the zed component of direction is negative.
    LocalVector dir=LocalVector(dirPhiInCh.x()/fabs(dirPhiInCh.z()),dirZInCh.y()/fabs(dirZInCh.z()),-1.);

    seg->setPosition(LocalPoint(posPhiInCh.x(),posZAt0.y(),0.));
    seg->setDirection(dir.unit());

    AlgebraicSymMatrix mat(4);

    // set cov matrix
    mat[0][0] = segPhi->parametersError()[0][0]; //sigma (dx/dz)
    mat[0][2] = segPhi->parametersError()[0][1]; //cov(dx/dz,x)
    mat[2][2] = segPhi->parametersError()[1][1]; //sigma (x)

    seg->setCovMatrix(mat);
    seg->setCovMatrixForZed(posZInCh);

  }

  else if (seg->hasPhi()) {
    DTChamberRecSegment2D *segPhi=seg->phiSegment();

    seg->setPosition(segPhi->localPosition());
    seg->setDirection(segPhi->localDirection());

    AlgebraicSymMatrix mat(4);
    // set cov matrix
    mat[0][0] = segPhi->parametersError()[0][0]; //sigma (dx/dz)
    mat[0][2] = segPhi->parametersError()[0][1]; //cov(dx/dz,x)
    mat[2][2] = segPhi->parametersError()[1][1]; //sigma (x)

    seg->setCovMatrix(mat);
  }

  else if (seg->hasZed()) {
    DTSLRecSegment2D *segZed = seg->zSegment();

    // Zed seg is in SL one
    GlobalPoint glbPosZ = ( theGeom->superLayer(segZed->superLayerId()) )->toGlobal(segZed->localPosition());
    LocalPoint posZInCh = ( theGeom->chamber(segZed->superLayerId().chamberId()) )->toLocal(glbPosZ);

    GlobalVector glbDirZ = (theGeom->superLayer(segZed->superLayerId()) )->toGlobal(segZed->localDirection());
    LocalVector dirZInCh = (theGeom->chamber(segZed->superLayerId().chamberId()) )->toLocal(glbDirZ);

    LocalPoint posZAt0 = posZInCh+
      dirZInCh*(-posZInCh.z())/cos(dirZInCh.theta());

    seg->setPosition(posZAt0);
    seg->setDirection(dirZInCh);

    AlgebraicSymMatrix mat(4);
    // set cov matrix
    seg->setCovMatrix(mat);
    seg->setCovMatrixForZed(posZInCh);
  }
}

bool DTSegmentUpdator::fit(DTSegmentCand* seg) const {
  if (!seg->good()) return false;

  vector<float> x;
  vector<float> y;
  vector<float> sigy;

  DTSegmentCand::AssPointCont hits=seg->hits();
  for (DTSegmentCand::AssPointCont::const_iterator iter=hits.begin();
       iter!=hits.end(); ++iter) {
    LocalPoint pos = (*iter).first->localPosition((*iter).second);
    x.push_back(pos.z()); 
    y.push_back(pos.x());
    sigy.push_back(sqrt((*iter).first->localPositionError().xx()));
  }

  LocalPoint pos;
  LocalVector dir;
  AlgebraicSymMatrix covMat(2);

  double chi2 = 0.;
  fit(x,y,sigy,pos,dir,covMat,chi2);

  seg->setPosition(pos);
  seg->setDirection(dir);

  //cout << "pos " << segPosition<< endl;
  //cout << "dir " << segDirection<< endl;

  seg->setCovMatrix(covMat);
  // cout << "Mat " << covMat << endl;

  seg->setChi2(chi2);
  return true;
}

void DTSegmentUpdator::fit(DTRecSegment2D* seg) const {
  // WARNING: since this method is called both with a 2D and a 2DPhi as argument
  // seg->geographicalId() can be a superLayerId or a chamberId 

  vector<float> x;
  vector<float> y;
  vector<float> sigy;

  vector<DTRecHit1D> hits=seg->specificRecHits();
  for (vector<DTRecHit1D>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {

    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = ( theGeom->layer( hit->wireId().layerId() ) )->toGlobal(hit->localPosition());
    LocalPoint pos = ( theGeom->idToDet(seg->geographicalId()) )->toLocal(glbPos);

    x.push_back(pos.z()); 
    y.push_back(pos.x());

    // Get local error in SL frame
    //RB: is it right in this way? 
    ErrorFrameTransformer tran;
    GlobalError glbErr =
      tran.transform( hit->localPositionError(),(theGeom->layer( hit->wireId().layerId() ))->surface());
    LocalError slErr =
      tran.transform( glbErr, (theGeom->idToDet(seg->geographicalId()))->surface());

    sigy.push_back(sqrt(slErr.xx()));
  }

  LocalPoint pos;
  LocalVector dir;
  AlgebraicSymMatrix covMat(2);
  double chi2 = 0.;

  fit(x,y,sigy,pos,dir,covMat,chi2);

  seg->setPosition(pos);
  seg->setDirection(dir);

  //cout << "pos " << segPosition << endl;
  //cout << "dir " << segDirection << endl;

  seg->setCovMatrix(covMat);
  // cout << "Mat " << mat << endl;

  seg->setChi2(chi2);
}

void DTSegmentUpdator::fit(const vector<float>& x,
                           const vector<float>& y, 
                           const vector<float>& sigy,
                           LocalPoint& pos,
                           LocalVector& dir,
                           AlgebraicSymMatrix& covMatrix,
                           double& chi2)  const {

  float slope     = 0.;
  float intercept = 0.;
  float covss     = 0.;
  float covii     = 0.;
  float covsi     = 0.;

  // do the fit
  theFitter->fit(x,y,x.size(),sigy,slope,intercept,covss,covii,covsi);
  // cout << "slope " << slope << endl;
  // cout << "intercept " << intercept << endl;

  // intercept is the x() in chamber frame when the segment cross the chamber
  // plane (at z()=0), the y() is not measured, so let's put the center of the
  // chamber.
  pos = LocalPoint(intercept,0.,0.);

  //  slope is dx()/dz(), while dy()/dz() is by definition 0, finally I want the
  //  segment to point outward, so opposite to local z
  dir = LocalVector(-slope,0.,-1.).unit();

  covMatrix = AlgebraicSymMatrix(2);
  covMatrix[0][0] = covss; // this is var(dy/dz)
  covMatrix[1][1] = covii; // this is var(y)
  covMatrix[1][0] = covsi; // this is cov(dy/dz,y)

  /* Calculate chi2. */
  chi2 = 0.;
  for(unsigned int i=0; i<x.size() ; ++i) {
    double resid= y[i] - (intercept + slope*x[i]);
    chi2 += (resid/sigy[i])*(resid/sigy[i]);
  }
}

// The GlobalPoint and the GlobalVector can be either the glb position and the direction
// of the 2D-segment itself or the glb position and direction of the 4D segment
void DTSegmentUpdator::updateHits(DTRecSegment2D* seg, GlobalPoint &gpos,
                                  GlobalVector &gdir, const int step) const{

  // it is not necessary to have DTRecHit1D* to modify the obj in the container
  // but I have to be carefully, since I cannot make a copy before the iteration!

  vector<DTRecHit1D> toBeUpdatedRecHits = seg->specificRecHits();
  vector<DTRecHit1D> updatedRecHits;

  for (vector<DTRecHit1D>::iterator hit= toBeUpdatedRecHits.begin(); 
       hit!=toBeUpdatedRecHits.end(); ++hit) {

    const DTLayer* layer = theGeom->layer( hit->wireId().layerId() );

    LocalPoint segPos=layer->toLocal(gpos);
    LocalVector segDir=layer->toLocal(gdir);

    // define impact angle needed by the step 2
    const float angle = atan(segDir.x()/-segDir.z());

    // define the local position (extr.) of the segment. Needed by the third step 
    LocalPoint segPosAtLayer=segPos+segDir*(-segPos.z())/cos(segDir.theta());

    DTRecHit1D newHit1D = (*hit);

    bool ok = true;

    if (step == 2) {
      ok = theAlgo->compute(layer,*hit,angle,newHit1D);

    } else if (step == 3) {

      LocalPoint hitPos(hit->localPosition().x(),+segPosAtLayer.y(),0.);

      GlobalPoint glbpos= theGeom->layer( hit->wireId().layerId() )->toGlobal(hitPos);

      newHit1D.setPosition(hitPos);

      ok = theAlgo->compute(layer,*hit,angle,glbpos,newHit1D);

    } else if (step == 4) {

      //const double vminf = seg->vDrift();   //  vdrift correction are recorded in the segment    
      double vminf =0.;
      if(vdrift_4parfit) vminf = seg->vDrift();   // use vdrift recorded in the segment only if vdrift_4parfit=True

      double cminf = 0.;
      if(seg->ist0Valid()) cminf = - seg->t0()*0.00543;

      //cout << "In updateHits: t0 = " << seg->t0() << endl;
      //cout << "In updateHits: vminf = " << vminf << endl;
      //cout << "In updateHits: cminf = " << cminf << endl;

      const float xwire = layer->specificTopology().wirePosition(hit->wireId().wire());
      const float distance = fabs(hit->localPosition().x() - xwire);

      const int ilc = ( hit->lrSide() == DTEnums::Left ) ? 1 : -1;

      const double dy_corr = (vminf*ilc*distance-cminf*ilc ); 

      LocalPoint point(hit->localPosition().x() + dy_corr, +segPosAtLayer.y(), 0.);

      LocalError error(T0_hit_resolution*T0_hit_resolution,0.,0.);

      //newHit1D.setPositionAndError(point, error);
      newHit1D.setPosition(point);

      //FIXME: check that the hit is still inside the cell
      ok = true;

    } else throw cms::Exception("DTSegmentUpdator")<<" updateHits called with wrong step " << endl;

    if (ok) updatedRecHits.push_back(newHit1D);
    else {
      LogError("DTSegmentUpdator")<<"DTSegmentUpdator::updateHits failed update" << endl;
      throw cms::Exception("DTSegmentUpdator")<<"updateHits failed update"<<endl;
    }

  }
  seg->update(updatedRecHits);
}

void DTSegmentUpdator::rejectBadHits(DTChamberRecSegment2D* phiSeg) const {

  vector<float> x;
  vector<float> y;
  
  if(debug) cout << " Inside the segment updator, now loop on hits:   ( x == z_loc , y == x_loc) " << endl;
 
  vector<DTRecHit1D> hits = phiSeg->specificRecHits();
  for (vector<DTRecHit1D>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {

    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = ( theGeom->layer( hit->wireId().layerId() ) )->toGlobal(hit->localPosition());
    LocalPoint pos = ( theGeom->idToDet(phiSeg->geographicalId()) )->toLocal(glbPos);

    x.push_back(pos.z()); 
    y.push_back(pos.x());
  }

  if(debug){
    cout << " end of segment! " << endl;
    cout << " size = Number of Hits: " << x.size() << "  " << y.size() << endl;
  }
  
  // Perform the 2 par fit:
  float par[2]={0.,0.}; // q , m

  //variables to perform the fit:
  float Sx = 0.;
  float Sy = 0.;
  float Sx2 = 0.;
  float Sy2 = 0.;
  float Sxy = 0.;

  size_t N =  x.size();
	
  for(size_t i = 0; i < N;++i){
    Sx += x.at(i);
    Sy += y.at(i);
    Sx2 += x.at(i)*x.at(i);
    Sy2 += y.at(i)*y.at(i);
    Sxy += x.at(i)*y.at(i);
		
  }
	
  const float delta = N*Sx2 - Sx*Sx;
  par[0] = ( Sx2*Sy - Sx*Sxy )/delta;
  par[1] = ( N*Sxy - Sx*Sy )/delta;

  if(debug) cout << "fit 2 parameters done ----> par0: "<< par[0] << "  par1: "<< par[1] << endl;

  // Calc residuals:
  float residuals[N];
	
  for(size_t i = 0; i < N;++i)
    residuals[i] = 0;
	
  for(size_t i = 0; i < N;++i)		
    residuals[i] = y.at(i) - par[1]*x.at(i) - par[0];
	
  if(debug) cout << " Residuals computed! "<<  endl;
		
		
  // Perform bad hit rejecting -- update hits
  vector<DTRecHit1D> updatedRecHits;
	
  float mean_residual = 0.; //mean of the absolute values of residuals
	
  for (size_t i = 0; i < N; ++i)
    mean_residual += fabs(residuals[i]);
	
  mean_residual = mean_residual/(N - 2);	
	
  if(debug) cout << " mean_residual: "<< mean_residual << endl;
	
  int i = 0;
	
  for (vector<DTRecHit1D>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {
		
    DTRecHit1D newHit1D = (*hit);

    if(fabs(residuals[i])/mean_residual < 1.5){
					
      updatedRecHits.push_back(newHit1D);
      if(debug) cout << " accepted "<< i+1 << "th hit" <<"  Irej: " << fabs(residuals[i])/mean_residual << endl;
      ++i;
    }
    else {
      if(debug) cout << " rejected "<< i+1 << "th hit" <<"  Irej: " << fabs(residuals[i])/mean_residual << endl;
      ++i;
      continue;
    }
  }
	
  phiSeg->update(updatedRecHits);	
  
  //final check!
  if(debug){ 
  
    vector<float> x_upd;
    vector<float> y_upd;
 
    cout << " Check the update action: " << endl;
 
    vector<DTRecHit1D> hits_upd = phiSeg->specificRecHits();
    for (vector<DTRecHit1D>::const_iterator hit=hits_upd.begin();
	 hit!=hits_upd.end(); ++hit) {

      // I have to get the hits position (the hit is in the layer rf) in SL frame...
      GlobalPoint glbPos = ( theGeom->layer( hit->wireId().layerId() ) )->toGlobal(hit->localPosition());
      LocalPoint pos = ( theGeom->idToDet(phiSeg->geographicalId()) )->toLocal(glbPos);

      x_upd.push_back(pos.z()); 
      y_upd.push_back(pos.x());

      cout << " x_upd: "<< pos.z() << "  y_upd: "<< pos.x() << endl;


    }
  
    cout << " end of segment! " << endl;
    cout << " size = Number of Hits: " << x_upd.size() << "  " << y_upd.size() << endl;
    
  }// end debug
  
  return;
} //end DTSegmentUpdator::rejectBadHits

void DTSegmentUpdator::calculateT0corr(DTRecSegment4D* seg) const {
  if(seg->hasPhi()) calculateT0corr(seg->phiSegment());
  if(seg->hasZed()) calculateT0corr(seg->zSegment());
}

void DTSegmentUpdator::calculateT0corr(DTRecSegment2D* seg) const {
  // WARNING: since this method is called both with a 2D and a 2DPhi as argument
  // seg->geographicalId() can be a superLayerId or a chamberId 

  vector<double> d_drift;
  vector<float> x;
  vector<float> y;
  vector<int> lc;

  vector<DTRecHit1D> hits=seg->specificRecHits();

  DTWireId wireId;
  int nptfit = 0;

  for (vector<DTRecHit1D>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {

    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = ( theGeom->layer( hit->wireId().layerId() ) )->toGlobal(hit->localPosition());
    LocalPoint pos = ( theGeom->idToDet(seg->geographicalId()) )->toLocal(glbPos);

    const DTLayer* layer = theGeom->layer( hit->wireId().layerId() );
    float xwire = layer->specificTopology().wirePosition(hit->wireId().wire());
    float distance = fabs(hit->localPosition().x() - xwire);

    int ilc = ( hit->lrSide() == DTEnums::Left ) ? 1 : -1;

    nptfit++;
    x.push_back(pos.z()); 
    y.push_back(pos.x());
    lc.push_back(ilc);
    d_drift.push_back(distance);

    // cout << " d_drift "<<distance  <<" npt= " <<npt<<endl;
  }

  double chi2fit = 0.;
  float cminf    = 0.;
  double vminf   = 0.;

  if ( nptfit > 2 ) {
    //NB chi2fit is normalized
    Fit4Var(x,y,lc,d_drift,nptfit,cminf,vminf,chi2fit);

    double t0cor = -999.;
    if(cminf > -998.) t0cor = - cminf/0.00543 ; // in ns

    //cout << "In calculateT0corr: t0 = " << t0cor << endl;
    //cout << "In calculateT0corr: vminf = " << vminf << endl;
    //cout << "In calculateT0corr: cminf = " << cminf << endl;
    //cout << "In calculateT0corr: chi2 = " << chi2fit << endl;

    seg->setT0(t0cor);          // time  and
    seg->setVdrift(vminf);   //  vdrift correction are recorded in the segment    
  }
}


void DTSegmentUpdator::Fit4Var(const vector<float>& xfit,
                               const vector<float>& yfit,
                               const vector<int>& lfit,
                               const vector<double>& tfit,
                               const int nptfit,
                               float& cminf,
                               double& vminf,
                               double& chi2fit) const { 

  const double sigma = 0.0295;// errors can be inserted .just load them/that is the usual TB resolution value for DT chambers 
  double aminf = 0.;
  double bminf = 0.;
  int nppar = 0;
  double sx = 0.;
  double  sx2 = 0.;
  double sy = 0.;
  double sxy = 0.;
  double sl = 0.;
  double sl2 = 0.;
  double sly = 0.;
  double slx = 0.;
  double st = 0.;
  double st2 = 0.;
  double slt = 0.;
  double sltx = 0.;
  double slty = 0.;
  double chi2fitN2 = -1. ;
  double chi2fit3 = -1.;
  double chi2fitN3 = -1. ;
  double chi2fitN4 = -1.;
  float bminf3 = bminf;
  float aminf3 = aminf;
  float cminf3 = cminf;
  int nppar2 = 0;
  int nppar3 = 0;
  int nppar4 = 0;

  cminf = -999.;
  vminf = 0.;

  for (int j=0; j<nptfit; j++){
    sx  = sx + xfit[j];       
    sy  = sy + yfit[j];
    sx2 = sx2 + xfit[j]*xfit[j];
    sxy = sxy + xfit[j]*yfit[j];
    sl  = sl + lfit[j];       
    sl2 = sl2 + lfit[j]*lfit[j];
    sly = sly + lfit[j]*yfit[j];
    slx = slx + lfit[j]*xfit[j];
    st = st + tfit[j];
    st2 = st2 + tfit[j] * tfit[j];
    slt = slt + lfit[j] * tfit[j];
    sltx = sltx + lfit[j] * tfit[j]*xfit[j];
    slty = slty + lfit[j] * tfit[j]*yfit[j];

  } //end loop

  const double delta = nptfit*sx2 - sx*sx;

  double a = 0.;
  double b = 0.;	       

  if (delta!=0){   //
    a = (sx2*sy - sx*sxy)/delta;
    b = (nptfit*sxy - sx*sy)/delta;

    //  cout << " NPAR=2 : slope = "<<b<< "    intercept = "<<a <<endl;
    for (int j=0; j<nptfit; j++){
      const double ypred = a + b*xfit[j];
      const double dy = (yfit[j] - ypred)/sigma;
      chi2fit = chi2fit + dy*dy;
    } //end loop chi2
  }

  bminf = b;
  aminf = a;

  nppar = 2; 
  nppar2 = nppar; 

  chi2fitN2 = chi2fit/(nptfit-2);

  // cout << "dt0 = 0chi2fit = " << chi2fit << "  slope = "<<b<<endl;

  if (nptfit >= 3) {

    const double d1 = sy;
    const double d2 = sxy;
    const double d3 = sly;
    const double c1 = sl;
    const double c2 = slx;
    const double c3 = sl2;
    const double b1 = sx;
    const double b2 = sx2;
    const double b3 = slx;
    const double a1 = nptfit;
    const double a2 = sx;
    const double a3 = sl;

    //these parameters are not used in the 4-variables fit
    const double b4 = b2*a1-b1*a2;
    const double c4 = c2*a1-c1*a2;
    const double d4 = d2*a1-d1*a2;
    const double b5 = a1*b3-a3*b1;
    const double c5 = a1*c3-a3*c1;
    const double d5 = a1*d3-d1*a3;
    const double a6 = slt;
    const double b6 = sltx;
    const double c6 = st;
    const double v6 = st2;	
    const double d6 = slty;

    if (((c5*b4-c4*b5)*b4*a1)!=0) {
      nppar = 3;
      chi2fit = 0.;
      cminf = (d5*b4-d4*b5)/(c5*b4-c4*b5);
      bminf = d4/b4 -cminf *c4/b4;
      aminf = (d1/a1 -cminf*c1/a1 -bminf*b1/a1);

      for (int j=0; j<nptfit; j++){
        const double ypred = aminf + bminf*xfit[j];
        const double dy = (yfit[j]-cminf*lfit[j] - ypred)/sigma;
        chi2fit = chi2fit + dy*dy;

      } //end loop chi2
      chi2fit3 = chi2fit;
      if (nptfit>3)
        chi2fitN3 = chi2fit /(nptfit-3);

    }
    else {
      cminf = -999.;
      bminf = b;
      aminf = a;
      chi2fit3 = chi2fit;
      chi2fitN3 = chi2fit /(nptfit-2);
    }

    bminf3 = bminf;
    aminf3 = aminf;
    cminf3 = cminf;
    nppar3 = nppar;

    if (debug) {
      cout << "dt0= 0 : slope 2 = " << b << " pos in  = " << a << " chi2fitN2 = " << chi2fitN2
	   << " nppar = " << nppar2 << " nptfit = " << nptfit << endl;
      cout << "dt0 = 0 : slope 3 = " << bminf << " pos out = " << aminf << " chi2fitN3 = "
	   << chi2fitN3 << " nppar = " << nppar3 << " T0_ev ns = " << cminf/0.00543 << endl;
    } 

    //***********************************
    //     cout << " vdrift_4parfit "<< vdrift_4parfit<<endl;
    if( nptfit>=5) { 
      const double det = (a1*a1*(b2*v6 - b6*b6) - a1*(a2*a2*v6 - 2*a2*a6*b6 + a6*a6*b2 + b2*c6*c6 + b3*(b3*v6 - 2*b6*c6))
			  + a2*a2*c6*c6 + 2*a2*(a3*(b3*v6 - b6*c6) - a6*b3*c6) + a3*a3*(b6*b6 - b2*v6)
			  + a6*(2*a3*(b2*c6 - b3*b6) + a6*b3*b3)); 

      // the dv/vdrift correction may be computed  under vdrift_4parfit request;
      if (det != 0) { 
        nppar = 4;
        chi2fit = 0.;
        // computation of   a, b, c e v
        aminf = (a1*(a2*(b6*d6 - v6*d2) + a6*(b6*d2 - b2*d6) + d1*(b2*v6 - b6*b6)) - a2*(b3*(c6*d6 - v6*d3)
                 + c6*(b6*d3 - c6*d2)) + a3*(b2*(c6*d6 - v6*d3) + b3*(v6*d2 - b6*d6) + b6*(b6*d3 - c6*d2))
                 + a6*(b2*c6*d3 + b3*(b3*d6 - b6*d3 - c6*d2)) - d1*(b2*c6*c6 + b3*(b3*v6 - 2*b6*c6)))/det;
        bminf = - (a1*a1*(b6*d6 - v6*d2) - a1*(a2*(a6*d6 - v6*d1) - a6*a6*d2 + a6*b6*d1 + b3*(c6*d6 - v6*d3)
                 + c6*(b6*d3 - c6*d2)) + a2*(a3*(c6*d6 - v6*d3) + c6*(a6*d3 - c6*d1)) + a3*a3*(v6*d2 - b6*d6)
                 + a3*(a6*(b3*d6 + b6*d3 - 2*c6*d2) - d1*(b3*v6 - b6*c6)) - a6*b3*(a6*d3 - c6*d1))/det;
        cminf = -(a1*(b2*(c6*d6 - v6*d3) + b3*(v6*d2 - b6*d6) + b6*(b6*d3 - c6*d2)) + a2*a2*(v6*d3 - c6*d6)
                 + a2*(a3*(b6*d6 - v6*d2) + a6*(b3*d6 - 2*b6*d3 + c6*d2) - d1*(b3*v6 - b6*c6))
                 + a3*(d1*(b2*v6 - b6*b6) - a6*(b2*d6 - b6*d2)) + a6*(a6*(b2*d3 - b3*d2) - d1*(b2*c6 - b3*b6)))/det;
        vminf = - (a1*a1*(b2*d6 - b6*d2) - a1*(a2*a2*d6 - a2*(a6*d2 + b6*d1) + a6*b2*d1 + b2*c6*d3
                 + b3*(b3*d6 - b6*d3 - c6*d2)) + a2*a2*c6*d3 + a2*(a3*(2*b3*d6 - b6*d3 - c6*d2) - b3*(a6*d3 + c6*d1))
                 + a3*a3*(b6*d2 - b2*d6) + a3*(a6*(b2*d3 - b3*d2) + d1*(b2*c6 - b3*b6)) + a6*b3*b3*d1)/det;

        //  chi 2
        for (int j=0; j<nptfit; j++) {
          const double ypred = aminf + bminf*xfit[j];
          const double dy = (yfit[j]+vminf*lfit[j]*tfit[j]-cminf*lfit[j] -ypred)/sigma; 
          chi2fit = chi2fit + dy*dy;

        } //end loop chi2
        if (nptfit<=nppar){ 
          chi2fitN4=-1;
          //		cout << "nptfit " << nptfit << " nppar " << nppar << endl;
        }
        else{
          chi2fitN4= chi2fit / (nptfit-nppar); 
        }
      }
      else {
        vminf = 0.;

        if (nptfit <= nppar) chi2fitN4=-1;
        else chi2fitN4	= chi2fit / (nptfit-nppar); 
      }

      if (fabs(vminf) >= 0.29) {
        // for safety and for code construction..dont accept correction on dv/vdrift greater then 0.09
        vminf = 0.;
        cminf = cminf3;
        aminf = aminf3;
        bminf = bminf3;
        nppar = 3;
        chi2fit = chi2fit3;
      }

    }  //end if vdrift

     if(!vdrift_4parfit){         //if not required explicitly leave the t0 and track step as at step 3
                                  // just update vdrift value vmin for storing in the segments for monitoring
       cminf = cminf3;
       aminf = aminf3;
       bminf = bminf3;
       nppar = 3;
       chi2fit = chi2fit3;
     }

    nppar4 = nppar;

  }  //end nptfit >=3

  if (debug) {
    cout << "   dt0= 0 : slope 4  = " << bminf << " pos out = " << aminf <<" chi2fitN4 = " << chi2fitN4
	 << "  nppar= " << nppar4 << " T0_ev ns= " << cminf/0.00543 <<" delta v = " << vminf <<endl;
    cout << nptfit << " nptfit " << " end  chi2fit = " << chi2fit/ (nptfit-nppar ) << " T0_ev ns= " << cminf/0.00543 << " delta v = " << vminf <<endl;
  }

  if ( fabs(vminf) >= 0.09 && debug ) {  //checks only vdrift less then 10 % accepted
    cout << "vminf gt 0.09 det=  " << endl;
    cout << "dt0= 0 : slope 4 = "<< bminf << " pos out = " << aminf << " chi2fitN4 = " << chi2fitN4
	 << " T0_ev ns = " << cminf/0.00543 << " delta v = "<< vminf << endl;
    cout << "dt0 = 0 : slope 2 = "<< b << " pos in = " << a <<" chi2fitN2 = " << chi2fitN2
	 << " nppar = " << nppar-1 << " nptfit = " << nptfit <<endl;
    cout << "dt0 = 0 : slope 3 = " << bminf << " pos out = " << aminf << " chi2fitN3 = "
	 << chi2fitN3 << " T0_ev ns = " << cminf/0.00543 << endl;
    cout << nptfit   <<" nptfit "<< "   end  chi2fit = " << chi2fit << "T0_ev ns= " << cminf/0.00543 << "delta v = "<< vminf <<endl;        
  }

  if (nptfit != nppar) chi2fit = chi2fit / (nptfit-nppar);
}
