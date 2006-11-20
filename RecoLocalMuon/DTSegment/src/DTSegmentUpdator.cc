/** \file
 *
 * $Date: 2006/05/26 10:51:09 $
 * $Revision: 1.15 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"

/* Collaborating Class Header */

#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"

#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "RecoLocalMuon/DTSegment/interface/DTLinearFit.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
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
  theFitter(new DTLinearFit()),
  T0_seg(config.getParameter<bool>("performT0SegCorrection")),
  T0_seg_debug(config.getUntrackedParameter<bool>("T0SegCorrectionDebug",false))
 {  
  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = DTRecHitAlgoFactory::get()->create(theAlgoName, 
					       config.getParameter<ParameterSet>("recAlgoConfig"));
}

/// Destructor

DTSegmentUpdator::~DTSegmentUpdator() {
  delete theFitter;
}

/* Operations */ 

void DTSegmentUpdator::setES(const edm::EventSetup& setup){
  setup.get<MuonGeometryRecord>().get(theGeom);
  theAlgo->setES(setup);
}

void DTSegmentUpdator::update(DTRecSegment4D* seg)  { 
//  if (T0_seg && T0_seg_debug) cout << "  before  fitT0_seg(seg) in Update 4D " << endl;

  bool hasPhi=seg->hasPhi();
  bool hasZed=seg->hasZed();
  int  step=0;
  float t0cor=0.;
  float t0corphi=0.;
  float t0corz=0.;
  
  if (hasPhi && hasZed) step=3;
  else step=2;

  GlobalPoint pos =  (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localPosition());
  GlobalVector dir = (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localDirection());

  if (hasPhi) {
   DTChamberRecSegment2D *segPhi=seg->phiSegment();
  float t0cor_seg = segPhi->theT0;  // get the current value of the t0seg
    t0corphi = t0cor_seg;
    if (T0_seg) {
      if (t0cor_seg == 0 ) fitT0_seg(seg->phiSegment(),t0cor);
      t0cor_seg = segPhi->theT0;  // get the current value of the t0seg
      t0corphi=t0cor_seg;
      if (T0_seg && T0_seg_debug)  cout << " After  fitT0_seg(seg) in Update 4D : Phi seg !! " << endl;
    }	
     if (T0_seg && T0_seg_debug) cout << " After   in Update 4D : Phi seg !! t0corphi = " << t0corphi  <<endl;
     
     updateHits(seg->phiSegment(),
                        pos,dir,step);
 }

  if(hasZed){
    DTSLRecSegment2D *segZed=seg->zSegment();
    float t0cor_seg = segZed->theT0;  // get the current value of the t0seg
    t0corz = t0cor_seg;
    if (T0_seg) {
       if (t0cor_seg == 0 ) fitT0_seg(seg->zSegment(),t0cor);
       t0cor_seg = segZed->theT0;  // get the current value of the t0seg
       t0corz = t0cor_seg;
    if (T0_seg && T0_seg_debug) cout << " After  fitT0_seg(seg) in Update 4D :z seg   !! " << endl;
    }	

   if (T0_seg && T0_seg_debug) cout << " After   in Update 4D : z seg !! t0cor = " << t0corz <<endl; 

   updateHits(seg->zSegment(),
                        pos,dir,step);
} 
   if (T0_seg && T0_seg_debug) cout << " After  fitT0_seg(seg) in Update 4D : z seg !! t0corphi = " << t0corphi <<" t0corz = " << t0corz <<endl;
   fit(seg);
}

void DTSegmentUpdator::update(DTRecSegment2D* seg)  {
 
  float t0cor=0;
  float t0cor_seg = seg->theT0;

 if (T0_seg && T0_seg_debug)   cout << "  entered in update(DTRecSegment2D* seg)  !! t0cor = "<< t0cor  << endl;
  if (T0_seg) {
    //aaaaa 
    if (t0cor_seg == 0. )fitT0_seg(seg,t0cor);
    if (T0_seg && T0_seg_debug) cout << " After  fitT0_seg(seg) in update(DTRecSegment2D* seg)!! " << endl;
  } 
  if (T0_seg && T0_seg_debug) cout << " After  fitT0_seg(seg) in update(DTRecSegment2D* seg) seg ?? t0cor =  " << t0cor <<endl;
     
  updateHits(seg);
  fit(seg);
}

void DTSegmentUpdator::fit(DTRecSegment4D* seg) {
  // after the update must refit the segments
  if(seg->hasPhi()) fit(seg->phiSegment());
  if(seg->hasZed()) fit(seg->zSegment());

  if(seg->hasPhi() && seg->hasZed() ) {

    DTChamberRecSegment2D *segPhi=seg->phiSegment();
    DTSLRecSegment2D *segZed=seg->zSegment();

    // NB Phi seg is already in chamber ref
    LocalPoint posPhiInCh = segPhi->localPosition();
    LocalVector dirPhiInCh= segPhi->localDirection();

    // Zed seg is in SL one
    GlobalPoint glbPosZ = ( theGeom->superLayer(segZed->superLayerId()) )->toGlobal(segZed->localPosition());
    LocalPoint posZInCh = ( theGeom->chamber(segZed->superLayerId().chamberId()) )->toLocal(glbPosZ);

    GlobalVector glbDirZ = (theGeom->superLayer(segZed->superLayerId()) )->toGlobal(segZed->localDirection());
    LocalVector dirZInCh = (theGeom->chamber(segZed->superLayerId().chamberId()) )->toLocal(glbDirZ);

    LocalPoint posZAt0 = posZInCh+
      dirZInCh*(-posZInCh.z())/cos(dirZInCh.theta());

    // given the actual definition of chamber refFrame, (with z poiniting to IP),
    // the zed component of direction is negative.
    LocalVector dir=LocalVector(dirPhiInCh.x()/fabs(dirPhiInCh.z()),
                                dirZInCh.y()/fabs(dirZInCh.z()),
                                -1.);

    seg->setPosition(LocalPoint(posPhiInCh.x(),posZAt0.y(),0.));
    seg->setDirection(dir.unit());

    AlgebraicSymMatrix mat(4);

    // set cov matrix
    mat[0][0]=segPhi->parametersError()[0][0]; //sigma (dx/dz)
    mat[0][2]=segPhi->parametersError()[0][1]; //cov(dx/dz,x)
    mat[2][2]=segPhi->parametersError()[1][1]; //sigma (x)
    
    seg->setCovMatrix(mat);
    seg->setCovMatrixForZed(posZInCh);

  }
  else if (seg->hasPhi()) {
    DTChamberRecSegment2D *segPhi=seg->phiSegment();

    seg->setPosition(segPhi->localPosition());
    seg->setDirection(segPhi->localDirection());

    AlgebraicSymMatrix mat(4);
    // set cov matrix
    mat[0][0]=segPhi->parametersError()[0][0]; //sigma (dx/dz)
    mat[0][2]=segPhi->parametersError()[0][1]; //cov(dx/dz,x)
    mat[2][2]=segPhi->parametersError()[1][1]; //sigma (x)

    seg->setCovMatrix(mat);
  }
  else if (seg->hasZed()) {
    DTSLRecSegment2D *segZed=seg->zSegment();

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


bool DTSegmentUpdator::fit(DTSegmentCand* seg) {
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
  double chi2=0;
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

void DTSegmentUpdator::fit(DTRecSegment2D* seg) {
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
  double chi2=0;
  fit(x,y,sigy,pos,dir,covMat,chi2);

  seg->setPosition(pos);

  seg->setDirection(dir);

  //cout << "pos " << segPosition<< endl;
  //cout << "dir " << segDirection<< endl;

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
                           double& chi2) {
  float slope,intercept,covss,covii,covsi;
  // do the fit
  theFitter->fit(x,y,x.size(),sigy,slope,intercept,covss,covii,covsi);
  // cout << " DTSegmentUpdator::fit slope = " << slope << " intercept= " << intercept << endl;
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

void DTSegmentUpdator::updateHits(DTRecSegment2D* seg) {
  GlobalPoint pos = (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localPosition());
  GlobalVector dir = (theGeom->idToDet(seg->geographicalId()))->toGlobal(seg->localDirection());
  updateHits(seg, pos, dir);
}

// The GlobalPoint and the GlobalVector can be either the glb position and the direction
// of the 2D-segment itself or the glb position and direction of the 4D segment
void DTSegmentUpdator::updateHits(DTRecSegment2D* seg,
                                  GlobalPoint &gpos,
                                  GlobalVector &gdir,
                                  int step) {

  // it is not necessary to have DTRecHit1D* to modify the obj in the container
  // but I have to be carefully, since I cannot make a copy before the iteration!

  vector<DTRecHit1D> toBeUpdatedRecHits = seg->specificRecHits();
  vector<DTRecHit1D> updatedRecHits;
  float t0cor_seg = seg->theT0;  // get the current value of the t0seg
  if ( T0_seg_debug)   cout << " in  updateHits 2D : ?? seg !! t0cor =  " << t0cor_seg <<endl;
  for (vector<DTRecHit1D>::iterator hit= toBeUpdatedRecHits.begin(); 
       hit!=toBeUpdatedRecHits.end(); ++hit) {
    float t0cor= t0cor_seg;
    const DTLayer* layer = theGeom->layer( hit->wireId().layerId() );
    
    LocalPoint segPos=layer->toLocal(gpos);
    LocalVector segDir=layer->toLocal(gdir);
    // define impact angle needed by the step 2
    float angle = atan(segDir.x()/-segDir.z());
    // define the local position (extr.) of the segment. Needed by the third step 
    LocalPoint segPosAtLayer=segPos+segDir*(-segPos.z())/cos(segDir.theta());
    
    DTRecHit1D newHit1D=(*hit);

    bool ok=true;

    if (step==2) {
    /* Do the update as in step 3 if T0_seg */
 
       if ((step==2 ) && T0_seg ) 
	  {   LocalPoint hitPos(hit->localPosition().x(),+segPosAtLayer.y(),0.);      
	        GlobalPoint glbpos= theGeom->layer( hit->wireId().layerId() )->toGlobal(hitPos);
	        ok = theAlgo->compute(layer,
			    (*hit),
			    angle,glbpos,
				      newHit1D,t0cor); } // per Nicola  ...ma non mi sembra che faccia alcun Update ....
            else {ok = theAlgo->compute(layer,
			    (*hit),			    
			     angle,
			     newHit1D,t0cor); }
    
     } else if (step==3) {
      
      LocalPoint hitPos(hit->localPosition().x(),+segPosAtLayer.y(),0.);
      
      GlobalPoint glbpos= theGeom->layer( hit->wireId().layerId() )->toGlobal(hitPos);

      ok = theAlgo->compute(layer,
			    (*hit),
			    angle,glbpos,
			    newHit1D,t0cor);
    } else {
      throw cms::Exception("DTSegmentUpdator")<<" updateHits called with wrong step"<<endl;
    }

    if (ok) {
      updatedRecHits.push_back(newHit1D);
    } else {
      LogError("DTSegmentUpdator")<<"DTSegmentUpdator::updateHits failed update" << endl;
      throw cms::Exception("DTSegmentUpdator")<<"updateHits failed update"<<endl;
    }
  }
  seg->update(updatedRecHits);
}

void DTSegmentUpdator::fitT0_seg(DTRecSegment2D* seg, float& t0cor) {
  // WARNING: since this method is called both with a 2D and a 2DPhi as argument
  // seg->geographicalId() can be a superLayerId or a chamberId 

  vector<double> d_drift;
  vector<float> t;
  vector<float> x;
  vector<float> y;
  vector<float> sigy;
  vector<int> lc;
  bool vdrift_4parfit=true;  // Per Nicola ... qui per applicare Vdrift_corr , dopo aver sostituito opportunamete la fitT0(x,..)
  vector<DTRecHit1D> hits=seg->specificRecHits();

  DTWireId wireId;
  int npt=0;
  for (vector<DTRecHit1D>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {
       wireId = hit->wireId();
//      cout <<  "NPT wireId " << wireId << endl;
       npt++;
  }
  npt=0;
   for (vector<DTRecHit1D>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {

    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = ( theGeom->layer( hit->wireId().layerId() ) )->toGlobal(hit->localPosition());
    LocalPoint pos = ( theGeom->idToDet(seg->geographicalId()) )->toLocal(glbPos);
    float time=0;
   //int wireId=(hit->wireId().wire());
   const DTLayer* layer = theGeom->layer( hit->wireId().layerId() );
   float xwire = layer->specificTopology().wirePosition(hit->wireId().wire());
   float distance = fabs(hit->localPosition().x() - xwire);
   time	=hit->digiTime();

   //  distanza dal filo: l'hit in coordinate locali - posizione del filo
   //  in coordinate locali

   //  cout << " Drift space  d_drift "<<distance  <<endl;
 
    int ilc=0;
    if ( hit->lrSide() == DTEnums::Left ) ilc= 1;
    else                                  ilc=-1;    

    npt++;
    x.push_back(pos.z()); 
    y.push_back(pos.x());
    lc.push_back(ilc);
   t.push_back(time);
   d_drift.push_back(distance);

  // cout << "fitT0_seg time "<< time<< " d_drift "<<distance  <<" npt= " <<npt<<endl;
  }

  double chi2fit=0;
  double chi20=0;
  int nptfit=npt;
  int nppar=0;
  float aminf;
  float bminf;
  float  cminf=0.;
  double vminf=0.;
  double vminf0=0;
  /*  cout << "dimensione vettori: " << x.size() << " "
                                 << y.size() << " "
                                 << lc.size() << " "
                                 << endl;*/
   chi20 =seg->chi2(); //previous chi2

//  if (nptfit>2) fitT0(x,y,sigy,lc,nptfit,nppar,pos,dir,aminf,bminf,cminf,chi2fit );
		 //NB chi2fit is normalized
  if (nptfit>2) Fit4Var(x,y,sigy,lc,d_drift,nptfit,nppar,aminf,bminf,cminf,vminf0,chi2fit,false);
  vminf=vminf0;

  seg->setT0(0.1000); //just for setting a dummy unused value to tell that T0 has been already applyed;

 float dvDrift0 = -0.000001;
 float t0cor_dvDrift=0.;

  if ( nptfit>2            )  {
     t0cor = -cminf/0.00543; // in ns ;
     if ( (abs(vminf))< 0.09 )  dvDrift0 = vminf;
     // Per Nicola ... si potrebbe sostituire il valore della vdrift costante  usata nell'algo per creare le hits...

     int   t0cor_10 = int(t0cor *10); //in 0.1 ns;

     t0cor_dvDrift=t0cor_10;

  			                       
     if (vdrift_4parfit) {
 		float dvDrift  = dvDrift0;
  		if ( dvDrift0 < 0. )  {   dvDrift=  - dvDrift0 +.1;  }
	
                          		t0cor_dvDrift =   dvDrift + t0cor_10 ;
 		if ( t0cor_10 < 0. )    t0cor_dvDrift = - dvDrift + t0cor_10 ;
		 
 		if (t0cor != 0) cout <<  "NPT " << npt <<  "   " <<  wireId.wheel() << "   " << wireId.station() << "   " <<  wireId.sector() <<" " << t0cor << " vdrift= " << dvDrift0 <<endl;
		//if (t0cor != 0) cout <<  "NPT " <<"t0cor " << t0cor_dvDrift << endl;
  		//if (t0cor != 0) cout <<  "NPT t0cor _10= " << t0cor_10 << " vdrift= " << dvDrift << "t0_seg= " << t0cor_dvDrift <<endl;
    		if (t0cor_dvDrift != 0) {
			for (int jnpt=0; jnpt<npt; jnpt++){
        		 cout <<  "NPT " <<jnpt+1 << "  " <<x[jnpt] << " " <<y[jnpt] <<"    " <<lc[jnpt]<< " " << d_drift[jnpt] << endl;
  			}
  		}
     }

     seg->setT0(t0cor_dvDrift); 
   }

//aaaa  if ((wireId.wheel() !=2) && (wireId.station()!=1 ) )
//aaaa   { t0cor= 0}
//aaaa  cout << " npt fit t0   seg->setT0( ns)= " << t0cor <<endl;
}



void DTSegmentUpdator::Fit4Var(
	    const vector<float>& xfit,
	    const vector<float>& yfit,
            const vector<float>& sigy,
            const vector<int>& lfit,
	    const vector<double>& tfit,
            int& nptfit,
            int& nppar,
//            LocalPoint& pos,
//            LocalVector& dir,
           float& aminf,
           float& bminf,
           float& cminf,
	   double& vminf,
	   double& chi2fit,
	   bool debug){ 

               double sigma = 0.0250;// errors can be inserted .just load them
               double delta = 0;
               double sx = 0;
               double  sx2 = 0;
               double sy = 0;
               double sxy = 0;
               double sl = 0;
               double sl2 = 0;
               double sly = 0;
               double slx = 0;
	       double st = 0;
	       double st2 = 0;
	       double slt = 0;
	       double sltx = 0;
	       double slty = 0;
             double  chi2fit2=-1;
	     double  chi2fitN2=-1 ;
             double  chi2fit3=-1;
	     double  chi2fitN3=-1 ;
//           double  chi2fit4=-1;
	     double  chi2fitN4=-1 ;
	    float bminf3=bminf;
	    float aminf3=aminf;
	    float cminf3=cminf;
	    int nppar2=0;
	    int nppar3=0;
	    int nppar4=0;

                  //cout << "sigma = "<<sigma;
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
         delta = nptfit*sx2 - sx*sx;
//	       cout << delta << " " << nptfit << " " << sigma << endl;
               //cout << "npfit"<< nptfit<< "delta"<< delta<<endl;
               //cout << <<endl;
               double a = 0;
               double b = 0;	       
	       if (delta!=0){
               a = (sx2*sy - sx*sxy)/delta;
               b = (nptfit*sxy - sx*sy)/delta;

//                   cout << " NPAR=2 : slope = "<<b<< "    intercept = "<<a <<endl;
                 for (int j=0; j<nptfit; j++){
                double ypred = a + b*xfit[j];
                double dy = (yfit[j] - ypred)/sigma;
                chi2fit = chi2fit + dy*dy;
                        } //end loop chi2
			}

             nppar=2; 
             nppar2=nppar; 

               chi2fit2 = chi2fit;
	       chi2fitN2 = chi2fit/(nptfit-2);
// cout << "dt0 = 0chi2fit = " << chi2fit << "  slope = "<<b<<endl;
// cout << chi2fit;


	double d1=  sy;
        double d2=  sxy;
    	double d3=  sly;
    	double c1=  sl;
    	double c2=  slx;
    	double c3=  sl2;
    	double b1=  sx;
    	double b2=  sx2;
    	double b3=  slx;
    	double a1= nptfit;
    	double a2=  sx;
    	double a3=  sl;
    	double b4= b2*a1-b1*a2; // questo e i seguenti sono inutilizzati nel fit a 4 variabili
    	double c4= c2*a1-c1*a2;
    	double d4= d2*a1-d1*a2;
    	double b5= a1*b3-a3*b1;
    	double c5= a1*c3-a3*c1;
    	double d5= a1*d3-d1*a3;
    	double a6 = slt;
    	double b6 = sltx;
   	double c6 = st;
    	double v6 = st2;	
//    	double d6 = slty;
    	double det = (a1*a1*(b2*v6 - b6*b6) - a1*(a2*a2*v6 - 2*a2*a6*b6 + a6*a6*b2 + b2*c6*c6 + b3*(b3*v6 - 2*b6*c6))
               + a2*a2*c6*c6 + 2*a2*(a3*(b3*v6 - b6*c6) - a6*b3*c6) + a3*a3*(b6*b6 - b2*v6)
	       + a6*(2*a3*(b2*c6 - b3*b6) + a6*b3*b3)); 
	cminf=0.;
   if (nptfit>=3) {
     	if (((c5*b4-c4*b5)*b4*a1)!=0) {
      	nppar=3;
      	chi2fit = 0.;
      	cminf=(d5*b4-d4*b5)/(c5*b4-c4*b5);
      	bminf=d4/b4 -cminf *c4/b4;
      	aminf=(d1/a1 -cminf*c1/a1 -bminf*b1/a1);

        for (int j=0; j<nptfit; j++){
                double ypred = aminf + bminf*xfit[j];
                double dy = (yfit[j]-cminf*lfit[j] - ypred)/sigma;
                chi2fit = chi2fit + dy*dy;
 
                        } //end loop chi2
     	chi2fit3 = chi2fit;
    	 if (nptfit>3)
     	chi2fitN3 = chi2fit /(nptfit-3);
   
    	}
    	else {
          cminf=0;
          bminf=b;
          aminf=a;
     	  chi2fit3 = chi2fit;
     	  chi2fitN3 = chi2fit /(nptfit-2);
   	 }

    bminf3=bminf;
    aminf3=aminf;
    cminf3=cminf;
   }  
   nppar3=nppar;
   if (debug) cout << "   dt0= 0 : slope 2  = "<<b     << "  pos in  = " << a     <<" chi2fitN2 = " << chi2fitN2 <<"  nppar= " << nppar2 << " nptfit= " << nptfit <<endl;
   if (debug) cout << "   dt0= 0 : slope 3  = "<<bminf << "  pos out = " << aminf <<" chi2fitN3 = " << chi2fitN3 <<"  nppar= " << nppar3 << " T0_ev ns= " << cminf/0.00543 <<endl;
     
    if (debug) cout << "   dt0= 0 : slope 4  = "<<bminf << "  pos out = " << aminf <<" chi2fitN4 = " << chi2fitN4 <<"  nppar= " <<nppar4<< " T0_ev ns= " << cminf/0.00543 <<" delta v = "<< vminf <<endl;
    if (abs(vminf)>=0.09) {
        cout << "  vminf gt 0.09 det=  " << det << endl;
	cout << "   dt0= 0 : slope 4  = "<<bminf << "  pos out = " << aminf <<" chi2fitN4 = " << chi2fitN4 << " T0_ev ns= " << cminf/0.00543 <<" delta v = "<< vminf <<endl;
        cout << "   dt0= 0 : slope 2  = "<<b     << "  pos in  = " << a     <<" chi2fitN2 = " << chi2fitN2 <<"  nppar= " <<nppar-1<< " nptfit= " << nptfit <<endl;
        cout << "   dt0= 0 : slope 3  = "<<bminf << "  pos out = " << aminf <<" chi2fitN3 = " << chi2fitN3 << " T0_ev ns= " << cminf/0.00543 <<endl;
	cout << nptfit   <<" nptfit "<< "   end  chi2fit = " << chi2fit << "T0_ev ns= " << cminf/0.00543 << "delta v = "<< vminf <<endl;
        
	}
    //   if (debug) cout << " 4-parameters fit:  " << " nppar= " <<nppar<< " nptfit= " <<nptfit<< endl;
    //   if (debug) cout << "   dt0= 0 : slope in = "<<b<< "  pos in = " << a <<endl;
    //   if (debug) cout << "   dt0= 0 : slope out = "<<bminf<< "  pos out = " << aminf <<endl;
    //   if (debug)  cout << nptfit   <<" nptfit "<< "   end  chi2fit = " << chi2fit << "T0_ev ns= " << cminf/0.00543 << "delta v = "<< vminf <<endl;
	chi2fit=-1000.; // chi2fit is the nomalized chi2 at the output end ;
	if (nptfit!= nppar) chi2fit	= chi2fit / (nptfit-nppar);
	 
 
}

void DTSegmentUpdator::fitT0(const vector<float>& xfit,
            const vector<float>& yfit, 
            const vector<float>& sigy,
            const vector<int>& lfit,
            int& nptfit,
            int& nppar,
            LocalPoint& pos,
            LocalVector& dir,
           float& aminf,
           float& bminf,
           float& cminf,
	   double& chi2fit) {
	   bool debug = false;
//int nptfit=2;float *xfit;float *yfit;float *a;float *b;float *chi2fit;int *idt0; 
	float sigma = 0.0295;// errors can be inseted .just load them 
	float delta = 0;
        float sx = 0;
	float  sx2 = 0;
	float sy = 0;
	float sxy = 0;
        float sl = 0;
	float sl2 = 0;
	float sly = 0;
	float slx = 0;
	   cminf=-0.00001; //dummy small value;
                  //cout << "sigma = "<<sigma;
                   for (int j=0; j<nptfit; j++){
                    		sx  = sx + xfit[j];        
				sy  = sy + yfit[j];
				sx2 = sx2 + xfit[j]*xfit[j];
				sxy = sxy + xfit[j]*yfit[j];
                    		sl  = sl + lfit[j];        
				sl2 = sl2 + lfit[j]*lfit[j];
				sly = sly + lfit[j]*yfit[j];
				slx = slx + lfit[j]*xfit[j];
				
		       } //end loop
		 delta = nptfit*sx2 - sx*sx;
      		 //cout << "npfit"<< nptfit<< "delta"<< delta<<endl;
      		 //cout << <<endl;
	      	 float a = (sx2*sy - sx*sxy)/delta;
		 float b = (nptfit*sxy - sx*sy)/delta;

                   //cout << " slope = "<<b<< "    intercept = "<<a<< "  slope_or = "<<SEG_ss[it]<<endl;
        		 for (int j=0; j<nptfit; j++){
				float ypred = a + b*xfit[j];
				float dy = (yfit[j] - ypred)/sigma;
				chi2fit = chi2fit + dy*dy/(nptfit-2);
       				 } //end loop chi2
		     nppar=2;  
			
		     float  chi2fit0 = chi2fit*(nptfit-2);
// cout << "dt0 = 0chi2fit = " << chi2fit << "  slope = "<<b<<endl;
// cout << chi2fit;/* */
	float d1=  sy;
	float d2=  sxy;
	float d3=  sly;
	float c1=  sl;
	float c2=  slx;
	float c3=  sl2;
	float b1=  sx;
	float b2=  sx2;
	float b3=  slx;
	float a1= nptfit;
	float a2=  sx; 
	float a3=  sl;
	float b4= b2*a1-b1*a2;
	float c4= c2*a1-c1*a2;
	float d4= d2*a1-d1*a2;
	float b5= a1*b3-a3*b1;
	float c5= a1*c3-a3*c1;
	float d5= a1*d3-d1*a3;
	 if (((c5*b4-c4*b5)*b4*a1)!=0) {
	  nppar=3;
	  chi2fit = 0.;
	  cminf=(d5*b4-d4*b5)/(c5*b4-c4*b5);
	  bminf=d4/b4 -cminf *c4/b4;
	  aminf=(d1/a1 -cminf*c1/a1 -bminf*b1/a1);

        		 for (int j=0; j<nptfit; j++){
				float ypred = aminf + bminf*xfit[j];
				float dy = (yfit[j]-cminf*lfit[j] - ypred)/sigma;
				chi2fit = chi2fit + dy*dy;
  
       				 } //end loop chi2
	 chi2fit0 = chi2fit;
	 chi2fit = chi2fit /(nptfit-3);
	
	}
	else {
	      cminf=0.1;
	      bminf=b;
	      aminf=a;
	}
	 
	 if (debug) cout << " NPT   dt0= 0 : slope in = "<<b<< "  pos in = " << a << " nppar= " <<nppar<< " nptfit= " <<nptfit<<endl;
  	 if (debug) cout << " NPT   dt0= 0 : slope out = "<<bminf<< "  pos out = " << aminf <<endl;
 	 if (debug) cout << " NPT   dt0= 0 chi2fit = " << chi2fit0 << "  T0_ev correction ns " <<-cminf/0.00543 <<endl;
}


