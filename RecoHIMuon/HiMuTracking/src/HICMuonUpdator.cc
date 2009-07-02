#include "RecoHIMuon/HiMuTracking/interface/HICMuonUpdator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
//#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <cmath>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>

//#define UPDATOR_BARREL_DEBUG_TRUE
//#define UPDATOR_BARREL_DEBUG
//#define UPDATOR_ENDCAP_DEBUG
//#define CORRECT_DEBUG
//#define LINEFIT_DEBUG

//#define DEBUG

using namespace std;

TrajectoryStateOnSurface HICMuonUpdator::update(const Trajectory& mt,
                                                const TrajectoryStateOnSurface& nTsos,
		                                const TrajectoryMeasurement& ntm, 
					        const DetLayer* layer,
					        double& chirz, double& chirf) const {
  double pi = 4.*atan(1.);
  double twopi=8.*atan(1.);
  TrajectoryStateOnSurface badtsos; 
  if(!nTsos.isValid()) {
   std::cout<<" HICMuonUpdator::update:: can not start::initial tsos is not valid " <<std::endl;
   return badtsos;
  }
// trajectory type
  
  vector<TrajectoryMeasurement> MTM=mt.measurements();
#ifdef DEBUG   
  std::cout<<" HICMuonUpdator::update::MTM size "<<MTM.size()<<" vertex "<<zvert<<std::endl;
  std::cout<<" HICMuonUpdator::update::charge from trajectory"<<(MTM.back()).updatedState().freeTrajectoryState()->parameters().charge()<<std::endl;
  std::cout<<" HICMuonUpdator::update::charge predicted tsos"<<nTsos.freeTrajectoryState()->charge()<<std::endl;
  std::cout<<" HICMuonUpdator::update::momentum "<<(MTM.back()).updatedState().freeTrajectoryState()->parameters().momentum()<<std::endl;
#endif 
  vector<double> phihit,rhit,zhit,dphihit,drhit,dzhit,dzhitl,ehitphi,dehitphi,ehitstrip;

  double rvert=0.;
  double ezvert=0.014;
  
  const TransientTrackingRecHit::ConstRecHitPointer pRecHit=(MTM.back()).recHit();
  const TransientTrackingRecHit::ConstRecHitPointer nRecHit=ntm.recHit();
  
//  double acharge=(MTM.back()).updatedState().freeTrajectoryState()->parameters().charge();
//  double acharge=nTsos.freeTrajectoryState()->charge();
  GlobalVector pold=(MTM.back()).updatedState().freeTrajectoryState()->parameters().momentum();
  double theta=pold.theta();
  
  for(vector<TrajectoryMeasurement>::const_iterator ihit=MTM.begin();ihit!=MTM.end();ihit++){
  
//    FreeTrajectoryState* ftshit = (*ihit).updatedState().freeTrajectoryState();
    phihit.push_back((*ihit).recHit()->globalPosition().phi());
    rhit.push_back((*ihit).recHit()->globalPosition().perp());
    zhit.push_back((*ihit).recHit()->globalPosition().z());
    GlobalPoint realhit = (*ihit).recHit()->globalPosition();
     
    double phierror=sqrt((*ihit).recHit()->globalPositionError().phierr(realhit));
    
    if(fabs(phierror)<0.0000001) {
        phierror=0.00008;
    }       
    ehitphi.push_back(phierror);
    
#ifdef UPDATOR_BARREL_DEBUG
    cout<<" Errors "<<phierror<<" "<<(*ihit).recHit()->globalPositionError().rerr(realhit)<<" "<<tan(theta)<<endl;
#endif


    if((*ihit).layer()->location()==GeomDetEnumerators::barrel){
    ehitstrip.push_back(sqrt((*ihit).recHit()->globalPositionError().czz()));
    } else{    
    ehitstrip.push_back(sqrt((*ihit).recHit()->globalPositionError().rerr(realhit)/fabs(tan(theta)))); 
    }   

  }
  
  phihit.push_back(nRecHit->globalPosition().phi());
  rhit.push_back(nRecHit->globalPosition().perp());
  zhit.push_back(nRecHit->globalPosition().z()); 
  ehitphi.push_back(sqrt(nRecHit->globalPositionError().phierr(nRecHit->globalPosition())));





  if(ntm.layer()->location()==GeomDetEnumerators::barrel){
    ehitstrip.push_back(sqrt(nRecHit->globalPositionError().czz())); 
  } else {
    ehitstrip.push_back(sqrt(nRecHit->globalPositionError().rerr(nRecHit->globalPosition()))/fabs(tan(theta)));
  }
  
// add vertex 
    
  rhit.push_back(rvert);
  zhit.push_back(zvert); 
  ehitstrip.push_back(ezvert);    
   
  for(vector<double>::const_iterator iphi=phihit.begin();iphi!=phihit.end()-1;iphi++){
  double dpnew=fabs(*iphi-*(iphi+1));
  if(dpnew>pi) dpnew=twopi-dpnew;
  
#ifdef UPDATOR_BARREL_DEBUG
  cout<<" dphi=dpnew="<<dpnew<<" "<<*iphi<<" "<<*(iphi+1)<<endl;
#endif

  dphihit.push_back(dpnew);  
  }
  
  for(vector<double>::const_iterator ir=rhit.begin();ir!=rhit.end()-2;ir++){
  double dpnew=fabs(*ir-*(ir+1));
  
#ifdef UPDATOR_BARREL_DEBUG
  cout<<" dr=dpnew="<<dpnew<<" "<<*ir<<" "<<*(ir+1)<<endl;
#endif

  drhit.push_back(dpnew);  
  }
  for(vector<double>::const_iterator iz=zhit.begin();iz!=zhit.end()-2;iz++){
  double dpnew=*iz-*(iz+1);
  
#ifdef UPDATOR_BARREL_DEBUG
  cout<<" dZ=dpnew="<<dpnew<<" "<<*iz<<" "<<*(iz+1)<<endl;
#endif
  
  dzhit.push_back(fabs(dpnew));
  dzhitl.push_back(dpnew);  
  }
  
  dzhitl.push_back(*(zhit.end()-1)-zvert);
  
  for(vector<double>::const_iterator ie=ehitphi.begin();ie!=ehitphi.end()-1;ie++){
  double dpnew=(*ie)*1.14;
  dehitphi.push_back(dpnew);  
  }
//
//=================fit in rf and rz planes separately
//
int tType = 1;
if ( (*(MTM.begin())).layer()->location()==GeomDetEnumerators::barrel){
//  std::cout<<" Update barrel "<<std::endl;
  TrajectoryStateOnSurface tsos=updateBarrel(rhit, zhit, dphihit, drhit, ehitstrip, dehitphi, pRecHit, nRecHit, 
                                                                 nTsos, chirz, chirf, tType);
	
	if(!tsos.isValid()) {
#ifdef DEBUG
      std::cout<<" Trajectory is not valid "<<std::endl;
#endif	
	 return badtsos;
	}							
              
   return tsos;

} else{
//  std::cout<<" Update endcap "<<std::endl;
  TrajectoryStateOnSurface tsos=updateEndcap(rhit, zhit, dphihit, dzhit, ehitstrip, dehitphi, pRecHit, nRecHit, 
                                                        nTsos, chirz, chirf, tType);
   return tsos;

}   
}

bool HICMuonUpdator::linefit2(const vector <double>& y, const vector <double>& x, 
                         const vector <double>& err, double& co1, double& co2,
                                                                            double & chi2) const{
double a00=0.;
double a01=0.;
double a10=0.;
double a11=0.;

double b0=0.;
double b1=0.;

bool fit;
fit=false;

#ifdef LINEFIT_DEBUG
cout<<" linefit2::sizes="<<x.size()<<" "<<y.size()<<" "<<err.size()<<endl;
cout<<" linefit2::a00 "<<a00<<" "<<a01<<" "<<a11<<" "<<b0<<" "<<b1<<endl;
#endif

if(x.size() != y.size()) return fit;
if(x.size() != err.size()) return fit;


for (unsigned int i=0;i<x.size();i++){
#ifdef LINEFIT_DEBUG
cout<<" line2fit "<<a00<<" "<<x[i]/err[i]<<" "<<x[i]<<" "<<err[i]<<" second try "<<err[i]<<endl; 
#endif
a00=a00+(x[i]/err[i])*(x[i]/err[i]);
a01=a01+(x[i]/err[i])/err[i];
a10=a01;
a11=a11+1./(err[i]*err[i]);
b0=b0+(x[i]/err[i])*(y[i]/err[i]);
b1=b1+(y[i]/err[i])/err[i];

#ifdef LINEFIT_DEBUG
cout<<"linefit2="<<x[i]<<" "<<y[i]<<" "<<err[i]<<" "<<a00<<" "<<a01<<" "<<a11<<" "<<b0<<" "<<b1<<endl;
#endif

}
double det=a00*a11-a01*a01;

#ifdef LINEFIT_DEBUG
cout<<" linefit2::det="<<det<<endl;
#endif

if(fabs(det)<0.00000001) return fit;
co1=(b0*a11-b1*a01)/det;
co2=(b1*a00-b0*a10)/det;

// check if it is 90 degree track
#ifdef LINEFIT_DEBUG
cout<<" linefit2::Previous element= "<<y[x.size()-3]<<" "<<x[x.size()-3]<<endl;
#endif

if(y[x.size()-2]<14.) {

#ifdef LINEFIT_DEBUG
cout<<" check 90 degree track "<<endl;
#endif

if(fabs(x[x.size()-2]-x[x.size()-1])<0.1){

#ifdef LINEFIT_DEBUG
cout<<" Redetermine line - 90 degree "<<endl;
#endif

if(fabs(x[x.size()-2]-x[x.size()-1])>0.0001){
co1=(y[x.size()-2]-y[x.size()-1])/(x[x.size()-2]-x[x.size()-1]);
co2=y[x.size()-2]-co1*x[x.size()-2];
}
 else 
     {
co1=10000.;
co2=-zvert*10000.;    
 }
}    
}

#ifdef LINEFIT_DEBUG
cout<<"linefit2::co1,co2="<<co1<<" "<<co2<<" size "<<x.size()<<endl;
#endif
// CHI2
chi2=0.;
for (unsigned int i=0;i<x.size();i++){
double zdet=(y[i]-co2)/co1;
chi2=chi2+(x[i]-zdet)*(x[i]-zdet)/(err[i]*err[i]);

#ifdef LINEFIT_DEBUG
cout<<"linefit2::chi2="<<chi2<<" err="<<err[i]<<" x[i]="<<x[i]<<" teor="<<zdet<<endl;
#endif

}
// chi2 on degree of freedom
chi2=chi2/x.size();
#ifdef LINEFIT_DEBUG
cout<<" linefit2::chi2= "<<chi2<<endl;
#endif

return fit=true;
}
bool HICMuonUpdator::linefit1(const vector <double>& y, const vector <double>& x, const vector <double>& err, 
                         double& co1, double& chi2)
          const{
double s1=0.;
double s2=0.;

bool fit;
fit=false;

#ifdef LINEFIT_DEBUG
cout<<" linefit1::sizes="<<x.size()<<" "<<y.size()<<" "<<err.size()<<endl;
#endif

if(x.size() != y.size()) return fit;
if(x.size() != err.size()) return fit;

for (unsigned int i=0;i<x.size();i++){
s1=s1+(x[i]/err[i])*(y[i]/err[i]);
s2=s2+(x[i]/err[i])*(x[i]/err[i]);

#ifdef LINEFIT_DEBUG
cout<<"linefit1="<<x[i]<<" "<<y[i]<<" "<<err[i]<<" "<<s1<<" "<<s2<<endl;
#endif

}

co1=s1/s2;

#ifdef LINEFIT_DEBUG
cout<<"linefit1::co1,co2="<<co1<<endl;
#endif
// CHI2
chi2=0.;
for (unsigned int i=0;i<x.size();i++){
chi2=chi2+(y[i]-co1*x[i])*(y[i]-co1*x[i])/(err[i]*err[i]);
}
chi2=chi2/x.size();
#ifdef LINEFIT_DEBUG
cout<<"linefit1::chi2="<<chi2<<endl;
#endif
return fit=true;
}


double
        HICMuonUpdator::findPhiInVertex(const FreeTrajectoryState& fts, const double& rc, const GeomDetEnumerators::Location location) const{

     double pi = 4.*atan(1.);
//     double twopi=8.*atan(1.);

     double acharge=fts.parameters().charge();
     double phiclus=fts.parameters().position().phi();
     double psi;
   if(location==GeomDetEnumerators::barrel){
     double xrclus=fts.parameters().position().perp();
     double xdouble=xrclus/(2.*rc);
     psi= phiclus+acharge*asin(xdouble);
   } else {
     double zclus=fts.parameters().position().z();
     double pl=fts.parameters().momentum().z(); 
     psi=phiclus+acharge*0.006*fabs(zclus)/fabs(pl);     
   }  
     double phic = psi-acharge*pi/2.;
#ifdef CORRECT_DEBUG	
	cout<<"Momentum of track="<<fts.parameters().momentum().perp()<<
	" rad of previous cluster= "<<fts.parameters().position().perp()<<
	" phi of previous cluster="<<fts.parameters().position().phi()<<endl;
	cout<<" position of the previous cluster="<<fts.parameters().position()<<endl;
	cout<<"radius of track="<<rc<<endl;
	cout<<"acharge="<<acharge<<endl;
	cout<<"psi="<<psi<<endl;
	cout<<"phic="<<phic<<" pi="<<pi<<" pi2="<<pi2<<endl;
#endif
     
     return phic;
}

TrajectoryStateOnSurface HICMuonUpdator::updateBarrel(vector<double>& rhit, vector<double>& zhit, 
                                                 vector<double>& dphihit, vector<double>& drhit, 
	                                         vector<double>& ehitstrip, vector<double>& dehitphi,
						 const TransientTrackingRecHit::ConstRecHitPointer& pRecHit, 
						 const TransientTrackingRecHit::ConstRecHitPointer& nRecHit, 
						 const TrajectoryStateOnSurface& nTsos,
						 double& chirz, double& chirf, int& tType) const{

// fit in (dphi dr), (dphi-dz)
  TrajectoryStateOnSurface badtsos; 
       double pi = 4.*atan(1.);
     double twopi=8.*atan(1.);

//  cout<<" Update barrel begin "<<endl;

  double ch1,dphi,dr,ptnew;
  double co1,co2;
  bool fitrf,fitrz;
  
// fit in (ZR)-coordinate 

  fitrz=this->linefit2(rhit,zhit,ehitstrip,co1,co2,chirz);
  
#ifdef UPDATOR_BARREL_DEBUG  
  cout<<"UPDATE::BARREL::line fit rz= "<<fitrz<<" chirz="<<chirz<<endl;
  cout<<" co1="<<co1<<" co2="<<co2<<endl;
#endif  

  if(!fitrz) {
#ifdef DEBUG  
  cout<<"UPDATE::BARREL::line fit failed rz= "<<fitrz<<" chirz="<<chirz<<endl;
#endif  
  
  return badtsos;
  }   
  if(dphihit.size()>1){
  fitrf=this->linefit1(dphihit,drhit,dehitphi,ch1,chirf);
  
#ifdef UPDATOR_BARREL_DEBUG
  cout<<"UPDATE::BARREL::line fit dphi= "<<fitrf<<" chirf="<<chirf<<endl;
  cout<<" ch1="<<ch1<<endl;
#endif

  if(!fitrf) {
#ifdef DEBUG  
  cout<<"UPDATE::BARREL::line fit failed dphi= "<<fitrf<<" chirz="<<chirf<<endl;
#endif  
  return badtsos;
  }
  }else{

  chirf = 0.;
  dphi=fabs(dphihit.back());
  dr=fabs(drhit.back());
  if(dphi > pi) dphi = twopi-dphi;
  ch1=dphi/dr;
  
#ifdef UPDATOR_BARREL_DEBUG
  cout<<"UPDATE::BARREL::line calc dphi= "<<dphi<<" dr="<<dr<<" chirf="<<chirf<<endl;
  cout<<" ch1="<<ch1<<endl;
#endif

  }
  
// Updating trajectory
  ptnew=0.006/ch1;
  GlobalPoint xrhit = nRecHit->globalPosition();
  TrackCharge aCharge = nTsos.freeTrajectoryState()->parameters().charge();
  double phiclus=xrhit.phi();
  double xrclus=xrhit.perp();
  //double xzclus=xrhit.z();
  double rc=100.*ptnew/(0.3*4.);
  double xdouble=xrclus/(2.*rc);
  double psi=phiclus-aCharge*asin(xdouble);
  double pznew=ptnew/co1;
  double znew=(xrclus-co2)/co1;
  double delphinew=fabs(0.006*drhit.back()/ptnew);
  double phinew=pRecHit->globalPosition().phi()+aCharge*delphinew;
  GlobalVector pnew(ptnew*cos(psi),ptnew*sin(psi),pznew);
  GlobalPoint xnew(xrclus*cos(phinew),xrclus*sin(phinew),znew);
  AlgebraicSymMatrix m(5,0);        
  m(1,1)=ptnew; m(2,2)=thePhiWin, 
  m(3,3)=theZWin, 
  m(4,4)=thePhiWin, 
  m(5,5)=theZWin;
  
#ifdef UPDATOR_BARREL_DEBUG  
  cout<< "MuUpdator::xnew=" << xnew<<endl;
#endif  
       
  TrajectoryStateOnSurface tsos(
                           GlobalTrajectoryParameters(xnew, pnew, aCharge, field),
                           CurvilinearTrajectoryError(m), nTsos.surface());
			   
//  cout<<" Update barrel end  "<<xnew<<endl;
  			   
  return tsos;						
}
TrajectoryStateOnSurface HICMuonUpdator::updateEndcap(vector<double>& rhit, vector<double>& zhit, 
                                                 vector<double>& dphihit, vector<double>& drhit, 
	                                         vector<double>& ehitstrip, vector<double>& dehitphi,
						 const TransientTrackingRecHit::ConstRecHitPointer& pRecHit, 
						 const TransientTrackingRecHit::ConstRecHitPointer& nRecHit, 
						 const TrajectoryStateOnSurface& nTsos,
						 double& chirz, double& chirf, int& tType) const{

// fit in (dphi dr), (dphi-dz)
  double pi = 4.*atan(1.);
  double twopi=8.*atan(1.);


  TrajectoryStateOnSurface badtsos;

//    cout<<" Update endcap begin "<<endl;

  double ch1,dphi,dr;
  double co1,co2;
  bool fitrf,fitrz;
  
#ifdef UPDATOR_ENDCAP_DEBUG
  cout<<"updateEndcap switched on"<<endl;
#endif
  
// fit in (ZR)-coordinate 

  fitrz=this->linefit2(rhit,zhit,ehitstrip,co1,co2,chirz);

#ifdef UPDATOR_ENDCAP_DEBUG  
  cout<<"line fit rz= "<<fitrz<<" chirz="<<chirz<<endl;
  cout<<" co1="<<co1<<" co2="<<co2<<endl;
#endif  

  if(!fitrz) {
#ifdef DEBUG  
  cout<<"UPDATE::ENDCAP::line fit failed rz= "<<fitrz<<" chirz="<<chirz<<endl;
#endif  
  return badtsos;
  }
  if(dphihit.size()>1){
  fitrf=this->linefit1(dphihit,drhit,dehitphi,ch1,chirf);
  if(zhit.front()<0.) ch1=-1.*ch1;
#ifdef UPDATOR_ENDCAP_DEBUG
  cout<<"MuUpdate::barrel::line fit dphi= "<<fitrf<<" chirf="<<chirf<<endl;
  cout<<" ch1="<<ch1<<endl;
#endif
  if(!fitrf) {
#ifdef DEBUG  
  cout<<"UPDATE::ENDCAP::line fit failed dphi= "<<fitrf<<" chirz="<<chirf<<endl;
#endif  
  
  return badtsos;
  }
  }else{
  dphi=fabs(dphihit.back());
  dr=fabs(drhit.back());
  if(dphi > pi) dphi = twopi-dphi;
  ch1=dphi/dr;
  if(zhit.front()<0.) ch1=-1.*ch1;  
  chirf = 0.;
#ifdef UPDATOR_ENDCAP_DEBUG  
  cout<< "MuUpdate::barrel::linedphi=" << dphi <<" dz= "<< dr << " ch1= " <<ch1<<" pznew= "<<0.006/ch1<<endl;
#endif  
  }
  
// Updating trajectory
  double pznew=0.006/ch1;
//  cout<<" point 1 "<<endl;
  GlobalPoint xrhit = nRecHit->globalPosition();
//  cout<<" point 2 "<<endl;
  TrackCharge aCharge = nTsos.freeTrajectoryState()->charge();
//  cout<<" point 3 "<<endl;
  double phiclus=xrhit.phi();
//  cout<<" point 4 "<<endl;
  double xzclus=xrhit.z();
//  cout<<" point 5 "<<endl;
  double psi=phiclus-aCharge*0.006*fabs(xzclus-zvert)/fabs(pznew);
//  cout<<" point 6 "<<endl;
  double ptnew=pznew*co1;
//  cout<<" point 7 "<<endl;
  double xrclus=co1*xzclus+co2;
//  cout<<" point 8 "<<endl;
  double delphinew=fabs(0.006*drhit.back()/pznew);
//  cout<<" point 9 "<<endl;
  double phinew=pRecHit->globalPosition().phi()+aCharge*delphinew;
//  cout<<" point 10 "<<endl;
  GlobalVector pnew(ptnew*cos(psi),ptnew*sin(psi),pznew);
//  cout<<" point 11 "<<endl;
  GlobalPoint xnew(xrclus*cos(phinew),xrclus*sin(phinew),xzclus); 
// OK changes. Start each time from the RealHit cluster.
//
//  GlobalPoint xnew(xrhit.x(),xrhit.y(),xzclus);
  
#ifdef UPDATOR_ENDCAP_DEBUG  
  cout<< "MuUpdator::xnew=" << xnew<<endl;
#endif  

    
  AlgebraicSymMatrix m(5,0);        
  m(1,1)=pznew; m(2,2)=thePhiWin, 
  m(3,3)=theZWin, 
  m(4,4)=thePhiWin, 
  m(5,5)=theZWin;
       
  TrajectoryStateOnSurface tsos(
                           GlobalTrajectoryParameters(xnew, pnew, aCharge, field),
                           CurvilinearTrajectoryError(m), nTsos.surface());
			   
//  cout<< "Update endcap end "<<endl;			   
  return tsos;						
}





