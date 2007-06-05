#include "Utilities/Configuration/interface/Architecture.h"
#include "TrackerReco/TkHitAssociation/interface/TkHitAssociator.h"

#include "Tracker/HICPattern/interface/HICMuonUpdator.h"

#include "CommonDet/BasicDet/interface/RecHit.h"
#include "CommonDet/BasicDet/interface/SimHit.h"
#include "CommonDet/BasicDet/interface/SimDet.h"
#include "CommonDet/BasicDet/interface/DetUnit.h"
#include "CommonDet/BasicDet/interface/Det.h"
#include "CommonDet/BasicDet/interface/DetType.h"

#include "CommonDet/DetGeometry/interface/BoundPlane.h"
#include "ClassReuse/GeomVector/interface/GlobalPoint.h"
#include "ClassReuse/GeomVector/interface/GlobalVector.h"

#include "CLHEP/Vector/ThreeVector.h"
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Geometry/Point3D.h>
#include "CLHEP/Units/PhysicalConstants.h"
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
TrajectoryStateOnSurface HICMuonUpdator::update(const Trajectory& mt,
                                                const TrajectoryStateOnSurface& nTsos,
		                                const RecHit& nRecHit, 
					        const DetLayer* layer,
					        double& chirz, double& chirf) const {
// take FTS from nTSOS. Extract momentum and errors
//
//  cout<<" HICMuonUpdator::update::begin "<<endl;

  TrajectoryStateOnSurface badtsos; 

// trajectory type
  int tType=0;
  vector<TrajectoryMeasurement> MTM=mt.data();

#ifdef UPDATOR_BARREL_DEBUG_TRUE
  bool itrust=true;
#endif 
 
  vector<double> phihit,rhit,zhit,dphihit,drhit,dzhit,dzhitl,ehitphi,dehitphi,ehitstrip;

  double rvert=0.;
  double ezvert=0.014;
  
  RecHit pRecHit=(MTM.back()).recHit();
  double acharge=(MTM.back()).updatedState().freeTrajectoryState()->parameters().charge();
  GlobalVector pold=(MTM.back()).updatedState().freeTrajectoryState()->parameters().momentum();
  double theta=pold.theta();
  
  for(vector<TrajectoryMeasurement>::const_iterator ihit=MTM.begin();ihit!=MTM.end();ihit++){
  
    FreeTrajectoryState* ftshit = (*ihit).updatedState().freeTrajectoryState();
    phihit.push_back((*ihit).recHit().globalPosition().phi());
    rhit.push_back((*ihit).recHit().globalPosition().perp());
    zhit.push_back((*ihit).recHit().globalPosition().z());
    

#ifdef UPDATOR_BARREL_DEBUG  
    if(itrust) {
    cout<< " MuUpdator::r, z= "<<(*ihit).recHit().globalPosition().perp()<<" "<<(*ihit).recHit().globalPosition().z()<<endl;
    cout<<" GlobalPosition ="<<(*ihit).recHit().globalPosition()<<" theta="<<theta<<" tan(theta)="<<tan(theta)<<endl;
    cout<<" Global Position error="<<(*ihit).recHit().globalPositionError().phierr((*ihit).recHit().globalPosition())
    <<" "<<(*ihit).recHit().globalPositionError().rerr((*ihit).recHit().globalPosition())<<
    " "<<(*ihit).recHit().globalPositionError().czz()<<endl;
    cout<<"Cartezian error= "<<(*ihit).recHit().globalPositionError().cxx()<<
                          " "<<(*ihit).recHit().globalPositionError().cyy()<<
			  " "<<(*ihit).recHit().globalPositionError().czz()<<endl;
    }			  
#endif 
    double phierror=sqrt((*ihit).recHit().globalPositionError().phierr((*ihit).recHit().globalPosition()));
    
    if(abs(phierror)<0.0000001) {
#ifdef UPDATOR_BARREL_DEBUG 
    if(itrust) {   
    cout<<"Phierror rhit problem, correction has made="<<phierror<<endl;
    }
#endif
    phierror=0.00008;
    }       
    ehitphi.push_back(phierror);
    
    if((*ihit).layer()->part()==barrel){
    ehitstrip.push_back(sqrt((*ihit).recHit().globalPositionError().czz()));
    } else{    
    ehitstrip.push_back(sqrt((*ihit).recHit().globalPositionError().rerr((*ihit).recHit().globalPosition()))/tan(theta)); 
    }   

  }
  
  phihit.push_back(nRecHit.globalPosition().phi());
  rhit.push_back(nRecHit.globalPosition().perp());
  zhit.push_back(nRecHit.globalPosition().z()); 
  ehitphi.push_back(sqrt(nRecHit.globalPositionError().phierr(nRecHit.globalPosition())));

  if(nRecHit.det().detUnits()[0]->type().part()==barrel){
    ehitstrip.push_back(sqrt(nRecHit.globalPositionError().czz())); 
  } else {
    ehitstrip.push_back(sqrt(nRecHit.globalPositionError().rerr(nRecHit.globalPosition()))/tan(theta));
  }
  
// add vertex 
    
  rhit.push_back(rvert);
  zhit.push_back(zvert); 
  ehitstrip.push_back(ezvert);    
   
  for(vector<double>::const_iterator iphi=phihit.begin();iphi!=phihit.end()-1;iphi++){
  double dpnew=abs(*iphi-*(iphi+1));
  if(dpnew>pi) dpnew=twopi-dpnew;
  
#ifdef UPDATOR_BARREL_DEBUG
  if(itrust) {
  cout<<" dphi=dpnew="<<dpnew<<" "<<*iphi<<" "<<*(iphi+1)<<endl;
  }
#endif

  dphihit.push_back(dpnew);  
  }
  
  for(vector<double>::const_iterator ir=rhit.begin();ir!=rhit.end()-2;ir++){
  double dpnew=abs(*ir-*(ir+1));
  
#ifdef UPDATOR_BARREL_DEBUG
  if(itrust) { 
  cout<<" dr=dpnew="<<dpnew<<" "<<*ir<<" "<<*(ir+1)<<endl;
  }
#endif

  drhit.push_back(dpnew);  
  }
  for(vector<double>::const_iterator iz=zhit.begin();iz!=zhit.end()-2;iz++){
  double dpnew=*iz-*(iz+1);
  
#ifdef UPDATOR_BARREL_DEBUG
  if(itrust) {
  cout<<" dZ=dpnew="<<dpnew<<endl;
  }
#endif
  
  dzhit.push_back(abs(dpnew));
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
if ( (*(MTM.begin())).layer()->part()==barrel){
  TrajectoryStateOnSurface tsos=updateBarrel(rhit, zhit, dphihit, drhit, ehitstrip, dehitphi, pRecHit, nRecHit, 
                                                                 nTsos, chirz, chirf, tType);
	
	if(!tsos.isValid()) {
#ifdef UPDATOR_BARREL_DEBUG_TRUE
         if(itrust){
	 cout<<"****** MuUpdator::updateBarrel is failed "<<endl;
         }
#endif	
	 return badtsos;
	}							

#ifdef UPDATOR_BARREL_DEBUG_TRUE
  if(itrust){  
  cout<<"MuUpdator::pz differ="<<tsos.freeTrajectoryState()->parameters().momentum().z()
      <<" Pzold="<<pold.z()<<" phinew= "<<tsos.freeTrajectoryState()->parameters().position().phi()
      <<" Phimeas= "<<nRecHit.globalPosition().phi()
      <<" znew= "<<tsos.freeTrajectoryState()->parameters().position().z()
      <<" zmeas= "<<nRecHit.globalPosition().z()  
      <<" Ptold="<<pold.perp()<<" Ptnew="<<tsos.freeTrajectoryState()->parameters().momentum().perp()
      <<endl;
   }
#endif

	
#ifdef UPDATOR_BARREL_DEBUG_TRUE
        if(itrust) {
        cout<<" MuUpdator::dfi,dz "<<nRecHit.globalPosition().phi()-
        tsos.freeTrajectoryState()->parameters().position().phi()<<" "
          <<nRecHit.globalPosition().z()-tsos.freeTrajectoryState()->parameters().position().z()<<" "<<endl;	   
        cout<<" UpdateBarrel:Chirz,chirf "<<chirz<<" "<<chirf<<endl;
	}
#endif	
//     if(diffEst ==0) return badtsos;
    
          
   return tsos;

} else{
  TrajectoryStateOnSurface tsos=updateEndcap(rhit, zhit, dphihit, dzhit, ehitstrip, dehitphi, pRecHit, nRecHit, 
                                                        nTsos, chirz, chirf, tType);
#ifdef UPDATOR_BARREL_DEBUG_TRUE 
   if(itrust) { 
  cout<<"pz differ="<<tsos.freeTrajectoryState()->parameters().momentum().z()<<
  " Pzold="<<pold.z()<<" phinew= "<<tsos.freeTrajectoryState()->parameters().position().phi()
  << " Phimeas= "<<nRecHit.globalPosition().phi()<<" Ptold="<<
  pold.perp()<<" Ptnew="<<tsos.freeTrajectoryState()->parameters().momentum().perp()<<endl;
   cout<<" UpdateEndcap:Chirz,chirf "<<chirz<<" "<<chirf<<endl; 
  }
#endif	
//   cout<<" updateEndcap "<<endl;

         
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
#endif

if(x.size() != y.size()) return fit;
if(x.size() != err.size()) return fit;


for (int i=0;i<x.size();i++){
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

if(abs(det)<0.00000001) return fit;
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

if(abs(x[x.size()-2]-x[x.size()-1])<0.1){

#ifdef LINEFIT_DEBUG
cout<<" Redetermine line - 90 degree "<<endl;
#endif

if(abs(x[x.size()-2]-x[x.size()-1])>0.0001){
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
for (int i=0;i<x.size();i++){
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

for (int i=0;i<x.size();i++){
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
for (int i=0;i<x.size();i++){
chi2=chi2+(y[i]-co1*x[i])*(y[i]-co1*x[i])/(err[i]*err[i]);
}
chi2=chi2/x.size();
#ifdef LINEFIT_DEBUG
cout<<"linefit1::chi2="<<chi2<<endl;
#endif
return fit=true;
}


double
        HICMuonUpdator::findPhiInVertex(const FreeTrajectoryState& fts, const double& rc, const Det* det) const{
     double acharge=fts.parameters().charge();
     double phiclus=fts.parameters().position().phi();
     double psi;
   if(det->detUnits()[0]->type().part()==barrel){
     double xrclus=fts.parameters().position().perp();
     double xdouble=xrclus/(2.*rc);
     psi= phiclus+acharge*asin(xdouble);
   } else {
     double zclus=fts.parameters().position().z();
     double pl=fts.parameters().momentum().z(); 
     psi=phiclus+acharge*0.006*abs(zclus)/abs(pl);     
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
						 const RecHit& pRecHit, const RecHit& nRecHit, 
						 const TrajectoryStateOnSurface& nTsos,
						 double& chirz, double& chirf, int& tType) const{

// fit in (dphi dr), (dphi-dz)
  TrajectoryStateOnSurface badtsos; 

//  cout<<" Update barrel begin "<<endl;

  double ch1,dphi,dr,ptnew;
  double co1,co2,co1n,chirz1;
  bool fitrf,fitrz,fitrz1;
  
// fit in (ZR)-coordinate 

  fitrz=this->linefit2(rhit,zhit,ehitstrip,co1,co2,chirz);
  
#ifdef UPDATOR_BARREL_DEBUG  
  cout<<"UPDATE::BARREL::line fit rz= "<<fitrz<<" chirz="<<chirz<<endl;
  cout<<" co1="<<co1<<" co2="<<co2<<endl;
#endif  

  if(!fitrz) return badtsos;
  
  if(dphihit.size()>1){
  fitrf=this->linefit1(dphihit,drhit,dehitphi,ch1,chirf);
  
#ifdef UPDATOR_BARREL_DEBUG
  cout<<"UPDATE::BARREL::line fit dphi= "<<fitrf<<" chirf="<<chirf<<endl;
  cout<<" ch1="<<ch1<<endl;
#endif

  if(!fitrf) return badtsos;
  }else{

  chirf = 0.;
  dphi=abs(dphihit.back());
  dr=abs(drhit.back());
  if(dphi > pi) dphi = twopi-dphi;
  ch1=dphi/dr;
  
#ifdef UPDATOR_BARREL_DEBUG
  cout<<"UPDATE::BARREL::line calc dphi= "<<dphi<<" dr="<<dr<<" chirf="<<chirf<<endl;
  cout<<" ch1="<<ch1<<endl;
#endif

  }
  
// Updating trajectory
  ptnew=0.006/ch1;
  GlobalPoint xrhit = nRecHit.globalPosition();
  double aCharge = nTsos.freeTrajectoryState()->parameters().charge();
  double phiclus=xrhit.phi();
  double xrclus=xrhit.perp();
  double xzclus=xrhit.z();
  double rc=100.*ptnew/(0.3*4.);
  double xdouble=xrclus/(2.*rc);
  double psi=phiclus-aCharge*asin(xdouble);
  double pznew=ptnew/co1;
  double znew=(xrclus-co2)/co1;
  double delphinew=abs(0.006*drhit.back()/ptnew);
  double phinew=pRecHit.globalPosition().phi()+aCharge*delphinew;
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
                           GlobalTrajectoryParameters(xnew, pnew, aCharge),
                           CurvilinearTrajectoryError(m), nTsos.surface());
			   
//  cout<<" Update barrel end  "<<xnew<<endl;
  			   
  return tsos;						
}
TrajectoryStateOnSurface HICMuonUpdator::updateEndcap(vector<double>& rhit, vector<double>& zhit, 
                                                 vector<double>& dphihit, vector<double>& drhit, 
	                                         vector<double>& ehitstrip, vector<double>& dehitphi,
						 const RecHit& pRecHit, const RecHit& nRecHit, 
						 const TrajectoryStateOnSurface& nTsos,
						 double& chirz, double& chirf, int& tType) const{

// fit in (dphi dr), (dphi-dz)
  TrajectoryStateOnSurface badtsos;

//    cout<<" Update endcap begin "<<endl;

  double ch1,dphi,dr;
  double co1,co2,co1n,chirz1;
  bool fitrf,fitrz,fitrz1;
  
#ifdef UPDATOR_ENDCAP_DEBUG
  cout<<"updateEndcap switched on"<<endl;
#endif
  
// fit in (ZR)-coordinate 

  fitrz=this->linefit2(rhit,zhit,ehitstrip,co1,co2,chirz);

#ifdef UPDATOR_ENDCAP_DEBUG  
  cout<<"line fit rz= "<<fitrz<<" chirz="<<chirz<<endl;
  cout<<" co1="<<co1<<" co2="<<co2<<endl;
#endif  

  if(!fitrz) return badtsos;
  
  if(dphihit.size()>1){
  fitrf=this->linefit1(dphihit,drhit,dehitphi,ch1,chirf);
  if(zhit.front()<0.) ch1=-1.*ch1;
#ifdef UPDATOR_ENDCAP_DEBUG
  cout<<"MuUpdate::barrel::line fit dphi= "<<fitrf<<" chirf="<<chirf<<endl;
  cout<<" ch1="<<ch1<<endl;
#endif
  if(!fitrf) return badtsos;
  }else{
  dphi=abs(dphihit.back());
  dr=abs(drhit.back());
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
  GlobalPoint xrhit = nRecHit.globalPosition();
//  cout<<" point 2 "<<endl;
  double aCharge = nTsos.freeTrajectoryState()->charge();
//  cout<<" point 3 "<<endl;
  double phiclus=xrhit.phi();
//  cout<<" point 4 "<<endl;
  double xzclus=xrhit.z();
//  cout<<" point 5 "<<endl;
  double psi=phiclus-aCharge*0.006*abs(xzclus-zvert)/abs(pznew);
//  cout<<" point 6 "<<endl;
  double ptnew=pznew*co1;
//  cout<<" point 7 "<<endl;
  double xrclus=co1*xzclus+co2;
//  cout<<" point 8 "<<endl;
  double delphinew=abs(0.006*drhit.back()/pznew);
//  cout<<" point 9 "<<endl;
  double phinew=pRecHit.globalPosition().phi()+aCharge*delphinew;
//  cout<<" point 10 "<<endl;
  GlobalVector pnew(ptnew*cos(psi),ptnew*sin(psi),pznew);
//  cout<<" point 11 "<<endl;
  GlobalPoint xnew(xrclus*cos(phinew),xrclus*sin(phinew),xzclus);
  
#ifdef UPDATOR_ENDCAP_DEBUG  
  cout<< "MuUpdator::xnew=" << xnew<<endl;
#endif  

    
  AlgebraicSymMatrix m(5,0);        
  m(1,1)=pznew; m(2,2)=thePhiWin, 
  m(3,3)=theZWin, 
  m(4,4)=thePhiWin, 
  m(5,5)=theZWin;
       
  TrajectoryStateOnSurface tsos(
                           GlobalTrajectoryParameters(xnew, pnew, aCharge),
                           CurvilinearTrajectoryError(m), nTsos.surface());
			   
//  cout<< "Update endcap end "<<endl;			   
  return tsos;						
}





