#include "RecoHIMuon/HiMuPropagator/interface/FastMuPropagator.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Vector/ThreeVector.h"
#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <cmath>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>

//#define PROPAGATOR_DB

using namespace std;
namespace cms {
TrajectoryStateOnSurface
             FastMuPropagator::propagate(const FreeTrajectoryState& fts,
  			                 const Cylinder& boundcyl) const
{
  TrajectoryStateOnSurface badtsos;

  if(!checkfts(fts)) return badtsos;

#ifdef PROPAGATOR_DB 
  cout<<"FastMuPropagator::propagate::Start propagation in barrel::zvert "<<theFmpConst->zvert<<endl;
#endif    

  // Extract information from muchambers

  int charge;
  int imin0,imax0;
  double ptmax,ptmin,pt,phi,theta,theta0,z,pl;
  double dfcalc,phnext,bound;
  float ptboun=theFmpConst->ptboun;
  float step=theFmpConst->step;
  
  GlobalVector moment=fts.parameters().momentum();
  GlobalPoint posit=fts.parameters().position();
  pt=moment.perp();
  theta0=moment.theta();
// Correct theta
          double zfts=fts.parameters().position().z()-theFmpConst->zvert;
          double rfts=fts.parameters().position().perp();
  theta = atan2(rfts,zfts);

#ifdef PROPAGATOR_DB
  cout<<"FastMuPropagator::propagate::Start propagation in barrel::theta old, new "<<theta0<<" "<<theta<<endl;
#endif

  phi=posit.phi();
  ptmax=pt+fts.curvilinearError().matrix()(1,1);
  ptmin=pt-fts.curvilinearError().matrix()(1,1);
  bound=boundcyl.radius();
  charge=fts.parameters().charge();

  imax0=(int)((ptmax-ptboun)/step)+1;
  imin0=(int)((ptmin-ptboun)/step)+1;
#ifdef PROPAGATOR_DB     
  cout<<"FastMuPropagator::Look ptboun,step,imin0,imax0="<<ptboun<<" "<<step<<
    " "<<imin0<<" "<<imax0<<endl;
  cout<<"FastMuPropagator::Parameters (pt,theta,phi,ptmax,ptmin,bound)="<<pt<<" "
      <<theta<<" "<<phi<<" "<<ptmax<<" "<<ptmin<<" "<<bound<<endl;	 
#endif	 
  if(imax0 > 1){
    if(imin0<1) imin0=1;
    // ========================== phi ========================================    
    // new parametrisation (because of second mu-station)
    //

#ifdef PROPAGATOR_DB
    cout<<"FastMuPropagator::New parameters="<<theFmpConst->newparam[0]<<endl;
    cout<<"FastMuPropagator::New parameters for pt>40="<<
           theFmpConst->newparamgt40[0]<<endl;
#endif

    if(pt<40.) {
      dfcalc=charge/(theFmpConst->newparam[0]+theFmpConst->newparam[1]*pt
             +theFmpConst->newparam[2]*pt*pt);	     
    }else{
      dfcalc=charge/(theFmpConst->newparamgt40[0]+
      theFmpConst->newparamgt40[1]*pt+theFmpConst->newparamgt40[2]*pt*pt);
    }
//
// !!!! temporarily decision till new parametrization appears.
//    
    if (abs(dfcalc) > 0.6 ) dfcalc = charge*0.6;
    
    phnext=dfcalc+phi;
    if(phnext>=twopi) phnext=phnext-twopi;
    if(phnext <0.) phnext=twopi+phnext;
#ifdef PROPAGATOR_DB    
    cout<<"FastMuPropagator::Phi in Muon Chamb="<<phi<<endl;
    cout<<"FastMuPropagator::Phnext determined="<<phnext<<" dfcalc="<<dfcalc<<endl;
#endif    
    // ==========================  Z  ========================================    
    if(abs(theta-pi/2.)>0.00001){
      z=bound/tan(theta)+theFmpConst->zvert;
      if(fabs(z)>140) {
// Temporary implementation =====      
//         if(z>0.) z = z-45.;
//	 if(z<0.) z = z+45.;
// ==============================	 
      }
    }else{
      z=0.;
    }
#ifdef PROPAGATOR_DB    
    cout<<"FastMuPropgator::Z coordinate="<<z<<"bound="<<bound<<"theta="<<theta<<" "<<
      abs(theta-pi/2.)<<endl;
#endif
    // ====================== fill global point/vector =======================    
    GlobalPoint aX(bound*cos(phnext),bound*sin(phnext),z);
    if(abs(theta-pi/2.)<0.00001){
      pl=0.;
    }else{
      pl=pt/tan(theta);
    }
    // Recalculate momentum with some trick...
    double dfcalcmom=charge*theFmpConst->partrack*bound/pt;    
    GlobalVector aP(pt*cos(phnext-dfcalcmom),pt*sin(phnext-dfcalcmom),pl);
     
#ifdef PROPAGATOR_DB
    cout<<"phnextmom="<<phnext-dfcalcmom<<endl;
    cout<<"Cylinder aP="<<aP<<endl;
    cout<<"Before propagation="<<pt*cos(phi)<< " "<<pt*sin(phi)<<" "<<pl<<endl;
#endif       
    // =======================================================================    
    
    GlobalTrajectoryParameters gtp(aX,aP,charge,field);
    AlgebraicSymMatrix55 m;
    int iwin;
    float awin,bwin,phbound;
    
    if(pt>40.) {
      iwin=8;
      phbound=theFmpConst->phiwinb[7];
    }else{
      iwin=(int)(pt/theFmpConst->ptstep);
      awin=(theFmpConst->phiwinb[iwin+1]-theFmpConst->phiwinb[iwin])
               /(theFmpConst->ptwmax[iwin]-theFmpConst->ptwmin[iwin]);
      bwin=theFmpConst->phiwinb[iwin]-awin*theFmpConst->ptwmin[iwin];
      phbound=awin*pt+bwin;

    }
#ifdef PROPAGATOR_DB
    cout<<"Size of window in phi="<<phbound<<" "<<iwin<<endl;
    cout<<theFmpConst->phiwinb[iwin+1]<<" "<<theFmpConst->phiwinb[iwin]
            <<" "<<theFmpConst->ptwmin[iwin]<<
      " "<<theFmpConst->ptwmax[iwin]<<" awin="<<awin<<" bwin="<<bwin<<endl;
#endif    
    m(0,0)=(ptmax-ptmin)/6.; m(1,1)=phbound/theFmpConst->sigf;
         m(2,2)=theFmpConst->zwin/theFmpConst->sigz;
    m(3,3)=phbound/(2.*theFmpConst->sigf);m(4,4)=theFmpConst->zwin/(2.*theFmpConst->sigz); 
    
    CurvilinearTrajectoryError cte(m);

    FreeTrajectoryState fts(gtp,cte);
    TrajectoryStateOnSurface tsos(gtp, CurvilinearTrajectoryError(m), boundcyl);
    return tsos;			   
    
  }
  return badtsos;
}


TrajectoryStateOnSurface
             FastMuPropagator::propagate(const FreeTrajectoryState& fts,
  			             const Plane& boundplane) const
				     
{
  TrajectoryStateOnSurface badtsos;

#ifdef PROPAGATOR_DB 
  cout<<"FastMuPropagator::propagate::Start propagation in forward::zvert "<<theFmpConst->zvert<<endl;
#endif    
  if(!checkfts(fts)) return badtsos;
   
  // Extract information from muchambers

  int imin0,imax0;
  double ptmax,ptmin,pt,phi,theta,theta0,z,r,pl,plmin,plmax;
  double dfcalc,phnext;
  TrackCharge charge;
  float ptboun=theFmpConst->ptboun;
  float step=theFmpConst->step;

  
  GlobalVector moment=fts.parameters().momentum();
  GlobalPoint posit=fts.parameters().position();
  pt=moment.perp();
  theta0=moment.theta();
// Correct theta
          double zfts=fts.parameters().position().z()-theFmpConst->zvert;
          double rfts=fts.parameters().position().perp();
  theta = atan2(rfts,zfts);

#ifdef PROPAGATOR_DB
  cout<<"FastMuPropagator::propagate::Start propagation in forward::theta old, new "<<theta0<<" "<<theta<<endl;
#endif

  phi=posit.phi();
  ptmax=pt+fts.curvilinearError().matrix()(1,1);
  ptmin=pt-fts.curvilinearError().matrix()(1,1);
  pl=pt/tan(theta);
  plmin=ptmin/tan(theta);
  plmax=ptmax/tan(theta);
  imax0=(int)((ptmax-ptboun)/step)+1;
  imin0=(int)((ptmin-ptboun)/step)+1;
#ifdef PROPAGATOR_DB     
  cout<<"FastMuPropagator::Look ptboun,step,imin0,imax0="<<ptboun<<" "<<step<<
    " "<<imin0<<" "<<imax0<<endl;
  cout<<"FastMuPropagator::Parameters (pt,theta,phi,ptmax,ptmin)="<<pt<<" "
      <<theta<<" "<<phi<<" "<<ptmax<<" "<<ptmin<<endl;	 
#endif	 
  if(imax0 > 1){
    if(imin0<1) imin0=1;
  
    z=boundplane.position().z();
    r=(z-theFmpConst->zvert)*tan(theta);
    charge=fts.parameters().charge();

    // pl evaluation  
  
    dfcalc=theFmpConst->forwparam[0]+
                   charge*theFmpConst->forwparam[1]/abs(pl);  
    phnext=dfcalc+phi;
    if(phnext>=twopi) phnext=phnext-twopi;
    if(phnext <0.) phnext=twopi+phnext;
#ifdef PROPAGATOR_DB    
    cout<<"FastMuPropagator::Phi in Muon Chamb="<<phi<<endl;
    cout<<"FastMuPropagator::Phnext determined="<<phnext<<" dfcalc="<<dfcalc<<endl;
#endif    

    int iwin;
    float awin,bwin,phbound;
    AlgebraicSymMatrix55 m;
      
    if(pt>40.) {
      phbound=theFmpConst->phiwinf[7];
    }else{ // r < bound
      iwin=(int)(pt/theFmpConst->ptstep);
      awin=(theFmpConst->phiwinf[iwin+1]-theFmpConst->phiwinf[iwin])
             /(theFmpConst->ptwmax[iwin]-theFmpConst->ptwmin[iwin]);
      bwin=theFmpConst->phiwinf[iwin]-awin*theFmpConst->ptwmin[iwin];
      phbound=awin*pt+bwin;
    }
#ifdef PROPAGATOR_DB
    cout<<"Forward::Size of window in phi="<<phbound<<endl;
#endif    
    m(0,0)=abs(plmax-plmin)/6.; m(1,1)=phbound/theFmpConst->sigf;
          m(2,2)=theFmpConst->zwin/theFmpConst->sigz;
    m(3,3)=phbound/(2.*theFmpConst->sigf);m(4,4)=theFmpConst->zwin/(2.*theFmpConst->sigz); 
    
    GlobalPoint aX(r*cos(phnext),r*sin(phnext),z);
    
    // Recalculate momentum with some trick...    
    double dfcalcmom=charge*theFmpConst->partrack*abs(z)/abs(pl);
    GlobalVector aP(pt*cos(phnext-dfcalcmom),pt*sin(phnext-dfcalcmom),pl); 
#ifdef PROPAGATOR_DB
    cout<<"dfcalcmom="<<charge<<" "<<theFmpConst->partrack<<" "<<z<<" "<<pl<<" "<<phnext<<endl;
    cout<<"phnextmom="<<phnext-dfcalcmom<<endl;

    cout<<"Plane aP="<<aP<<endl;
    cout<<"Before propagation="<<pt*cos(phi)<< " "<<pt*sin(phi)<<" "<<pl<<endl;
#endif       
       
    GlobalTrajectoryParameters gtp(aX,aP,charge,field);
    CurvilinearTrajectoryError cte(m);
   
    TrajectoryStateOnSurface tsos(gtp, CurvilinearTrajectoryError(m), boundplane);
    return tsos;			   			   
  }
    
  return badtsos;
}
  bool
         FastMuPropagator::checkfts(const FreeTrajectoryState& fts) const {
	  bool check=true;
          double z=fts.parameters().position().z();
          double r=fts.parameters().position().perp();
          float mubarrelrad=theFmpConst->mubarrelrad;
          float muforwardrad=theFmpConst->muforwardrad;
          float muoffset=theFmpConst->muoffset;
	  mubarrelrad=350.; muforwardrad=400.;
#ifdef PROPAGATOR_DB
        cout<<"checkfts::r,z="<<r<<" "<<z<<" "<<check<<endl;
#endif	  
	  if(r<mubarrelrad-muoffset&&abs(z)<muforwardrad-muoffset){
	  check=false;
#ifdef PROPAGATOR_DB
        cout<<"checkfts::false="<<check<<endl;
#endif	  
	  }   
	  return check;
}
}
