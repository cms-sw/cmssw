#include "RecoHIMuon/HiMuTracking/interface/HICTrajectoryCorrector.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

//#define CORRECT_DEBUG
using namespace std;

TrajectoryStateOnSurface
             HICTrajectoryCorrector::correct(FreeTrajectoryState& fts, FreeTrajectoryState& ftsnew,
  			                const GeomDet* det) const
  {
        double pi=4.*atan(1.);
        double twopi=8.*atan(1.);

#ifdef CORRECT_DEBUG
        std::cout<<" HICTrajectoryCorrector::correct::start "<<std::endl;
#endif
        double zvert=theHICConst->zvert;
#ifdef CORRECT_DEBUG
        std::cout<<"HICTrajectoryCorrector::zvert "<<zvert<<std::endl;
#endif
  	LocalVector lp(1,0,0); double adet,bdet;
	GlobalVector gp=det->toGlobal(lp);
	GlobalPoint gdet=det->position();
	double rdet=gdet.perp();
	double phidet=gdet.phi();
	double zdet=gdet.z();
	if(phidet<0.) phidet=phidet+twopi;
	double zpoint,a,b; 
	
// Parameters of line for trajectory correction from trajectory on previous layer

	if(fabs(fts.parameters().momentum().z())>0.0000001) 
	{ 
            a = fts.parameters().momentum().perp()/fts.parameters().momentum().z();
            b = -a*zvert;
	if( det->subDetector() == GeomDetEnumerators::PixelBarrel || 
                    det->subDetector() == GeomDetEnumerators::TIB || det->subDetector() == GeomDetEnumerators::TOB )
	{  
	    zpoint=(rdet-b)/a;
	}
	 else
	    {
	       zpoint=det->position().z();
	       double phinew=fts.parameters().position().phi();
	       if(phinew<0.) phinew=twopi+phinew;
	       double dfcalc=0.006*fts.parameters().charge()*fabs(zpoint-fts.parameters().position().z())/fabs(fts.parameters().momentum().z());
	 
//	 cout <<" MuUpdator::correct::forward phi "<< phinew<<" "<<dfcalc<<" "<<phinew+dfcalc<<endl; 

	       GlobalPoint xnew1(rdet*cos(phinew+dfcalc),rdet*sin(phinew+dfcalc),zpoint);
	
#ifdef CORRECT_DEBUG	
	cout<<" MuUpdator::correct::New forward r,phi,z= "<<xnew1.perp()<<" "<<xnew1.phi()<<" "<<xnew1.z()<<endl;
#endif	
                TrajectoryStateOnSurface tsos( 
                                   GlobalTrajectoryParameters(xnew1, ftsnew.parameters().momentum(), 
			           ftsnew.parameters().charge(), field),
                                   CurvilinearTrajectoryError(ftsnew.curvilinearError().matrix()),
	                           det->surface() );
	        return tsos;
	}
	
	} //  fts.parameters().momentum().z() > 0.0000001
	  else 
	  {
	        a=10000000.;
	        b=zvert;
	        zpoint=zvert;
	  }
	  
#ifdef CORRECT_DEBUG	
   cout<<" MuUpdator::correct::Detector::rdet= "<<rdet<<" Zdet= "<<zdet<<" Phidet= "<<phidet<<" Newz= "<<zpoint<<endl;
   cout<<" MuUpdator::correct::a,b= "<<a<<" "<<b<<endl;
   cout<<" MuUpdator::correct::detector= "<<gp.x()<<" "<<gp.y()<<" "<<gp.z()<<endl;
   cout<<" MuUpdator::correct::Detector::x,y,z "<<gdet.x()<<" "<<gdet.y()<<" "<<gdet.z()<<endl;
#endif
	
// detector parameters	
        if(fabs(gdet.x())>0.00001) 
	{
	  if(fabs(gdet.y())>0.00001)
	  {
	     adet=gp.y()/gp.x();
	     bdet=rdet*sin(phidet)-adet*rdet*cos(phidet);
	  } 
	    else 
	    {
	     adet=100000.;
	     bdet=gdet.x();  
	    }
	} 
	  else 
	  {
	     adet=0.;
	     bdet=gdet.y();
	}
		
	float width=det->surface().bounds().width();
	float tolerance=atan(width/rdet)/2.+0.001;

#ifdef CORRECT_DEBUG	
	cout<<"MuUpdator::correct::Tolerance="<<tolerance<<" Detector line::adet= "<<adet<<" bdet= "<<bdet<<endl;
#endif	
  TrajectoryStateOnSurface badtsos;
    
// calculation of intersection point.
         double rc,phic,xc,yc,ph1,ph2;
         double b1,c1,determinant,x1,x2,y1,y2;
// rc-radius of track,phic-phi coordinate of center
         rc=100.*(fts.parameters().momentum().perp())/(0.3*4.);
         double acharge=fts.parameters().charge();
         phic=findPhiInVertex(fts, rc, det);     
         xc=rc*cos(phic);
         yc=rc*sin(phic);

#ifdef CORRECT_DEBUG	
	cout<<"MuUpdator::correct::Momentum of track="<<fts.parameters().momentum().perp()<<
	" rad of previous cluster= "<<fts.parameters().position().perp()<<
	" phi of previous cluster="<<fts.parameters().position().phi()<<endl;
	cout<<"MuUpdator::correct::position of the previous cluster::r,phi,z="<<fts.parameters().position().perp()<<" "<<
	fts.parameters().position().phi()<<" "<<fts.parameters().position().z()<<endl;
	
	cout<<"MuUpdator::correct::radius of track="<<rc<<endl;
	cout<<"MuUpdator::correct::charge="<<acharge<<endl;
	cout<<"MuUpdator::correct::phic="<<phic<<endl;
        cout<<"MuUpdator::correct::xc,yc="<<xc<<" "<<yc<<endl;
#endif

    if(fabs(adet)>99999.) 
    {
       if(yc<0.)
       {
           acharge = 1.;
       }
	else
	{
           acharge = -1.;             
        } 
          x1 = bdet;
          y1 = yc + acharge*sqrt(rc*rc-(x1-xc)*(x1-xc));
	
	GlobalPoint xnew1(x1,y1,zpoint);
	
#ifdef CORRECT_DEBUG	
	cout<<" MuUpdator::correct::New r,phi,z= "<<xnew1.perp()<<" "<<xnew1.phi()<<" "<<xnew1.z()<<endl;
#endif
	
        TrajectoryStateOnSurface tsos( 
                                GlobalTrajectoryParameters(xnew1, ftsnew.parameters().momentum(), 
			        ftsnew.parameters().charge(), field),
                                CurvilinearTrajectoryError(ftsnew.curvilinearError().matrix()),
	                        det->surface());
	return tsos;
			   
                    
    }

   if(fabs(adet)<0.00001) 
   {
       if(xc<0.) 
       {
           acharge = 1;
       }
        else
	{
           acharge = -1;            
        } 
 
          y1=bdet;
          x1 = xc+acharge*sqrt(rc*rc-(y1-yc)*(y1-yc));

	
	  GlobalPoint xnew1(x1,y1,zpoint);
	
          TrajectoryStateOnSurface tsos( 
                           GlobalTrajectoryParameters(xnew1, ftsnew.parameters().momentum(), 
			   ftsnew.parameters().charge(), field),
                           CurvilinearTrajectoryError(ftsnew.curvilinearError().matrix()),
	                   det->surface());
	  return tsos;
                     
   }

          b1=2.*(adet*(bdet-yc)-xc)/(1.+adet*adet);
          c1=(bdet*bdet-2.*bdet*yc)/(1.+adet*adet);
	  
          determinant = b1*b1 - 4.*c1;
	  
#ifdef CORRECT_DEBUG	  
           cout<<"b1="<<b1<<"c1="<<c1<<endl;
#endif    	  
    if(determinant<0.) 
    { // problems with decision
#ifdef CORRECT_DEBUG    
          cout<<"HICTrajectoryCorrector::problem with decision::badtsos is returned::Determinant="<<determinant<<endl;
#endif
          return badtsos;
    }  

          x1 = (-b1+sqrt(determinant))/2.;
          x2 = (-b1-sqrt(determinant))/2.;
          y1 = adet*x1+bdet;
          y2 = adet*x2+bdet;
	 
	
	GlobalPoint xnew1(x1,y1,zpoint);
	GlobalPoint xnew2(x2,y2,zpoint);
	  
	  
	ph1=xnew1.phi();
	ph2=xnew2.phi();
	  
        if(ph1<0.) ph1 = ph1+twopi;
        if(ph2<0.) ph2 = ph2+twopi;
	double dfi1=fabs(ph1-phidet);
	double dfi2=fabs(ph2-phidet);
	if(dfi1>pi) dfi1=twopi-dfi1;
	if(dfi2>pi) dfi2=twopi-dfi1;
	
#ifdef CORRECT_DEBUG
       cout<<" MuUpdator::correct::Coordinates="<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<endl;
       cout<<" MuUpdator::correct::New phi="<<ph1<<" "<<ph2<<" phidet"<<phidet<<" dfi1="<<dfi1<<" dfi2="<<dfi2<<endl;
       cout<<" MuUpdator::correct::New r,phi,z= "<<xnew1.perp()<<" "<<xnew1.phi()<<" "<<xnew1.z()<<endl;
       
#endif

       if(dfi1<dfi2)
       {
#ifdef CORRECT_DEBUG
        cout<<" MuUpdator::correct::dfi1<dfi2= "<<dfi1<<" Tolerance "<<tolerance<<endl;
#endif
       if(dfi1<tolerance)
       {
#ifdef CORRECT_DEBUG
        cout<<" MuUpdator::correct::keep tsos "<<endl;
#endif      
             TrajectoryStateOnSurface tsos( 
                           GlobalTrajectoryParameters(xnew1, ftsnew.parameters().momentum(), 
			   ftsnew.parameters().charge(), field),
                           CurvilinearTrajectoryError(ftsnew.curvilinearError().matrix()),
	                   det->surface());

       return tsos;
       }
       }
         else
	 {
       
       if(dfi1>dfi2)
       {
       if(dfi2<tolerance)
       {       
            TrajectoryStateOnSurface tsos( 
                           GlobalTrajectoryParameters(xnew2, ftsnew.parameters().momentum(), 
			   ftsnew.parameters().charge(), field),
                           CurvilinearTrajectoryError(ftsnew.curvilinearError().matrix()),
	                   det->surface());
			   return tsos;
	}		   
        }
	}
#ifdef CORRECT_DEBUG	
       cout<<"HICTrajectoryCorrector::False detector::badtsos is returned"<<endl;
#endif	
        return badtsos;

  }

double HICTrajectoryCorrector::findPhiInVertex(const FreeTrajectoryState& fts, const double& rc, const GeomDet* det) const{
        double pi=4.*atan(1.);
        double twopi=8.*atan(1.);

     double acharge=fts.parameters().charge();
     double phiclus=fts.parameters().position().phi();
     double psi;
   if(det->subDetector() == GeomDetEnumerators::PixelBarrel || det->subDetector() == GeomDetEnumerators::TIB || det->subDetector() == GeomDetEnumerators::TOB){
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
	cout<<"phic="<<phic<<" pi="<<pi<<" pi2="<<twopi<<endl;
#endif
     
     return phic;
}

