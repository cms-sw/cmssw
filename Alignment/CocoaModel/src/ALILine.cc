//   COCOA class implementation file
//Id:  ALILine.C
//CAT: Fit
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/ALILine.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h" 
#include "Alignment/CocoaModel/interface/ALIPlane.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <cstdlib>


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Constructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALILine::ALILine( const CLHEP::Hep3Vector& point, const CLHEP::Hep3Vector& direction ) 
{
  _point = point;
  _direction = direction* (1/direction.mag());
  if (ALIUtils::debug >= 9) ALIUtils::dump3v( _point, " ALILine point _point = ");
  if (ALIUtils::debug >= 9) ALIUtils::dump3v( _direction, " ALILine direction _direction = ");
}




CLHEP::Hep3Vector ALILine::intersect( const ALILine& l2, bool notParallel ) 
{ 
  
  if(ALIUtils::debug >= 3 ){
    std::cout << "***** ALILine::intersect (constructor) two lines: " << std::endl 
			<< " line1: " << *this << std::endl
			<< " line2: " << l2 << std::endl;
  }
  CLHEP::Hep3Vector inters; 
  //--- Check that they are not parallel 
  double acosvec = vec().dot(l2.vec()) / vec().mag() / l2.vec().mag();
  
  if(ALIUtils::debug >= 5 ){
  
        std::cout << "\t Determination of acosvec = vec().dot(l2.vec()) / vec().mag() / l2.vec().mag() "
	<< std::endl << std::endl;
//  	std::cout << "\t vec().dot = " << vec().dot << std::endl; 
	std::cout << "\t l2.vec() = " << l2.vec() << std::endl;

  	std::cout << "\t vec().mag() = " << vec().mag() << std::endl;
	std::cout << "\t l2.vec().mag() = " << l2.vec().mag() << std::endl << std::endl;
  
  			}
  
  if(ALIUtils::debug >= 5 ) std::cout << " acosvec = " << acosvec << std::endl;
  if( 1 - fabs(acosvec) < 1E-8 ) {
    if( notParallel ) {
      std::cerr << " !!!EXITING ALILine::intersect: two lines are parallel"
	<< std::endl;
      exit(1);
    } else {
      if(ALIUtils::debug >= 5 ) std::cout << " !!! ALILine::intersect: two lines are parallel (no errors)" << std::endl;
      //gcc2952      inters = CLHEP::Hep3Vector( DBL_MAX, DBL_MAX, DBL_MAX );
      inters = CLHEP::Hep3Vector( ALI_DBL_MAX, ALI_DBL_MAX, ALI_DBL_MAX );
    }
  } else { 

    
 //	****************************************************************	//  
 //	****************************************************************	//
 //	****************************************************************	//
 //  			     Determination of Fact				//
 //										//
 //	Problem :  3D quantity was determined by doing calculation with		//
 //		   the 2D projections of the std::vectors.  It is possible 		//
 //	   	   for projection in a particular plane to be 0 		//
 //										//
 //	Solution : Test for problem and redo calculation if necessary by	//
 //		   projecting into a different plane				//
 //	****************************************************************	//  
 //	****************************************************************	//
 //	****************************************************************	//
  
 
 	
    ALIdouble fact = ( vec().y() * l2.pt().x() - vec().x() * l2.pt().y()
       - vec().y() * pt().x() + vec().x() * pt().y() )
      / ( vec().x() * l2.vec().y() - vec().y() * l2.vec().x() );



  if(ALIUtils::debug >= 2 ){
       std::cout << std::endl << std::endl << "*** START CALC OF FACT ***" << std::endl;
       std::cout << "    ==================" << std::endl << std::endl;            
       std::cout << "*** Determination of fact ->";      
       std::cout << "\t fact = (" << vec().y() * l2.pt().x() - vec().x() * l2.pt().y()
       - vec().y() * pt().x() + vec().x() * pt().y() << "/";
       
       std::cout <<  vec().x() * l2.vec().y() - vec().y() * l2.vec().x() << ") = " << fact << std::endl;
 
        			}
	
  //    ALIdouble old_fact = fact;
    ALIdouble old_fact_denominator = vec().x() * l2.vec().y() - vec().y() * l2.vec().x();				
    //-    ALIdouble old_fact2 = 0.;
    ALIdouble old_fact_denominator2 = 999;
	       

   if(fabs(fact) > 1e8 || fabs(old_fact_denominator) < 1.e-10)
    {
      
      // do calculation by rotating 90 degrees into xy plane
      // Z-> X
      // X-> -Z
      
       if(ALIUtils::debug >= 2 && fabs(old_fact_denominator) < 1.e-10) std::cout << " ** Problem:  old_fact_denominator -> " << old_fact_denominator << std::endl;
       
       if(ALIUtils::debug >= 2 && fabs(fact) > 1e8) std::cout << " ** Problem: fact -> " << fact << std::endl;
       
       if(ALIUtils::debug >= 2 && (vec().x() * l2.vec().y() - vec().y() * l2.vec().x()) == 0) std::cout << " ** Division by 0 !!! " << std::endl;       
       if(ALIUtils::debug >= 2 ) std::cout << " ** Must rotate to yz plane for calculation (X-> Z) ";
 
    
    	fact = ( -1 * vec().y() * l2.pt().z() + vec().z() * l2.pt().y()
       	+ vec().y() * pt().z() - vec().z() * pt().y() )
      	/ ( -1*vec().z() * l2.vec().y() + vec().y() * l2.vec().z() );
      
        if(ALIUtils::debug >= 2 ) std::cout << "\t -- 1st Recalculation of fact in yz plane = " << fact << std::endl;  

        old_fact_denominator2 = -1*vec().z() * l2.vec().y() + vec().y() * l2.vec().z();

	
    	if(fabs(-1*vec().z() * l2.vec().y() + vec().y() * l2.vec().z())< 1.e-10)
     	{
	  
	  if(ALIUtils::debug >= 2 ) std::cout << " ** Must rotate to xz plane for calculation (Y-> -Z) ";
	    if(ALIUtils::debug >= 2 && fabs(old_fact_denominator2) < 1.e-10) std::cout << " ** Problem: old_fact_denominator2 -> " << old_fact_denominator2 << std::endl;       
	  //-       	   if(ALIUtils::debug >= 2 && fabs(old_fact2) > 1.e8) std::cout << " ** Problem: old_fact2 -> " << old_fact2 << std::endl;
	  
	  fact = ( -1*vec().z() * l2.pt().x() + vec().x() * l2.pt().z()
		   + vec().z() * pt().x() - vec().x() * pt().z() )
	    / ( -1*vec().x() * l2.vec().z() + vec().z() * l2.vec().x() );
	  
       	   if(ALIUtils::debug >= 2 ) std::cout << "\t -- 2nd Recalculation of fact in xz plane = " << fact << std::endl;  


    	}
	else {
	  if(ALIUtils::debug >= 2 ) std::cout << "*!* 2nd calculation sufficient" << std::endl;
	}
       
    }    
    else
    {

     if(ALIUtils::debug >= 2 ) std::cout << "*!* Standard calculation - things are fine" << std::endl;
     if(ALIUtils::debug >= 2 ) std::cout << "\t ----> fact = ( " << vec().y() * l2.pt().x() - vec().x() *
     l2.pt().y() - vec().y() * pt().x() + vec().x() * pt().y() << " / " << vec().x() * l2.vec().y() 
     - vec().y()* l2.vec().x() << " ) = " << fact << std::endl;    
    }

          
	  
 //	****************************************************************	//  
 //	****************************************************************	//
 //	****************************************************************	//
 //  			     	Finished With Fact					//
 //	****************************************************************	//  
 //	****************************************************************	//
 //	****************************************************************	//	  
	  
	  
     inters =  l2.pt() + fact * l2.vec();
     
     if(ALIUtils::debug >= 2 ){           
        std::cout << "Determination of intersection = l2.pt() + fact * l2.vec()" << std::endl << std::endl;
	ALIUtils::dump3v( l2.pt(), "\t --> l2.pt() = ");
  	std::cout << "\t --> fact = " << fact << std::endl;
	std::cout << "\t --> l2.vec() = " << l2.vec() << std::endl;
        ALIUtils::dump3v( inters, "***\t --> ALILine::intersection at: ");
     }

  }
 
  return inters;
 
}

std::ostream& operator<<(std::ostream& out, const ALILine& li)
{
  out << " ALILine point " << li._point << std::endl;
  out << " ALILine direc " << li._direction;

  return out;  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ intersect: Intersect a LightRay with a plane and change thePoint to the intersection point
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

CLHEP::Hep3Vector ALILine::intersect( const ALIPlane& plane, bool notParallel)
{
  if(ALIUtils::debug >= 5) std::cout << "***** ALILine::intersect WITH PLANE" << std::endl;
  if(ALIUtils::debug >= 4) {
    ALIUtils::dump3v( plane.point(), "plane point"); 
    ALIUtils::dump3v( plane.normal(), "plane normal"); 
  }
  
  //---------- Check that they intersect
  if( fabs( plane.normal()*_direction ) < 1.E-10 ) {
    std::cerr << " !!!! INTERSECTION NOT POSSIBLE: LightRay is perpendicular to plane " << std::endl;
    std::cerr << " plane.normal()*direction() = " << plane.normal()*_direction << std::endl;
    ALIUtils::dump3v( _direction, "LightRay direction ");
    ALIUtils::dump3v( plane.normal(), "plane normal ");
    exit(1);
  }

  //---------- Get intersection point between LightRay and plane
  //-  	if(ALIUtils::debug >= 4) std::cout << " ALILine::intersect WITH LightRay" << std::endl;
	
  CLHEP::Hep3Vector vtemp = plane.point() - _point;
  	if(ALIUtils::debug >= 4) ALIUtils::dump3v( vtemp, "n_r = point  - point_plane"); 
	
  ALIdouble dtemp = _direction * plane.normal();
  	if(ALIUtils::debug >= 4) std::cout << " lightray* plate normal" << dtemp << std::endl;
	
  if ( dtemp != 0. ) {
      dtemp = (vtemp * plane.normal()) / dtemp;
      if(ALIUtils::debug >= 4)  std::cout << " n_r*plate normal (dtemp) : " << dtemp << std::endl;
      if(ALIUtils::debug >= 4)  std::cout << " Old vtemp : " << vtemp << std::endl;
  	} else std::cerr << "!!! LightRay: Intersect With Plane: plane and light ray parallel: no intersection" << std::endl;     
  		
  vtemp = _direction * dtemp;
  	if(ALIUtils::debug >= 4) ALIUtils::dump3v( vtemp, "n_r scaled (vtemp) : "); 
  	if(ALIUtils::debug >= 4) ALIUtils::dump3v( CLHEP::Hep3Vector(dtemp), "dtemp analog to vtemp : "); 
    
  CLHEP::Hep3Vector inters = vtemp + _point;
  	if(ALIUtils::debug >= 4) ALIUtils::dump3v( inters, "intersection point = vtemp + _point"); 
  	if(ALIUtils::debug >= 4) ALIUtils::dump3v( vtemp, "vtemp =  "); 
  	if(ALIUtils::debug >= 4) ALIUtils::dump3v( _point, "_point =  "); 

  return inters;
}

