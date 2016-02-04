
#include "DetectorDescription/Base/interface/DDdebug.h"
// GENERATED FILE. DO NOT MODIFY!
#include "global_angular.h"
// I tried the following on CERN lxplus and still no MAX_DOUBLE was defined.
// so I tried DBL_MAX which does exist, but I do not know the source of this.
// So, in keeping with everything else I saw:
#include <cfloat>
#define MAX_DOUBLE DBL_MAX
//#include <climits>

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <Math/RotationZ.h>


 //your code here 


// always the same ctor
global_angular_0::global_angular_0(AlgoPos * a,std::string label)
 : AlgoImpl(a,label),
   rotate_(0), center_(3), rotateSolid_(0),
   alignSolid_(true), n_(1), startCopyNo_(1), incrCopyNo_(1),
   startAngle_(0), rangeAngle_(360.*deg)
{ 
  DCOUT('A', "Creating angular label=" << label);
}

global_angular_0::~global_angular_0() 
{ }


DD3Vector fUnitVector(double theta, double phi)
{
  return DD3Vector(cos(phi)*sin(theta),
                     sin(phi)*sin(theta),
		     cos(theta));
}


bool global_angular_0::checkParameters() 
{
  bool result = true;
  
  planeRot_   = DDRotationMatrix();
  solidRot_   = DDRotationMatrix();
  
  radius_     = ParE_["radius"][0];
  
  startAngle_ = ParE_["startAngle"][0];
  rangeAngle_ = ParE_["rangeAngle"][0];
  n_          = int(ParE_["n"][0]);
  startCopyNo_ = int(ParE_["startCopyNo"][0]);
  incrCopyNo_ = int(ParE_["incrCopyNo"][0]);
  
  if (fabs(rangeAngle_-360.0*deg)<0.001*deg) { // a full 360deg range
    delta_    =   rangeAngle_/double(n_);
  }
  else {
    if (n_ > 1) {
    delta_    =   rangeAngle_/double(n_-1);
    }
    else {
      delta_ = 0.;
    }
  }  
  
  DCOUT('a', "  startA=" << startAngle_/deg << " rangeA=" << rangeAngle_/deg <<
             " n=" << n_ << " delta=" << delta_/deg);

  //======= collect data concerning the rotation of the solid 
  typedef parE_type::mapped_type::size_type sz_type;
  sz_type sz = ParE_["rotateSolid"].size();
  rotateSolid_.clear();
  rotateSolid_.resize(sz);
  if (sz%3) {
    err_ += "\trotateSolid must occur 3*n times (defining n subsequent rotations)\n";
    err_ += "\t  currently it appears " + d2s(sz) + " times!\n";
    result = false;
  }
  for (sz_type i=0; i<sz; ++i) {
    rotateSolid_[i] = ParE_["rotateSolid"][i];
  }
  for (sz_type i=0; i<sz; i += 3 ) {
    if ( (rotateSolid_[i] > 180.*deg) || (rotateSolid_[i] < 0.) ) {
      err_ += "\trotateSolid \'theta\' must be in range [0,180*deg]\n";
      err_ += "\t  currently it is " + d2s(rotateSolid_[i]/deg) 
            + "*deg in rotateSolid[" + d2s(double(i)) + "]!\n";
      result = false;	    
    }
    DDAxisAngle temp(fUnitVector(rotateSolid_[i],rotateSolid_[i+1]),
		     rotateSolid_[i+2]);
    DCOUT('a', "  rotsolid[" << i <<  "] axis=" << temp.Axis() << " rot.angle=" << temp.Angle()/deg);
    solidRot_ = temp*solidRot_;			  
  }
  //  DCOUT('a', "  rotsolid axis=" << solidRot_.getAxis() << " rot.angle=" << solidRot_.delta()/deg);			    
  
  
  //======== collect data concerning the rotation of the x-y plane
  sz = ParE_["rotate"].size();
  rotate_.clear();
  rotate_.resize(sz);
  if (sz%3) {
    err_ += "\trotate must occur 3*n times (defining n subsequent rotations)\n";
    err_ += "\t  currently it appears " + d2s(sz) + " times!\n";
    result = false;
  }
  for (sz_type i=0; i<sz; ++i) {
    rotate_[i] = ParE_["rotate"][i];
  }
  for (sz_type i=0; i<sz; i += 3 ) {
    if ( (rotate_[i] > 180.*deg) || (rotate_[i] < 0) ) {
      err_ += "\trotate \'theta\' must be in range [0,180*deg]\n";
      err_ += "\t  currently it is " + d2s(rotate_[i]/deg) 
            + "*deg in rotate[" + d2s(double(i)) + "]!\n";
      result = false;	    
    }  
    DDAxisAngle temp(fUnitVector(rotateSolid_[i],rotateSolid_[i+1]),
		     rotateSolid_[i+2]);
    DCOUT('a', "  rotsolid[" << i <<  "] axis=" << temp.Axis() << " rot.angle=" << temp.Angle()/deg);
    planeRot_ =  planeRot_*temp;
  }
  //  DCOUT('a', "  rotplane axis=" << planeRot_.getAxis() << " rot.angle=" << planeRot_.delta()/deg);

  center_[0]      = ParE_["center"][0];
  center_[1]      = ParE_["center"][1];
  center_[2]      = ParE_["center"][2];
  
  if (ParS_["alignSolid"][0] != "T") {
    DCOUT('a', "  alignSolid = false");
    alignSolid_ = false;
  }
  else {
    alignSolid_ = true;
  }  
  
  return result;
}


DDTranslation global_angular_0::translation()  
{
  double angle = startAngle_+ double(count_-1)*delta_;
  
  DD3Vector v = fUnitVector(90*deg,angle)*radius_ ;
  
  if (rotate_[2]!=0) {
    //v = planeRot_.inverse()*v;
    v = planeRot_*v;
  }
  
  v += DD3Vector(center_[0], center_[1], center_[2]); // offset
  
  DCOUT('A', "  angle=" << angle/deg << " translation=" << v << "  count_=" << count_);
  return v;
}


DDRotationMatrix global_angular_0::rotation()
{
  //your code here
  DDRotationMatrix rot = solidRot_;

  if (alignSolid_) { // rotate the solid as well
    double angle = startAngle_+ double(count_-1)*delta_;
    ROOT::Math::RotationZ r2(angle);
    rot = r2*rot;
  }
  // DCOUT('A', "  rot.axis=" << rot.getAxis() << " rot.angle=" << rot.delta()/deg);

  if (rotate_[2]!=0) {
    rot = planeRot_*rot;
    //rot = rot*planeRot_.inverse();
    //rot = planeRot_.inverse()*rot;
  }

  return rot;
}

// optional, not in the XML, omitted.
int global_angular_0::copyno() const
{
  // for the moment rely on the automatic copy-number generation
  //  ( copy-no == invocation count count_ )
  int factor = AlgoImpl::copyno() - 1;
  return startCopyNo_ + factor*incrCopyNo_;
}



// optional, not in the XML, omitted.
void global_angular_0::checkTermination()
{
 if ( (n_-count_) == -1 ) terminate();
}


void global_angular_0::stream(std::ostream & os) const
{
  os << "global_angular_0::stream(): not implemented.";
}


/***********************************************************************************/

/**************************************************************************
 
 The following Code gets only generated IF the code-generator
 has the capability of generating evaluation/checking code for
 the specification of the parameters in the algorithm XML.
 
 If the following is not generated, there will be not automatic
 checking of the basic properties of the user-supplied parameters
 (bounds, occurences, default values, ...)
 
***************************************************************************/

// IT IS ADVISABLE TO HAVE SCHEMA VALIDATION DURING PARSING THE ALGORITHM-XML
// IN ORDER TO GET DEFAULT VALUES & CONTRAINTS FOR THE PARAMTERS OF THE ALGORITHM
// The code-generator has to fill data-structures containing the parameter names
// and their constraints. This information is only completely available during
// parsing of an algorithm-defintion-xml when schema validation is turned on.
global_angular_Check::global_angular_Check()
{
  // for the time being only expression-valued parameters can be automatically checked
  // against their specified constraints
  
  // schema default values will be shown if necessary in the following XML comments 
  // on the second line. The fist shows the values as given in the instance document

  // expressions have to be converted into doubles. No variables [bla] shall be allowed
  // inside the expressions; SystemOfUnits-symbols are the only supported ones.
  

  constraintsE_["startAngle"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					0.,  // min
					360.0*deg,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0.0
                                       );

  constraintsE_["rangeAngle"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					0.,  // min
					360.0*deg,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					360.0*deg
                                       );

  constraintsE_["n"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					1,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					1
                                       );
  constraintsE_["startCopyNo"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					1
                                       );

  constraintsE_["incrCopyNo"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					1
                                       );
				       

  constraintsE_["radius"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					0.,  // min
					+MAX_DOUBLE,  // max 
					true,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					2
                                       );

  constraintsS_["alignSolid"] = ConstraintS( 1,      // minOccurs
                                        1,      // maxOccurs
					
					 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					"T"
                                       );

  constraintsE_["center"] = ConstraintE( 3,      // minOccurs
                                        3,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0.
                                       );

  constraintsE_["rotate"] = ConstraintE( 3,      // minOccurs
                                        9,      // maxOccurs
					-360.0*deg,  // min
					360.0*deg,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0.0
                                       );

  constraintsE_["rotateSolid"] = ConstraintE( 3,      // minOccurs
                                        9,      // maxOccurs
					-360.0*deg,  // min
					360.0*deg,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0.0
                                       );

}


