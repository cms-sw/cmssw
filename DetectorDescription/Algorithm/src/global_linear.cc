
//#include "DetectorDescription/Base/interface/DDdebug.h"
// GENERATED FILE. DO NOT MODIFY!
#include "global_linear.h"
// I tried the following on CERN lxplus and still no MAX_DOUBLE was defined.
// so I tried DBL_MAX which does exist, but I do not know the source of this.
// So, in keeping with everything else I saw:
#define MAX_DOUBLE DBL_MAX
//#include <climits>




		#include <cmath>
		 
	                


// always the same ctor
global_linear_0::global_linear_0(AlgoPos * a,std::string label)
 : AlgoImpl(a,label)
{ }

global_linear_0::~global_linear_0() 
{ }

bool global_linear_0::checkParameters() 
{

		  bool result = true;   
		  
		  // check for valid delta-value
		  if (ParE_["delta"][0] == 0.) {
		    err_ += "\tdelta must not be 0\n";
		    result = false;
		  }
		  
		  // check for presence of base
		  if (!ParE_["base"].size()) {
		    result = false; // don't select this implementation, because base is missing
		  }  		  
		  
		  return result;  	      
	         
}


DDTranslation global_linear_0::translation()  
{

	         // we can safely fetch all parameters, because they
		 // have been checked already ...
		 double theta  = ParE_["theta"][0]/rad;
                 double phi    = ParE_["phi"][0]/rad;
                 double offset = ParE_["offset"][0];
                 double delta  = ParE_["delta"][0];
   
                 DDTranslation direction( sin(theta)*cos(phi),
                                          sin(theta)*sin(phi),
			                  cos(theta) );
	         
		 DDTranslation base(ParE_["base"][0],
		                    ParE_["base"][1],
				    ParE_["base"][2]);				  
   			    
                 return base + (offset + double(curr_)*delta)*direction;	      
		 
}


DDRotationMatrix global_linear_0::rotation()
{

	        return DDRotationMatrix();
	 	
}   







void global_linear_0::stream(std::ostream & os) const
{
  os << "global_linear_0::stream(): not implemented.";
}



		#include <cmath>
		 
	                


// always the same ctor
global_linear_1::global_linear_1(AlgoPos * a,std::string label)
 : AlgoImpl(a,label)
{ }

global_linear_1::~global_linear_1() 
{ }

bool global_linear_1::checkParameters() 
{

	         /* besides the automatic generated checking for the input params, 
		    we have to decide, whether the params are correct and should
		    select this implementation 
		 */
		  
		  bool result = true;   
		  
		  // check for valid delta-value
		  if (ParE_["delta"][0] == 0.) {
		    err_ += "\tdelta must not be 0\n";
		    result = false;
		  }
		  
		  // check for presence of base
		  if (ParE_["base"].size()) {
		    result = false; // don't select this implementation, because base is present
		  }  
		  
		  return result;  
	            
}


DDTranslation global_linear_1::translation()  
{

	         // we can safely fetch all parameters, because they
		 // have been checked already ...
		 double theta  = ParE_["theta"][0]/rad;
                 double phi    = ParE_["phi"][0]/rad;
                 double offset = ParE_["offset"][0];
                 double delta  = ParE_["delta"][0];
   
                 DDTranslation direction( sin(theta)*cos(phi),
                                          sin(theta)*sin(phi),
			                  cos(theta) );
   			    
                 return (offset + double(curr_)*delta)*direction;
		
}


DDRotationMatrix global_linear_1::rotation()
{

	        // there are no rotations involved in this algorithm.
		// simply returns the unit matrix.
		return DDRotationMatrix();
		
}   







void global_linear_1::stream(std::ostream & os) const
{
  os << "global_linear_0::stream(): not implemented.";
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
global_linear_Check::global_linear_Check()
{
  // for the time being only expression-valued parameters can be automatically checked
  // against their specified constraints
  
  // schema default values will be shown if necessary in the following XML comments 
  // on the second line. The fist shows the values as given in the instance document

  // expressions have to be converted into doubles. No variables [bla] shall be allowed
  // inside the expressions; SystemOfUnits-symbols are the only supported ones.
  

  constraintsE_["theta"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					0*deg,  // min
					180*deg,  // max 
					true,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					0.
                                       );

  constraintsE_["phi"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					0*deg,  // min
					360*deg,  // max 
					true,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					0.
                                       );

  constraintsE_["delta"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					true,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					0.
                                       );

  constraintsE_["offset"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0
                                       );

  constraintsE_["base"] = ConstraintE( 3,      // minOccurs
                                        3,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					0.
                                       );

}


