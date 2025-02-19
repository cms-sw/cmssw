
//#include "DetectorDescription/Base/interface/DDdebug.h"
// GENERATED FILE. DO NOT MODIFY!
#include "global_simpleAngular.h"
// I tried the following on CERN lxplus and still no MAX_DOUBLE was defined.
// so I tried DBL_MAX which does exist, but I do not know the source of this.
// So, in keeping with everything else I saw:
#define MAX_DOUBLE DBL_MAX
//#include <climits>


#include "DetectorDescription/Core/interface/DDTransform.h"
//#include "CLHEP/Geometry/Transform3D.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <Math/RotationZ.h>
                                  

// always the same ctor
global_simpleAngular_0::global_simpleAngular_0(AlgoPos * a,std::string label)
 : AlgoImpl(a,label)
{ }

global_simpleAngular_0::~global_simpleAngular_0() 
{ }

bool global_simpleAngular_0::checkParameters() 
{
/* besides the automatic generated checking for the input params, 
     we have to decide, whether the params are correct and should
     select this implementation 
  */
		  
  bool result = true;   
		  
  // check for presence of delta
  if (ParE_["delta"].size() == 0) {
    result = false; // don't select this implementation, because delta is missing
  }
  else {
    // check for valid delta-value
    if (ParE_["delta"][0] == 0.) {
      err_ += "\tdelta can not be zero\n";
      result = false;
    }
  }

  // check for presence of number
  if (ParE_["number"].size() != 0) {
//    err_ += "\tcan not specify a delta and a number\n";
    result = false;
  }  
  return result;  
}


DDTranslation global_simpleAngular_0::translation()  
{
  double offset  = ParE_["offset"][0]/rad;
  double delta   = ParE_["delta"][0]/rad;
  double radius  = ParE_["radius"][0];
//  std::cout << "x = " << radius * cos(offset + delta * (count_ - 1)) << "  y = " << radius * sin(offset + delta * (count_ - 1)) << std::endl;
  DDTranslation trans(radius * cos(offset + delta * (count_ - 1)),
		      radius * sin(offset + delta * (count_ - 1)),
		      0. );
   		 
  return trans;
}


DDRotationMatrix global_simpleAngular_0::rotation()
{
  std::cout << "ParS_[\"rotate\"][0] = " << ParS_["rotate"][0] << std::endl; 
  if (ParS_["rotate"][0] == "T" || ParS_["rotate"][0] == "1"
      || ParS_["rotate"][0] == "True")
    {
      double angle = -(ParE_["offset"][0]/rad + (ParE_["delta"][0]/rad) * (count_ - 1));
      DDRotationMatrix rm1;
      if (ParS_["orientation"].size() != 0)
	{
	  std::string name=ParS_["orientation"][0];
	  size_t foundColon = 0;
	  std::string rn = "";
	  std::string ns = "";
	  while (foundColon < name.size() && name[foundColon] != ':')
	    ++foundColon;
	  if (foundColon != name.size())
	    {
	      for (size_t j = foundColon + 1; j < name.size(); ++j)
		rn = rn + name[j];
	      for (size_t i = 0; i < foundColon; ++i)
		ns = ns + name[i];
	    }
	  if (rn != "" && ns != "")
	    {
	      DDRotation myDDRotation(DDName(rn, ns));
	      rm1 = *(myDDRotation.rotation());
	    }
	  else
	    std::cout << "MAJOR PROBLEM: expected a fully qualified DDName but got :" 
		 << name << std::endl;
	}
      ROOT::Math::RotationZ rm(angle);
      rm1.Invert(); 
      rm.Invert();
      return rm * rm1;
    }
  else if (ParS_["orientation"].size() != 0)
    {
      // return the orientation matrix
      std::string name=ParS_["orientation"][0];
      size_t foundColon = 0;
      std::string rn = "";
      std::string ns = "";
      while (foundColon < name.size() && name[foundColon] != ':')
	++foundColon;
      if (foundColon != name.size())
	{
	  for (size_t j = foundColon + 1; j < name.size(); ++j)
	    rn = rn + name[j];
	  for (size_t i = 0; i < foundColon; ++i)
	    ns = ns + name[i];
	}
      if (rn != "" && ns != "")
	{
	  
	  DDRotation myDDRotation(DDName(rn, ns));
	  std::cout << "about to return *(myDDRotation.rotation())" << std::endl;
	  std::cout << *myDDRotation.rotation() << std::endl;
	  return *(myDDRotation.rotation());
	}
      else
	std::cout << "MAJOR PROBLEM: expected a fully qualified DDName but got " 
	     << name << " therefore could not look up the rotation." << std::endl;
      return DDRotationMatrix();
    }
  else
    {
      return DDRotationMatrix(); // return identity matrix.
    }
}   





// optional, not in the XML, omitted.
void global_simpleAngular_0::checkTermination()
{
  // initially I thought of ending our rotation algorithm at the vertical
  // (+x), but decided it should always go full circle.
  //  if ((ParE_["offset"][0] + ParE_["delta"][0] * count_ / deg) > 360.)
  if (ParE_["delta"][0] * count_ / deg > 360.)
    terminate();

}


void global_simpleAngular_0::stream(std::ostream & os) const
{
  os << "global_simplesimpleAngular_0::stream(): not implemented.";
}




// always the same ctor
global_simpleAngular_1::global_simpleAngular_1(AlgoPos * a,std::string label)
 : AlgoImpl(a,label)
{ }

global_simpleAngular_1::~global_simpleAngular_1() 
{ }

bool global_simpleAngular_1::checkParameters() 
{
  bool result = true;   
		  
  // check for delta 
  if (ParE_["number"].size() == 0) {
    result = false; // don't select this implementation, because number is missing
  }  		  
  else {
    // check for valid number value
    if (ParE_["number"][0] == 0.) {
      err_ += "\tnumber must not be 0\n";
      result = false;
    }
  }
		  
  // check for presence of delta
  if (ParE_["delta"].size() != 0) {
//     err_ += "\tcan not specify delta and number\n";
    result = false;
  }  		  

  return result;   
}


DDTranslation global_simpleAngular_1::translation()  
{
    // we can safely fetch all parameters, because they
  // have been checked already ...
  double offset = ParE_["offset"][0];
  double number = ParE_["number"][0];
  double delta = (360.0 / number) * deg;
  double radius  = ParE_["radius"][0];
//  std::cout << "x = " << radius * cos(offset + delta * (count_ - 1)) << "  y = " << radius * sin(offset + delta * (count_ - 1)) << std::endl;
  DDTranslation trans(radius * cos(offset + delta * (count_ - 1)),
		      radius * sin(offset + delta * (count_ - 1)),
		      0. );
   
  return trans;
}


DDRotationMatrix global_simpleAngular_1::rotation()
{
  double number = ParE_["number"][0];
  double delta = (360.0 / number) * deg;
  if (ParS_["rotate"][0] == "T" || ParS_["rotate"][0] == "1"
      || ParS_["rotate"][0] == "True")
    {
      double angle = -(ParE_["offset"][0]/rad + (delta/rad) * (count_ - 1));

      DDRotationMatrix rm1;
      if (ParS_["orientation"].size() != 0)
	{
	  std::string name=ParS_["orientation"][0];
	  size_t foundColon = 0;
	  std::string rn = "";
	  std::string ns = "";
	  while (foundColon < name.size() && name[foundColon] != ':')
	    ++foundColon;
	  if (foundColon != name.size())
	    {
	      for (size_t j = foundColon + 1; j < name.size(); ++j)
		rn = rn + name[j];
	      for (size_t i = 0; i < foundColon; ++i)
		ns = ns + name[i];
	    }
	  if (rn != "" && ns != "")
	    {
	      DDRotation myDDRotation(DDName(rn, ns));
	      rm1 = *(myDDRotation.rotation());
	    }
	  else
	    std::cout << "MAJOR PROBLEM: expected a fully qualified DDName but got :" 
		 << name << " therefore could not look up the rotation." << std::endl;
	}
      ROOT::Math::RotationZ rm(angle);
      rm1.Invert(); 
      rm.Invert();
      return rm * rm1;
    }
  else if (ParS_["orientation"].size() != 0)
    {
      std::string name=ParS_["orientation"][0];
      size_t foundColon = 0;
      std::string rn = "";
      std::string ns = "";
      while (foundColon < name.size() && name[foundColon] != ':')
	++foundColon;
      if (foundColon != name.size())
	{
	  for (size_t j = foundColon + 1; j < name.size(); ++j)
	    rn = rn + name[j];
	  for (size_t i = 0; i < foundColon; ++i)
	    ns = ns + name[i];
	}
      if (rn != "" && ns != "")
	{
	  
	  DDRotation myDDRotation(DDName(rn, ns));
	  return *(myDDRotation.rotation());
	}
      else
	std::cout << "MAJOR PROBLEM: expected a fully qualified DDName but got " 
	     << name << " therefore could not look up the rotation." << std::endl;

      return DDRotationMatrix();
    }
  else
    {
      return DDRotationMatrix(); // return identity matrix.
    }
}   





// optional, not in the XML, omitted.
void global_simpleAngular_1::checkTermination()
{
  //double delta = (360.0 / ParE_["number"][0]) * deg;
  //  if ((ParE_["offset"][0] + count_ * delta) / deg > 360.)
  if (count_ > ParE_["number"][0])
    terminate();
}


void global_simpleAngular_1::stream(std::ostream & os) const
{
  os << "global_simpleAngular_0::stream(): not implemented.";
}




// always the same ctor
global_simpleAngular_2::global_simpleAngular_2(AlgoPos * a,std::string label)
 : AlgoImpl(a,label)
{ }

global_simpleAngular_2::~global_simpleAngular_2() 
{ }

bool global_simpleAngular_2::checkParameters() 
{
  bool result = true;
		  
  // check for delta 
  if (ParE_["number"].size() == 0) {
    result = false; // don't select this implementation, because number is missing
  }  		  
  else {
    // check for valid number value
    if (ParE_["number"][0] == 0.) {
      err_ += "\tnumber must not be 0\n";
      result = false;
    }
  }
		  
  // check for presence of delta
  if (ParE_["delta"].size() == 0) {
    result = false; // don't select this implementation, because delta is missing.
  }  		  
  else {
    // check for valid delta value
    if (ParE_["delta"][0] == 0.) {
      err_ += "\tdelta must not be 0\n";
      result = false;
    }
  }

  double delta = ParE_["delta"][0];
  double number = ParE_["number"][0];
  if (delta * number > 360. * deg) {
    err_ += "\tat this time delta * number can not be greater than 360 degrees\n";
    result = false;
  }
  return result;   
}


DDTranslation global_simpleAngular_2::translation()  
{
    // we can safely fetch all parameters, because they
  // have been checked already ...
  double offset = ParE_["offset"][0];
//  double number = ParE_["number"][0];
  double delta = ParE_["delta"][0];
  double radius  = ParE_["radius"][0];
//  std::cout << "x = " << radius * cos(offset + delta * (count_ - 1)) << "  y = " << radius * sin(offset + delta * (count_ - 1)) << std::endl;
  DDTranslation trans(radius * cos(offset + delta * (count_ - 1)),
		      radius * sin(offset + delta * (count_ - 1)),
		      0. );
   
  return trans;
}


DDRotationMatrix global_simpleAngular_2::rotation()
{
  
//  double number = ParE_["number"][0];
  double delta = ParE_["delta"][0];
  if (ParS_["rotate"][0] == "T" || ParS_["rotate"][0] == "1"
      || ParS_["rotate"][0] == "True")
    {
      double angle = -(ParE_["offset"][0]/rad + (delta/rad) * (count_ - 1));
      DDRotationMatrix rm1;
      if (ParS_["orientation"].size() != 0)
	{
	  std::string name=ParS_["orientation"][0];
	  size_t foundColon = 0;
	  std::string rn = "";
	  std::string ns = "";
	  while (foundColon < name.size() && name[foundColon] != ':')
	    ++foundColon;
	  if (foundColon != name.size())
	    {
	      for (size_t j = foundColon + 1; j < name.size(); ++j)
		rn = rn + name[j];
	      for (size_t i = 0; i < foundColon; ++i)
		ns = ns + name[i];
	    }
	  if (rn != "" && ns != "")
	    {
	      DDRotation myDDRotation(DDName(rn, ns));
	      rm1 = *(myDDRotation.rotation());
	    }
	  else
	    std::cout << "MAJOR PROBLEM: expected a fully qualified DDName but got :" 
		 << name << std::endl;
	}
      ROOT::Math::RotationZ rm(angle);
      rm1.Invert(); 
      rm.Invert();
      return rm * rm1;
    }
  else if (ParS_["orientation"].size() != 0)
    {
      // return the orientation matrix
      std::string name=ParS_["orientation"][0];
      size_t foundColon = 0;
      std::string rn = "";
      std::string ns = "";
      while (foundColon < name.size() && name[foundColon] != ':')
	++foundColon;
      if (foundColon != name.size())
	{
	  for (size_t j = foundColon + 1; j < name.size(); ++j)
	    rn = rn + name[j];
	  for (size_t i = 0; i < foundColon; ++i)
	    ns = ns + name[i];
	}
      if (rn != "" && ns != "")
	{
	  
	  DDRotation myDDRotation(DDName(rn, ns));
	  return *(myDDRotation.rotation());
	}
      else
	std::cout << "MAJOR PROBLEM: expected a fully qualified DDName but got " 
	     << name << " therefore could not look up the rotation." << std::endl;
      return DDRotationMatrix();
    }
  else
    {
      return DDRotationMatrix(); // return identity matrix.
    }
}   





// optional, not in the XML, omitted.
void global_simpleAngular_2::checkTermination()
{
  //double delta = (360.0 / ParE_["number"][0]) * deg;
  //  if ((ParE_["offset"][0] + count_ * delta) / deg > 360.)
  if (count_ > ParE_["number"][0])
    terminate();
}


void global_simpleAngular_2::stream(std::ostream & os) const
{
  os << "global_simpleAngular_0::stream(): not implemented.";
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
global_simpleAngular_Check::global_simpleAngular_Check()
{
  // for the time being only expression-valued parameters can be automatically checked
  // against their specified constraints
  
  // schema default values will be shown if necessary in the following XML comments 
  // on the second line. The fist shows the values as given in the instance document

  // expressions have to be converted into doubles. No variables [bla] shall be allowed
  // inside the expressions; SystemOfUnits-symbols are the only supported ones.
  

  constraintsE_["radius"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0.0
                                       );

  constraintsE_["offset"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					0.0*deg,  // min
					360.0*deg,  // max 
					false,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0.0
                                       );

  constraintsE_["delta"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					0.0*deg,  // min
					360.0*deg,  // max 
					false,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					0.
                                       );

  constraintsE_["number"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					false,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					0.
                                       );

  constraintsE_["radius"] = ConstraintE( 1,      // minOccurs
                                        1,      // maxOccurs
					-MAX_DOUBLE,  // min
					+MAX_DOUBLE,  // max 
					true,    // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					0.0
                                       );

  constraintsS_["rotate"] = ConstraintS( 1,      // minOccurs
                                        1,      // maxOccurs
					
					 
					false,   // use, true=required, false=optional
					true,    // use, true=use default, false=no default
					"1"
                                       );

  constraintsS_["orientation"] = ConstraintS( 1,      // minOccurs
                                        1,      // maxOccurs
					
					 
					false,    // use, true=required, false=optional
					false,    // use, true=use default, false=no default
					""
                                       );

}


