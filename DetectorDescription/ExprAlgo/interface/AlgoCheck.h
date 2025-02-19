#ifndef ExprAlgo_AlgoCheck_h
#define ExprAlgo_AlgoCheck_h

#include <string>
#include <map>
#include "DetectorDescription/Base/interface/DDAlgoPar.h"


//! base class for generated checking code for algorithm parameters. 
class AlgoCheck
{
public:
  //! the ctor of the derived class has to fill the members constraintsE_,S_
  AlgoCheck()  { }
  
  virtual ~AlgoCheck() { }
  
  //! constraints as defined for algorithm-parameters in the XML schema Algorithm.xsd, expressions
  struct ConstraintE {
   ConstraintE() { } // for STL conformance
   ConstraintE(int minOccurs,
               int maxOccurs,
	       double min,
	       double max,
	       bool use,
	       bool deflt,
	       double defltVal
	       )
   : minOccurs_(minOccurs), maxOccurs_(maxOccurs), use_(use),
     default_(deflt), min_(min), max_(max), defaultVal_(defltVal) { }	        
   int minOccurs_, maxOccurs_;
   bool use_, default_; // use==true==required, default==true==default-val-specified
   double min_, max_, defaultVal_;  	       
  };
  
  //! constraints as defined for algorithm-parameters in the XML schema Algorithm.xsd, strings
  struct ConstraintS {
   ConstraintS() { } // for STL conformance
   ConstraintS(int minOccurs,
               int maxOccurs,
	       bool use,
	       bool deflt,
	       std::string defltVal
	       )
   : minOccurs_(minOccurs), maxOccurs_(maxOccurs), use_(use), default_(deflt),
     defaultVal_(defltVal)
   { }	        
   int minOccurs_, maxOccurs_;
   bool use_, default_; 
   std::string defaultVal_;
  };
  
  //! returns true if the check was successfull (parameters conform to XML specification)
  bool check(parS_type & ps, parE_type & pe, std::string & err);
  
  typedef std::map<std::string,ConstraintE> constraintsE_type;
  typedef std::map<std::string,ConstraintS> constraintsS_type;

protected:
  bool checkBounds(parE_type::iterator pit, // current parameter to be checked
                            constraintsE_type::iterator cit, // corresponding constraints
			    std::string & err // error std::string
			    );
  bool checkStrings(parS_type::iterator sit, // current parameter to be checked
		 constraintsS_type::iterator cit, // corresponding constraints
		 std::string & err // error std::string
		 );

  //! ahh, converts a double into a std::string ... yet another one of this kind!
  static std::string d2s(double x);
  
  //! format: "ParameterName" -> ConstraintE
  constraintsE_type constraintsE_;  
  
  //! format: "ParameterName" -> ConstraintS
  constraintsS_type constraintsS_;
};

#endif
