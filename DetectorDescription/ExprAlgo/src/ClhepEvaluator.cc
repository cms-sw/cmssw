#include <iostream>
#include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

ClhepEvaluator::ClhepEvaluator()
{ 
  // enable standard mathematical funtions
  evaluator_.setStdMath();
  
  // set Geant4 compatible units
  evaluator_.setSystemOfUnits(1.e+3, 1./1.60217733e-25, 1.e+9, 1./1.60217733e-10,
                              1.0, 1.0, 1.0);

  // set some global vars, which are in fact known by Clhep::SystemOfUnits
  // but are NOT set in CLHEP::Evaluator ...
  evaluator_.setVariable("mum","1.e-3*mm");
  evaluator_.setVariable("fm","1.e-15*meter");			      			      
}
  
  
ClhepEvaluator::~ClhepEvaluator()
{ 
  clear();
}

void dd_exchange_value(std::vector<std::string> & vars, std::vector<std::string> & vals, 
  const std::string & var, const std::string & val)
{
   std::vector<std::string>::iterator it(vars.begin()), ed(vars.end());
   std::vector<std::string>::size_type count(0);
   for (; it != ed; ++it) {
      if ( *it == var) {
         // a potential memory leak below! But CLHEP::Evaluator should take care about it!!
	 vals[count] = val;
	 break;
      }
      ++count;
   }
} 
      
  
void ClhepEvaluator::set(const std::string & ns, const std::string & name, const std::string & exprValue)
{
   checkname(ns); // fancy characters in ns or name ??
   checkname(name);
   std::string newVar;
   std::string newVal;
   prepare(ns,name,exprValue,newVar,newVal);
   DCOUT_V('C', "ClhepEvaluator: " 
        << "  in: " << ns << " " << name << " " << exprValue 
	<< "  pr: " << newVar << " " << newVal);
   //set(newVar,newVal);
   evaluator_.setVariable(newVar.c_str(), newVal.c_str());
   switch(evaluator_.status()) {
    case HepTool::Evaluator::WARNING_EXISTING_VARIABLE:
      dd_exchange_value(variables_,values_,newVar,newVal);
      break;
    case HepTool::Evaluator::OK:
    case HepTool::Evaluator::WARNING_EXISTING_FUNCTION:
    case HepTool::Evaluator::WARNING_BLANK_STRING:
      variables_.push_back(newVar);
      values_.push_back(newVal);
      break;
    default:
      std::cout << "set-var: ns=" << ns << " nm=" << name << " val=" << exprValue << std::endl;
      evaluator_.print_error();
      throwex(ns,name,exprValue,"can't set parameter !");
    }
}
  
void ClhepEvaluator::set(const std::string & n, const std::string & v)
{
  evaluator_.setVariable(n.c_str(),v.c_str());
   switch(evaluator_.status()) {
    case HepTool::Evaluator::WARNING_EXISTING_VARIABLE:
      dd_exchange_value(variables_,values_,n,v);
      break;
    case HepTool::Evaluator::OK:
    case HepTool::Evaluator::WARNING_EXISTING_FUNCTION:
    case HepTool::Evaluator::WARNING_BLANK_STRING:
      variables_.push_back(n);
      values_.push_back(v);
      break;
    default:
      std::cout << "set-varname=" << n << " val=" << v << std::endl;
      evaluator_.print_error();
      throwex("",n,v,"can't set parameter !");
    }  
}

        
double ClhepEvaluator::eval(const std::string & ns, const std::string & expr)
{

   // eval does not store std::strings in the values_!
   // eval throws if it can't evaluate!
   std::string pseudo("(evaluating)");
   std::string prepared;
   
   prepare(ns,pseudo,expr, pseudo,prepared);
   
   double result = evaluator_.evaluate(prepared.c_str());
   if(evaluator_.status()!=HepTool::Evaluator::OK) {
      std::cout << "expr: " << prepared << std::endl;
      std::cout << "------";
      for (int i=0; i<evaluator_.error_position(); ++i) std::cout << "-";
      std::cout << "^" << std::endl;
      evaluator_.print_error();
      throwex(ns,prepared,expr,"can't evaluate: " + expr + std::string("!"));
    }
   
   return result;
}
  
double ClhepEvaluator::eval(const char * expression)
{
   double result = evaluator_.evaluate(expression);
   if (evaluator_.status()!=HepTool::Evaluator::OK) {
      std::cout << "expr: " << expression << std::endl;
      std::cout << "------";
      for (int i=0; i<evaluator_.error_position(); ++i) std::cout << "-";
      std::cout << "^" << std::endl;
      evaluator_.print_error();
      throwex("",expression,"","can't evaluate: " + std::string(expression) + std::string("!"));   
   }
   return result;
}

bool ClhepEvaluator::isDefined(const std::string & ns, //< current namespace
                 const std::string & name //< name of the variable inside current namespace
		 ) 
{
   std::string newVar; 
   std::string newVal;
   prepare(ns,name,"0", newVar,newVal);
   return evaluator_.findVariable(newVar.c_str());
}		 
 
  
void ClhepEvaluator::clear()
{
   // clear the dictionary
   evaluator_.clear();
   
   // clear the cache of values & variable-names   
   variables_.clear();
   values_.clear();
}


void ClhepEvaluator::prepare(const std::string & ns, 
                             const std::string & name, 
			     const std::string & exprValue,
                             std::string & nameResult, 
			     std::string & valResult) const
{
   static const std::string sep("___"); // separator between ns and name
   // SOME SPAGHETTI CODE ...
   //  break it down into some addional member functions ... 
   
   // the name and namespaces are not checked for 'forbidden' symbols like [,],: ...
   nameResult = ns + sep + name;
   
   // scan the expression std::string and remove [ ], and insert the current namespace if it's missing
   std::string temp;
   
   // 2 pass for simplicity (which is NOT efficient ...)
   // pass 1: find variables without namespace, e.g. [abcd], and mark them
   // pass 2: remove [ ] & ( exchange ':' with '_' | add the namespace at marked variables )
   
   std::string::size_type sz = exprValue.size();
   std::string::size_type idx =0;
   bool insideBracket = false;
   bool nsFound = false;
   int varCount=0; // count the variables from 1,2,3,...
   std::vector<int> hasNs(1); // marked[i]=1 ... variable number i has a namespace attached with ':'
   
   while(idx<sz) {
     switch(exprValue[idx]) {

     case '[':     
       if (nsFound || insideBracket) { // oops, something went wrong. simply throw!
         throwex(ns,name,exprValue,
	         "found a ':' outside '[..]' , or too many '[' !",idx); 
       }
       insideBracket=true; 
       ++varCount;
       break;
     
     case ']':
       if (!insideBracket) {
         throwex(ns,name,exprValue,"too many ']' !",idx);
       }
       insideBracket=false;  
       if (nsFound) {
         nsFound=false; // reset
	 hasNs.push_back(1);
       }
       else {
        hasNs.push_back(0);	 
       }	
       break;
     
     case ':':
       if ( (!insideBracket) || nsFound ) { // oops, a namespace outside [] or a 2nd ':' inside []! !
         throwex(ns,name,exprValue,
	         "found a ':' outside '[..]' , or multiple ':' inside '[..]'",idx);
       }	         
       nsFound=true;
       break;  
     
     default:
      ;
     } // switch
     
     ++idx;
   } // while(sz)
   
   // status after pass 1 must be: every [ ] is closed and no ':' 
   if ( insideBracket || nsFound ) {
     throwex(ns,name,exprValue,
	     "'[..]' not closed , or ':' outside of '[..]'",idx);
   }
   
   // Pass 2: now remove all '[' ']', replace ':' or add 'ns' + '_'
   //sz = exprValue.size();
   idx=0;
   varCount=0;
   //bool ommit = false;
   while (idx<sz) {
     switch(exprValue[idx]) {
     
     case '[':
       ++varCount;
       if ( !hasNs[varCount] ) {
         valResult = valResult + ns + sep;
       }	 
       break;
     
     case ']':
       break;
     
     case ':':
       valResult = valResult + sep;
       break;  
       
     default:
       valResult = valResult + exprValue[idx];
     } // switch
   
     ++idx;
   } // while 
}	       
	       
	

 void ClhepEvaluator::throwex(const std::string & ns, 
                              const std::string & name, 
	                      const std::string & expr,
	                      const std::string & reason,
	                      int idx) const
{
   std::string er = std::string("ClhepEvaluator ERROR: ") + reason + std::string("\n")
	           + std::string(" nmspace=") + ns 
		   + std::string("\n varname=") + name
		   + std::string("\n exp=") + expr
		   + std::string("\n  at=") + expr.substr(0,idx);
	 throw DDException(er);	   

}			      


 void ClhepEvaluator::checkname(const std::string & s) const
 {
   // '['   ']'   ' '  ':'   are forbidden for names and namespaces of parameters
   std::string::size_type sz = s.size();
   while(sz) {
     --sz;
     //bool stop = false;
     switch (s[sz]) {
     case ']':
     case '[':
     case ' ':
     case ':':
     case '\n':
     case '\t':
    // case '.':
     case '&':
     case '*':
     case '+':
     case '-':
     case '/':
     case '^':
       std::string e = std::string("ClhepEvaluator ERROR: forbidden character '")
                + s[sz] + std::string("' found in '") + s + std::string("' !");
       throw DDException(e);         
       break;  
     }
   }
 
 }
 

