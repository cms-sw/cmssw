#ifndef DDValue_h
#define DDValue_h
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "DetectorDescription/Core/interface/DDValuePair.h"
#include "DetectorDescription/Base/interface/Ptr.h"

#include "boost/shared_ptr.hpp"

class DDValue;
class DDSpecifics;
class DDLSpecPar;

//std::ostream & operator<< ( std::ostream &, const DDValue &);
/** A DDValue std::maps a std::vector of DDValuePair (std::string,double) to a name. Names of DDValues are stored
 transiently. Furthermore, an ID is assigned std::mapping to the name.
 If a particular DDValue is not used anymore, use DDValue::clear() to free the internally
 allocated memory. Use DDValue::setEvalState(true) to indicate whether the double numbers stored in
 the DDValuePair make sense, otherwise an exception will be thrown when trying to get access
 to these values via DDValue::doubles() or DDValue::operator[].
*/
class DDValue //: public std::vector<DDValuePair>
{
  friend class DDSpecifics;
  friend class DDLSpecPar;
public:
  //! create a unnamed emtpy value. One can assing a named DDValue to it.
  DDValue() : id_(0), vecPair_(0) { }
  
  //! create a named empty value
  DDValue(const std::string &);

  //! create a named empty value
  DDValue(const char *);
 
  void init(const std::string &);
 
  //! creates a named DDValue initialized with a std::vector of values 
  explicit DDValue(const std::string &, const std::vector<DDValuePair>&);
  
  //! creates a single double valued named DDValue. The corresponding std::string-value is an empty std::string
  explicit DDValue(const std::string &, double);
  
  /** creates a single std::string & numerical-valued named DDValue.  */
  explicit DDValue(const std::string &, const std::string &, double);
  
  /** creates a single std::string-valued named DDValue */
  explicit DDValue(const std::string & name, const std::string & val);
  
  explicit DDValue(unsigned int);
  //~DDValue() { destroy(); }
  //DDValue(const DDValue &);
  //DDValue & operator=(const DDValue &);
  //const DDValuePair &  
  
  ~DDValue();
  
  //! returns the ID of the DDValue
  unsigned int id() const { return id_; }
  
  //! converts a DDValue object into its ID
  operator unsigned int() const { return id_; }
  
  //! the name of the DDValue
  const std::string & name() const { return names()[id_]; }
  
  /** access to the values stored in DDValue by an index. Note, that
   the index is not checked for bounds excess! */
  DDValuePair operator[](unsigned int i) const; 
   
  //! a reference to the std::string-valued values stored in the given instance of DDValue
  const std::vector<std::string> & strings() const { return vecPair_->second.first; }
  
  //! a reference to the double-valued values stored in the given instance of DDValue
  const std::vector<double> & doubles() const;
  //const DDValuePair & operator[](unsigned int i) const { return (*valPairs_)[i]  ; }
  
  //! the size of the stored value-pairs (std::string,double)
  unsigned int size() const { 
   return vecPair_ ? vecPair_->second.first.size() : 0 ; 
  } 
  
  static void clear();
  
  //! set to true, if the double-values (method DDValue::doubles()) make sense
  void setEvalState(bool newState); 
  
  //! true, if values are numerical evaluated; else false. 
  /** in case of a 'true' return value, the method DDValue::doubles() and the operator
     DDValue::operator[] can be used */
  bool isEvaluated() const;
  
  //! Two DDValues are equal only if their id() is equal AND their values are equal
  /** If the DDValue::isEvalued() == true, the numerical representation is taken for comparison,
      else the std::string representation */
  bool operator==(const DDValue & v) const;
  /* {
    return id()==v.id() 
      && ( 
	  vecPair_ == v.vecPair_ || 
	  (
	   vecPair_ && v.vecPair_ &&
	   ( (vecPair_->first) ? (vecPair_->second.second == v.vecPair_->second.second) 
	     : (vecPair_->second.first == v.vecPair_->second.first) 
	     )
	   )
	   )
  }
  */
  
  //! A DDValue a is smaller than a DDValue b if (a.id()<b.id()) OR (a.id()==b.id() and value(a)<value(b))
  bool operator<(const DDValue &) const;
  /*{
    return id()<v.id() || 
      (
       id()==v.id() && 
       vecPair_ && v.vecPair_ &&
       ( (vecPair_->first) ? (vecPair_->second.second < v.vecPair_->second.second) 
	 : (vecPair_->second.first < v.vecPair_->second.first)
	 )
       )
  }
  */
  
private:  
  typedef std::pair<bool, std::pair<std::vector<std::string>, std::vector<double> > >vecpair_type;
  static std::vector<std::string>& names();
  static std::map<std::string,unsigned int>& indexer();
  static std::vector<boost::shared_ptr<vecpair_type> >& mem(vecpair_type*);
  //DDValue(const DDValue &);
  //DDValue & operator=(const DDValue &);
  //void destroy() { delete vecPair_;} //delete valPairs_; }
  
  //static std::vector<std::map<std::string,unsigned int>::const_iterator> names_;
  unsigned int id_;
  
  //Ptr<vecpair_type>  vecPair_;
public:  
  vecpair_type* vecPair_;

public:  
  //bool isEvaluated_;
  //auto_ptr<vecpair_type> vecPair_;
  //std::vector<DDValuePair> * valPairs_;
};


std::ostream & operator<<(std::ostream & o, const DDValue & v);
#endif // DDValue_h
