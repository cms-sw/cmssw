#ifndef DDValue_h
#define DDValue_h
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "DetectorDescription/DDCore/interface/DDValuePair.h"
#include "DetectorDescription/DDBase/interface/Ptr.h"

using std::string;
using std::vector;
using std::ostream;
using std::pair;
using std::map;
//using DDI::Ptr;

class DDValue;
class DDSpecifics;
class DDLSpecPar;

//ostream & operator<< ( ostream &, const DDValue &);
/** A DDValue maps a vector of DDValuePair (string,double) to a name. Names of DDValues are stored
 transiently. Furthermore, an ID is assigned mapping to the name.
 If a particular DDValue is not used anymore, use DDValue::clear() to free the internally
 allocated memory. Use DDValue::setEvalState(true) to indicate whether the double numbers stored in
 the DDValuePair make sense, otherwise an exception will be thrown when trying to get access
 to these values via DDValue::doubles() or DDValue::operator[].
*/
class DDValue //: public vector<DDValuePair>
{
  friend class DDSpecifics;
  friend class DDLSpecPar;
public:
  //! create a unnamed emtpy value. One can assing a named DDValue to it.
  DDValue() : id_(0), vecPair_(0) { } ;
  
  //! create a named empty value
  DDValue(const string &);
  
  //! creates a named DDValue initialized with a vector of values 
  explicit DDValue(const string &, const vector<DDValuePair>&);
  
  //! creates a single double valued named DDValue. The corresponding string-value is an empty string
  explicit DDValue(const string &, double);
  
  /** creates a single string & numerical-valued named DDValue.  */
  explicit DDValue(const string &, const string &, double);
  
  /** creates a single string-valued named DDValue */
  explicit DDValue(const string & name, const string & val);
  
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
  const string & name() const { return names()[id_]; }
  
  /** access to the values stored in DDValue by an index. Note, that
   the index is not checked for bounds excess! */
  DDValuePair operator[](unsigned int i) const; 
   
  //! a reference to the string-valued values stored in the given instance of DDValue
  const vector<string> & strings() const { return vecPair_->second.first; }
  
  //! a reference to the double-valued values stored in the given instance of DDValue
  const vector<double> & doubles() const;
  //const DDValuePair & operator[](unsigned int i) const { return (*valPairs_)[i]  ; }
  
  //! the size of the stored value-pairs (string,double)
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
      else the string representation */
  inline bool operator==(const DDValue &) const;
  
  //! A DDValue a is smaller than a DDValue b if (a.id()<b.id()) OR (a.id()==b.id() and value(a)<value(b))
  inline bool operator<(const DDValue &) const;
  
private:  
  typedef pair<bool, pair<vector<string>, vector<double> > >vecpair_type;
  static vector<string>& names();
  static map<string,unsigned int>& indexer();
  static vector<vecpair_type*>& mem(vecpair_type*);
  //DDValue(const DDValue &);
  //DDValue & operator=(const DDValue &);
  //void destroy() { delete vecPair_;} //delete valPairs_; }
  
  //static vector<map<string,unsigned int>::const_iterator> names_;
  unsigned int id_;
  
  //Ptr<vecpair_type>  vecPair_;
public:  
  vecpair_type* vecPair_;

public:  
  //bool isEvaluated_;
  //auto_ptr<vecpair_type> vecPair_;
  //vector<DDValuePair> * valPairs_;
};


ostream & operator<<(ostream & o, const DDValue & v);
#endif // DDValue_h
