#ifndef DetectorDescription_Core_DDValue_h
#define DetectorDescription_Core_DDValue_h

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include "tbb/concurrent_vector.h"
#include "tbb/concurrent_unordered_map.h"

#include "DetectorDescription/Core/interface/DDValuePair.h"

/** A DDValue std::maps a std::vector of DDValuePair (std::string,double) to a name. Names of DDValues are stored
 transiently. Furthermore, an ID is assigned std::mapping to the name.
 Use DDValue::setEvalState(true) to indicate whether the double numbers stored in
 the DDValuePair make sense, otherwise an exception will be thrown when trying to get access
 to these values via DDValue::doubles() or DDValue::operator[].
*/
class DDValue
{
public:
  //! create a unnamed emtpy value. One can assing a named DDValue to it.
  DDValue( void ) : id_(0), vecPair_() { }
  
  //! create a named empty value
  explicit DDValue( const std::string & );

  //! create a named empty value
  explicit DDValue( const char * );
 
  //! creates a named DDValue initialized with a std::vector of values 
  explicit DDValue( const std::string &, const std::vector<DDValuePair>& );
  
  //! creates a single double valued named DDValue. The corresponding std::string-value is an empty std::string
  explicit DDValue( const std::string &, double );
  
  /** creates a single std::string & numerical-valued named DDValue.  */
  explicit DDValue( const std::string &, const std::string &, double );
  
  /** creates a single std::string-valued named DDValue */
  explicit DDValue( const std::string & name, const std::string & val );
  
  explicit DDValue( unsigned int );
  
  ~DDValue( void );
  
  //! returns the ID of the DDValue
  unsigned int id( void ) const { return id_; }
  
  //! converts a DDValue object into its ID
  operator unsigned int( void ) const { return id_; }
  
  //! the name of the DDValue
  const std::string & name( void ) const { return *(names()[id_].string_); }
  
  /** access to the values stored in DDValue by an index. Note, that
   the index is not checked for bounds excess! */
  DDValuePair operator[]( unsigned int i ) const; 
   
  //! a reference to the std::string-valued values stored in the given instance of DDValue
  const std::vector<std::string> & strings() const { return vecPair_->second.first; }
  
  //! a reference to the double-valued values stored in the given instance of DDValue
  const std::vector<double> & doubles() const;
  //const DDValuePair & operator[](unsigned int i) const { return (*valPairs_)[i]  ; }
  
  //! the size of the stored value-pairs (std::string,double)
  unsigned int size() const { 
   return vecPair_ ? vecPair_->second.first.size() : 0 ; 
  } 
  
  //! set to true, if the double-values (method DDValue::doubles()) make sense
  void setEvalState( bool newState ); 
  
  //! true, if values are numerical evaluated; else false. 
  /** in case of a 'true' return value, the method DDValue::doubles() and the operator
     DDValue::operator[] can be used */
  bool isEvaluated( void ) const;
  
  //! Two DDValues are equal only if their id() is equal AND their values are equal
  /** If the DDValue::isEvalued() == true, the numerical representation is taken for comparison,
      else the std::string representation */
  bool operator==( const DDValue & v ) const;
  
  //! A DDValue a is smaller than a DDValue b if (a.id()<b.id()) OR (a.id()==b.id() and value(a)<value(b))
  bool operator<( const DDValue & ) const;

private:
  ///Only used internally
  struct StringHolder {
    StringHolder() {}
    explicit StringHolder(std::string iString): string_(new std::string{std::move(iString)}) {}
    explicit StringHolder(StringHolder const& iOther): string_(new std::string{*(iOther.string_)}) {
    }
    StringHolder& operator=(const StringHolder&) = delete;
    ~StringHolder() { delete string_.load();}
    
    
    std::atomic<std::string*> string_;
  };
  struct AtomicUInt {
    AtomicUInt(unsigned int iValue): value_(iValue) {}
    AtomicUInt() {}
    AtomicUInt( const AtomicUInt& iOther) : value_(iOther.value_.load()) {}
    AtomicUInt& operator=(const AtomicUInt& iOther) {
      value_ = iOther.value_.load();
      return *this;
    }

    std::atomic<unsigned int> value_;
  };
  
  void init( const std::string & );
 
  using Names = tbb::concurrent_vector<StringHolder,tbb::zero_allocator<StringHolder>>;
  static Names& names();
  static Names initializeNames();

  using NamesToIndicies = tbb::concurrent_unordered_map<std::string,AtomicUInt>;
  static NamesToIndicies& indexer();

  unsigned int id_;
  using vecpair_type =  std::pair<bool, std::pair<std::vector<std::string>, std::vector<double>>>;  
  std::shared_ptr<vecpair_type> vecPair_;
};

std::ostream & operator<<(std::ostream & o, const DDValue & v);

#endif // DetectorDescription_Core_DDValue_h
