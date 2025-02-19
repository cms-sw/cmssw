#ifndef DDCore_DDFilter_h
#define DDCore_DDFilter_h
/**
 * 28/11/2006  VI clean up of data structure
 *
 *
 */
#include <vector>
#include <string>
#include <iosfwd>

#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDQuery;
class DDExpandedView;

//! A Filter accepts or rejects a DDExpandedNode based on a user-coded decision rule
class DDFilter
{
  friend class DDQuery;

public:
  DDFilter();
  
  virtual ~DDFilter();
  
  //! true, if the DDExpandedNode fulfills the filter criteria
  virtual bool accept(const DDExpandedView &) const = 0;
  
};

#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDValuePair.h"


//! The DDGenericFilter is a runtime-parametrized Filter looking on DDSpecifcs
class DDSpecificsFilter : public DDFilter
{

  friend std::ostream & operator<<(std::ostream & os, const DDSpecificsFilter & f);

public:
  //! comparison operators to be used with this filter
  //! \todo use functors!
  enum comp_op { equals, matches, not_equals, not_matches, smaller, bigger, smaller_equals, bigger_equals };
  
  //! logical operations to obtain one result from two filter comparisons
  enum log_op { AND, OR };

public:
  DDSpecificsFilter();
  
  ~DDSpecificsFilter();
  
  bool accept(const DDExpandedView &) const; 
	      
  void setCriteria(const DDValue & nameVal, // name & value of a variable 
                   comp_op, 
		   log_op l = AND, 
		   bool asString = true, // compare strings otherwise doubles
		   bool merged = true // use merged-specifics or simple-specifics
		   );
		      
  struct SpecificCriterion {
    SpecificCriterion(const DDValue & nameVal, 
		      comp_op op,
		      bool asString,
		      bool merged)
     : nameVal_(nameVal), 
       comp_(op), 
       asString_(asString),
       merged_(merged)
     { }
     
     DDValue nameVal_;
     comp_op comp_;
     bool asString_;
     bool merged_;
  };
  
protected:  

  bool accept_impl(const DDExpandedView &) const;
    
  typedef SpecificCriterion  criterion_type;  
  typedef std::vector<criterion_type> criteria_type;
  typedef std::vector<log_op> logops_type;

  criteria_type criteria_; 
  logops_type logOps_;
};

std::ostream & operator<<(std::ostream & os, const DDSpecificsFilter & f);
#endif


