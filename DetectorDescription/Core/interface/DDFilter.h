#ifndef DDCore_DDFilter_h
#define DDCore_DDFilter_h

#include "DetectorDescription/Core/interface/DDValue.h"
#include <vector>
#include <iosfwd>

class DDQuery;
class DDExpandedView;

//! comparison operators to be used with this filter
enum class DDCompOp { equals, matches, not_equals, not_matches, smaller, bigger, smaller_equals, bigger_equals };
  
//! logical operations to obtain one result from two filter comparisons
enum class DDLogOp { AND, OR };

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

//! The DDGenericFilter is a runtime-parametrized Filter looking on DDSpecifcs
class DDSpecificsFilter : public DDFilter
{
  friend std::ostream & operator<<(std::ostream & os, const DDSpecificsFilter & f);

public:
  DDSpecificsFilter();
  
  ~DDSpecificsFilter();
  
  bool accept(const DDExpandedView &) const; 
	      
  void setCriteria(const DDValue & nameVal, // name & value of a variable 
                   DDCompOp, 
		   DDLogOp l = DDLogOp::AND, 
		   bool asString = true, // compare strings otherwise doubles
		   bool merged = true // use merged-specifics or simple-specifics
		   );
		      
  struct SpecificCriterion {
    SpecificCriterion(const DDValue & nameVal, 
		      DDCompOp op,
		      bool asString,
		      bool merged)
     : nameVal_(nameVal), 
       comp_(op), 
       asString_(asString),
       merged_(merged)
     { }
     
     DDValue nameVal_;
     DDCompOp comp_;
     bool asString_;
     bool merged_;
  };
  
protected:  

  bool accept_impl(const DDExpandedView &) const;

  std::vector<SpecificCriterion> criteria_; 
  std::vector<DDLogOp> logOps_;
};

std::ostream & operator<<(std::ostream & os, const DDSpecificsFilter & f);
#endif


