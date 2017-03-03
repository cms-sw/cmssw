#ifndef DDCore_DDFilter_h
#define DDCore_DDFilter_h

#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDValue.h"

class DDExpandedView;
class DDQuery;

//! comparison operators to be used with this filter
enum class DDCompOp { equals, not_equals};
  
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
  
  bool accept(const DDExpandedView &) const override final; 
	      
  void setCriteria(const DDValue & nameVal, // name & value of a variable 
                   DDCompOp );
		      
  struct SpecificCriterion {
    SpecificCriterion(const DDValue & nameVal, 
		      DDCompOp op)
     : nameVal_(nameVal), 
       comp_(op) 
     { }
     
     DDValue nameVal_;
     DDCompOp comp_;
  };
  
protected:  

  bool accept_impl(const DDExpandedView &) const;

  std::vector<SpecificCriterion> criteria_; 
};

std::ostream & operator<<(std::ostream & os, const DDSpecificsFilter & f);
#endif


