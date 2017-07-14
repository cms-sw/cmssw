#ifndef DDCore_DDFilter_h
#define DDCore_DDFilter_h

#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDValue.h"

class DDExpandedView;

//! comparison operators to be used with this filter
enum class DDCompOp { equals, not_equals};
  
//! A Filter accepts or rejects a DDExpandedNode based on a user-coded decision rule
class DDFilter
{
public:
  DDFilter();
  
  virtual ~DDFilter();
  
  //! true, if the DDExpandedNode fulfills the filter criteria
  virtual bool accept(const DDExpandedView &) const = 0;  
};

//! A DDFilter that always returns true
class DDPassAllFilter : public DDFilter
{
public:
  bool accept(const DDExpandedView &) const final {
    return true;
  }
};

//! The DDGenericFilter is a runtime-parametrized Filter looking on DDSpecifcs
class DDSpecificsFilter : public DDFilter
{
  friend std::ostream & operator<<(std::ostream & os, const DDSpecificsFilter & f);

public:
  DDSpecificsFilter();
  
  ~DDSpecificsFilter() override;
  
  bool accept(const DDExpandedView &) const final; 
	      
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

class DDSpecificsHasNamedValueFilter : public DDFilter {
public:
  explicit DDSpecificsHasNamedValueFilter(const std::string& iAttribute):
    attribute_(iAttribute,"",0 )
    {}

  bool accept(const DDExpandedView &) const final; 

private:
  DDValue attribute_;

};

class DDSpecificsMatchesValueFilter : public DDFilter {
public:
  explicit DDSpecificsMatchesValueFilter(const DDValue& iValue):
    value_(iValue)
    {}

  bool accept(const DDExpandedView &) const final; 

private:
  DDValue value_;

};

template<typename F1, typename F2>
class DDAndFilter : public DDFilter {
public:
  DDAndFilter(F1 iF1, F2 iF2):
    f1_(std::move(iF1)),
    f2_(std::move(iF2)) {}

  bool accept(const DDExpandedView & node) const final {
    return f1_.accept(node) && f2_.accept(node);
  }
private:
  F1 f1_;
  F2 f2_;
};

template<typename F1, typename F2>
  DDAndFilter<F1,F2> make_and_ddfilter(F1 f1, F2 f2) 
{
  return DDAndFilter<F1,F2>(std::move(f1), std::move(f2));
}

#endif


