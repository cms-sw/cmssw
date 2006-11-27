#ifndef DDCore_DDFilter_h
#define DDCore_DDFilter_h

#include <vector>
#include <string>
#include <utility>

//#include "DetectorDescription/Core/interface/DDScope.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
//#include "DetectorDescription/Core/interface/tree.h"

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
  virtual bool accept(const DDExpandedView &) = 0;
  
		      //const DDGeoHistory &,
		      //const DDsvalues_type *) = 0;
  
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
  
  bool accept(const DDExpandedView &); 
              //const DDGeoHistory &,
	      //const DDsvalues_type * );
	      
  inline bool accept_impl(const DDExpandedView &);	      
    
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
  typedef std::pair<bool, SpecificCriterion >  criterion_type;  
  typedef std::vector<criterion_type> criteria_type;
  typedef std::vector<log_op> logops_type;
  // descision tree: logical-operation are the nodes, index to the criteria-std::vector the edges
  // typedef TreeNode<criterion_type,log_op> dectree_type;
  // result of one SpecificCriterion comparison is stored in the bool of the pair
  // dectree_type * decTree_; // decissiontree
  criteria_type criteria_; 
  logops_type logOps_;
  bool allAnd;
  bool allOr;
};

// SpecificsFilterString-comparison (sfs)
// template <class C> 
 inline bool sfs_compare(const DDSpecificsFilter::SpecificCriterion & crit,
                  const DDsvalues_type & sv);
 inline bool sfd_compare(const DDSpecificsFilter::SpecificCriterion & crit,
                  const DDsvalues_type & sv);		  
 inline bool sfs_compare_nm(const DDSpecificsFilter::SpecificCriterion & crit,
			    const std::vector<const DDsvalues_type *> & specs);
 inline bool sfd_compare_nm(const DDSpecificsFilter::SpecificCriterion & crit,
			    const std::vector<const DDsvalues_type *> & specs);
 std::ostream & operator<<(std::ostream & os, const DDSpecificsFilter & f);
#endif


