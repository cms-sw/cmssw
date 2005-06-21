#ifndef DDCore_DDQuery_h
#define DDCore_DDQuery_h

#include <vector>
#include <utility>

#include "DetectorDescription/DDCore/interface/DDExpandedView.h"
#include "DetectorDescription/DDCore/interface/DDFilter.h"

//class DDCompactView;

//! Base class for querying for nodes in the DDExpandedView
class DDQuery
{
public:
  enum log_op { AND, OR };
  //! sets up a query
  DDQuery(const DDCompactView &);
  
  virtual ~DDQuery();
  
  virtual const vector<DDExpandedNode> & exec();
  
  virtual void addFilter(const DDFilter &, log_op op=AND);
  
  virtual void setScope(const DDScope &);
  
  //void composeQuery(const DDQuery &);
  
protected:
  //const DDCompactView & cpv_;
  DDExpandedView epv_;  
  const DDScope * scope_;
  typedef pair<bool, DDFilter *> criterion_type;
  typedef vector<criterion_type> criteria_type;
  typedef vector<log_op> logops_type;
  
  criteria_type criteria_; // one filter and the result on the current node
  logops_type logOps_; // logical operation for merging the result of 2 filters
  
  vector<DDExpandedNode> result_; // query result
};

#endif
