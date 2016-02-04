#ifndef DDCore_DDQuery_h
#define DDCore_DDQuery_h

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDScope.h"

#include <map>
#include <vector>
#include <utility>

//class DDCompactView;

//! Base class for querying for nodes in the DDExpandedView
class DDQuery
{
public:
  enum log_op { AND, OR };
  //! sets up a query
  DDQuery(const DDCompactView &);
  
  virtual ~DDQuery();
  
  virtual const std::vector<DDExpandedNode> & exec();
  
  virtual void addFilter(const DDFilter &, log_op op=AND);
  
  virtual void setScope(const DDScope &);
  
  //void composeQuery(const DDQuery &);
  
protected:
  //const DDCompactView & cpv_;
  DDExpandedView epv_;  
  const DDScope * scope_;
  typedef std::pair<bool, DDFilter *> criterion_type;
  typedef std::vector<criterion_type> criteria_type;
  typedef std::vector<log_op> logops_type;
  
  criteria_type criteria_; // one filter and the result on the current node
  logops_type logOps_; // logical operation for merging the result of 2 filters
  
  std::vector<DDExpandedNode> result_; // query result
};

#endif
