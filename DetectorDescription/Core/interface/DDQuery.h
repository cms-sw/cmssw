#ifndef DDCore_DDQuery_h
#define DDCore_DDQuery_h

#include <map>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDScope.h"

class DDCompactView;
class DDScope;

//! Base class for querying for nodes in the DDExpandedView
class DDQuery
{
public:
  //! sets up a query
  DDQuery(const DDCompactView &);
  
  virtual ~DDQuery();
  
  virtual const std::vector<DDExpandedNode> & exec();
  
  virtual void addFilter(const DDFilter &);
  
  virtual void setScope(const DDScope &);
  
protected:
  DDExpandedView epv_;  
  const DDScope * scope_;
  
  std::vector<std::pair<bool, DDFilter *> > criteria_; // one filter and the result on the current node
  std::vector<DDExpandedNode> result_; // query result
};

#endif
