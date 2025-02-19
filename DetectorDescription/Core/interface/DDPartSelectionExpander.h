#ifndef DD_DDPartSelectionExpander_h
#define DD_DDPartSelectionExpander_h

#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"


typedef graph<DDPartSelectionLevel,char> PartSelectionTree;
 

class DDPartSelectionExpander
{
public:
  explicit DDPartSelectionExpander(const DDCompactView &);
  
  size_t expand(const DDPartSelection & input, PartSelectionTree & result) const;
  
private:
  const DDCompactView::graph_type & graph_;
};

#endif
