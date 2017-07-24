#ifndef DD_DDPartSelectionExpander_h
#define DD_DDPartSelectionExpander_h

#include <stddef.h>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/Graph.h"

class DDPartSelection;
struct DDPartSelectionLevel;

class DDPartSelectionExpander
{
public:
  using PartSelectionTree = Graph<DDPartSelectionLevel, char>;
  explicit DDPartSelectionExpander(const DDCompactView &);
  
  size_t expand(const DDPartSelection & input, PartSelectionTree & result) const;
  
private:
  const DDCompactView::graph_type & graph_;
};

#endif
