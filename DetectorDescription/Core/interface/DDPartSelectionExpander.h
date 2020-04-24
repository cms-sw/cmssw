#ifndef DETECTOR_DESCRIPTION_CORE_DD_PART_SELECTION_EXPANDER_H
#define DETECTOR_DESCRIPTION_CORE_DD_PART_SELECTION_EXPANDER_H

#include <cstddef>
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DataFormats/Math/interface/Graph.h"

class DDPartSelection;
struct DDPartSelectionLevel;


using PartSelectionTree = math::Graph<DDPartSelectionLevel,char>;
 

class DDPartSelectionExpander
{
public:
  explicit DDPartSelectionExpander(const DDCompactView &);
  
  size_t expand(const DDPartSelection & input, PartSelectionTree & result) const;
  
private:
  const DDCompactView::graph_type & graph_;
};

#endif
