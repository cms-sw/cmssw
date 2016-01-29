/*
 * FKDTree.h
 *
 *  Created on: Jan 28, 2016
 *      Author: fpantale
 */

#ifndef FKDTREE_H_
#define FKDTREE_H_
#include <array>
#include <utility>

template <int numberOfDimensions>
struct KDRange
{

	using 1DRange = std::pair<float, float>;
	std::array<1DRange, numberOfDimensions> theKDRange;

  public:

  KDTreeBox(float d1min, float d1max,
	    float d2min, float d2max)
    : dim1min (d1min), dim1max(d1max)
    , dim2min (d2min), dim2max(d2max)
  {}

  KDTreeBox()
    : dim1min (0), dim1max(0)
    , dim2min (0), dim2max(0)
  {}
};



#endif /* CMSSW_8_0_0_PRE4_SRC_RECOPIXELVERTEXING_PIXELTRIPLETS_PLUGINS_FKDTREE_H_ */
