/*
 * CAGraph.h
 *
 *  Created on: Aug 19, 2016
 *      Author: fpantale
 */

#ifndef CAGRAPH_H_
#define CAGRAPH_H_

#include <vector>
#include <array>
#include <string>
#include <queue>
#include <functional>
#include "CACell.h"

class CALayer
{
public:
	CALayer(const std::string& layerName, std::size_t numberOfHits )
	: theName(layerName)
	{
		isOuterHitOfCell.resize(numberOfHits);
	}

	bool operator==(const std::string& otherString)
	{
		return otherString == theName;
	}

	std::string name() const
	{
		return theName;
	}

	std::vector<int> theOuterLayerPairs;
	std::vector<int> theInnerLayerPairs;

	std::vector<int> theOuterLayers;
	std::vector<int> theInnerLayers;
	std::vector< std::vector<CACell*> >  isOuterHitOfCell;


private:

	std::string theName;
};

struct CALayerPair
{

	CALayerPair(int a, int b)

	{
		theLayers[0] = a;
		theLayers[1] = b;
	}



	bool operator==(const CALayerPair& otherLayerPair)
	{
		return (theLayers[0] == otherLayerPair.theLayers[0])
				&& (theLayers[1] == otherLayerPair.theLayers[1]);
	}

	std::array<int, 2> theLayers;
	std::vector<CACell> theFoundCells;

};

struct CAGraph
{
	std::vector<CALayer> theLayers;
	std::vector<CALayerPair> theLayerPairs;
	std::vector<int> theRootLayers;

};

#endif /* CAGRAPH_H_ */
