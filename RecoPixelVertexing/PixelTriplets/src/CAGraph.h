#ifndef RECOPIXELVERTEXING_PIXELTRIPLETS_CAGRAPH_H_
#define RECOPIXELVERTEXING_PIXELTRIPLETS_CAGRAPH_H_

#include <vector>
#include <array>
#include <string>

struct CALayer
{
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
	std::vector< std::vector<unsigned int> >  isOuterHitOfCell;


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
	std::array<unsigned int, 2> theFoundCells= {{0,0}};
};

struct CAGraph
{
	std::vector<CALayer> theLayers;
	std::vector<CALayerPair> theLayerPairs;
	std::vector<int> theRootLayers;
};

#endif /* CAGRAPH_H_ */
