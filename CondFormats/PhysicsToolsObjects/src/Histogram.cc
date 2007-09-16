#include <cstddef>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"

namespace { // anonymous
	template<typename T>
	inline const T &clamp(const T &min, const T &val, const T &max)
	{
		if (val < min)
			return min;
		if (val > max)
			return max;
		return val;
	}
} // anonymous namespace

namespace PhysicsTools {
namespace Calibration {

Histogram::Histogram() :
	range(0, 0), total(0.0)
{
}

Histogram::Histogram(const Histogram &orig) :
	binLimits(orig.binLimits), binValues(orig.binValues),
	range(orig.range), total(orig.total)
{
}

Histogram::Histogram(std::vector<float> binLimits) :
	binLimits(binLimits), binValues(binLimits.size() + 1),
	range(0, 0), total(0.0)
{
	if (binLimits.size() < 2)
		throw cms::Exception("TooFewBinsError")
			<< "Trying to generate degenerate histogram: "
			<< "Fewer than one bin requested.";

	range.min = binLimits.front();
	range.max = binLimits.back();
}

Histogram::Histogram(unsigned int nBins, Range range) :
	binValues(nBins + 2), range(range), total(0.0)
{
	if (!nBins)
		throw cms::Exception("TooFewBinsError")
			<< "Trying to generate degenerate histogram: "
			<< "Fewer than one bin requested.";
}

Histogram::Histogram(unsigned int nBins, float min, float max) :
	binValues(nBins + 2), range(min, max), total(0.0)
{
	if (!nBins)
		throw cms::Exception("TooFewBinsError")
			<< "Trying to generate degenerate histogram: "
			<< "Fewer than one bin requested.";
}

Histogram::~Histogram()
{
}

Histogram &Histogram::operator = (const Histogram &orig)
{
	binLimits = orig.binLimits;
	binValues = orig.binValues;
	range = orig.range;
	total = orig.total;
	return *this;
}

void Histogram::normalize()
{
	total = std::accumulate(binValues.begin() + 1,
	                        binValues.end() - 1, 0.0); 
}

void Histogram::reset()
{
	std::fill(binValues.begin(), binValues.end(), 0.0);
	total = 0.0;
}

void Histogram::setBinContent(int bin, float value)
{
	if (bin < 0 || (unsigned int)bin >= binValues.size())
		throw cms::Exception("RangeError")
			<< "Histogram bin " << bin << " out of range "
			<< "[0, " << (binValues.size() - 1) << "].";

	binValues[bin] = value;
}

void Histogram::fill(float x, float weight)
{
	int bin = findBin(x);
	binValues[bin] += weight;
}

Histogram::Range Histogram::getBinRange(int bin) const
{
	std::size_t size = binValues.size();
	if (bin < 1 || (unsigned int)bin > size - 2)
		throw cms::Exception("RangeError")
			<< "Histogram bin " << bin << " out of range "
			<< "[1, " << (size - 2) << "].";

	if (hasEquidistantBins()) {
		float min = (float)(bin - 1) / (size - 2);
		float max = (float)bin / (size - 2);
		min *= range.width();
		min += range.min;
		max *= range.width();
		max += range.min;
		return Range(min, max);
	} else
		return Range(binLimits[bin - 1], binLimits[bin]);
}

int Histogram::findBin(float x) const
{
	if (hasEquidistantBins()) {
		std::size_t size = binValues.size();
		x -= range.min;
		x *= size - 2;
		x /= range.width();
		return clamp(0, (int)(std::floor(x) + 1.5), (int)size - 1);
	} else
		return std::upper_bound(binLimits.begin(),
		                        binLimits.end(), x) -
		       binLimits.begin();
}

float Histogram::integral(float hBound, float lBound, int mode) const
{
	if (hBound < lBound)
		throw cms::Exception("InvalidIntervalError")
			<< "Upper boundary below lower boundary in "
			<< "histogram integral.";

	std::size_t size = binValues.size();
	int lBin = clamp(1, findBin(lBound), (int)size - 2);
	int hBin = clamp(1, findBin(hBound), (int)size - 2);

	double sum = 0.0;
	Range lBinRange, hBinRange;

	if (hBin > lBin)
		sum = std::accumulate(binValues.begin() + (lBin + 1),
		                      binValues.begin() + hBin, 0.0); 

	if (hasEquidistantBins()) {
		double binWidth = range.width() / (size - 2);
		lBinRange = Range((lBin - 1) * binWidth, lBin * binWidth);
		hBinRange = Range((hBin - 1) * binWidth, hBin * binWidth);
	} else {
		lBinRange = Range(binLimits[lBin - 1], binLimits[lBin]);
		hBinRange = Range(binLimits[hBin - 1], binLimits[hBin]);
	}

	switch(mode) {
	    case 0:
		break;
	    case 1:
		sum += binValues[lBin] * (lBinRange.max - lBound)
		                       / lBinRange.width();
		sum += binValues[hBin] * (hBound - hBinRange.min)
		                       / hBinRange.width();
	    default:
		throw cms::Exception("InvalidMode")
			<< "Invalid mode " << mode << " in "
			<< "Histogram::integral()";
	}

	return sum;
}

} // namespace Calibration
} // namespace PhysicsTools
