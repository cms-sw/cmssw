#ifndef CondFormats_PhysicsToolsObjects_Histogram2D_h
#define CondFormats_PhysicsToolsObjects_Histogram2D_h

#include <utility>
#include <vector>
#include <cmath>

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"

namespace PhysicsTools {
namespace Calibration {

template<typename Value_t, typename AxisX_t = Value_t,
         typename AxisY_t = AxisX_t>
class Histogram2D {
    public:
	typedef Range<AxisX_t> RangeX;
	typedef Range<AxisY_t> RangeY;

	Histogram2D();

	Histogram2D(const Histogram2D &orig);

	template<typename OValue_t, typename OAxisX_t, typename OAxisY_t>
	Histogram2D(const Histogram2D<OValue_t, OAxisX_t, OAxisY_t> &orig);

	Histogram2D(const std::vector<AxisX_t> &binULimitsX,
	            const std::vector<AxisY_t> &binULimitsY);

	template<typename OAxisX_t, typename OAxisY_t>
	Histogram2D(const std::vector<OAxisX_t> &binULimitsX,
	            const std::vector<OAxisY_t> &binULimitsY);

	template<typename OAxisX_t, typename OAxisY_t>
	Histogram2D(const std::vector<OAxisX_t> &binULimitsX,
	            unsigned int nBinsY,
	            const PhysicsTools::Calibration::Range<OAxisY_t> &rangeY);

	template<typename OAxisX_t, typename OAxisY_t>
	Histogram2D(unsigned int nBinsX,
	            const PhysicsTools::Calibration::Range<OAxisX_t> &rangeX,
	            const std::vector<OAxisY_t> &binULimitsY);

	template<typename OAxisX_t, typename OAxisY_t>
	Histogram2D(unsigned int nBinsX,
	            const PhysicsTools::Calibration::Range<OAxisX_t> &rangeX,
	            unsigned int nBinsY,
	            const PhysicsTools::Calibration::Range<OAxisY_t> &rangeY);

	Histogram2D(unsigned int nBinsX, AxisX_t minX, AxisX_t maxX,
	            unsigned int nBinsY, AxisY_t minY, AxisY_t maxY);

	~Histogram2D();

	Histogram2D &operator = (const Histogram2D &orig);

	template<typename OValue_t, typename OAxisX_t, typename OAxisY_t>
	Histogram2D &operator = (const Histogram2D<OValue_t,
	                         OAxisX_t, OAxisY_t> &orig);

	void reset();

	const std::vector<AxisX_t> upperLimitsX() const { return binULimitsX; }
	const std::vector<AxisY_t> upperLimitsY() const { return binULimitsY; }

	inline int bin2D(int binX, int binY) const
	{ return binY * stride + binX; }

	Value_t binContent(int bin) const { return binValues[bin]; }
	Value_t binContent(int binX, int binY) const
	{ return binValues[bin2D(binX, binY)]; }
	Value_t value(AxisX_t x, AxisY_t y) const
	{ return binContent(findBin(x, y)); }
	Value_t normalizedValue(AxisX_t x, AxisY_t y) const
	{ return binContent(findBin(x, y)) / normalization(); }
	Value_t normalizedXValue(AxisX_t x, AxisY_t y) const;
	Value_t normalizedYValue(AxisX_t x, AxisY_t y) const;

	Value_t binError(int bin) const { return std::sqrt(binContent(bin)); }
	Value_t binError(int binX, int binY) const
	{ return binError(bin2D(binX, binY)); }
	Value_t error(AxisX_t x, AxisY_t y) const
	{ return binError(findBin(x, y)); }
	Value_t normalizedError(AxisX_t x, AxisY_t y) const
	{ return std::sqrt(binContent(findBin(x, y))) / normalization(); }
	Value_t normalizedXError(AxisX_t x, AxisY_t y) const;
	Value_t normalizedYError(AxisX_t x, AxisY_t y) const;

	void setBinContent(int bin, Value_t value);
	void setBinContent(int binX, int binY, Value_t value)
	{ setBinContent(bin2D(binX, binY), value); }
	void fill(AxisX_t x, AxisY_t y, Value_t weight = 1.0);

	bool empty() const { return binValues.empty(); }
	bool hasEquidistantBinsX() const { return binULimitsX.empty(); }
	bool hasEquidistantBinsY() const { return binULimitsY.empty(); }
	int numberOfBinsX() const { return stride - 2; }
	int numberOfBinsY() const { return binValues.size() / stride - 2; }
	int numberOfBins() const
	{ return numberOfBinsX() * numberOfBinsY(); }

	inline const std::vector<Value_t> &values() const
	{ return binValues; }

	void setValues(const std::vector<Value_t> &values);

	template<typename OValue_t>
	void setValues(const std::vector<OValue_t> &values);

	inline RangeX rangeX() const { return limitsX; }
	inline RangeY rangeY() const { return limitsY; }
	RangeX binRangeX(int binX) const;
	RangeY binRangeY(int binY) const;
	std::pair<RangeX, RangeY> binRange(int bin) const;
	std::pair<RangeX, RangeY> binRange(int binX, int binY) const
	{ return binRange(bin2D(binX, binY)); }

	int findBinX(AxisX_t x) const;
	int findBinY(AxisY_t y) const;
	int findBin(AxisX_t x, AxisY_t y) const
	{ return bin2D(findBinX(x), findBinY(y)); }
	Value_t normalization() const;
	Value_t normalizationX(int binY) const;
	Value_t normalizationY(int binX) const;

    protected:
	unsigned int			stride;
	std::vector<AxisX_t>		binULimitsX;
	std::vector<AxisY_t>		binULimitsY;
	std::vector<Value_t>		binValues;
	RangeX				limitsX;
	RangeY				limitsY;

	// transient cache variables
	mutable Value_t			total;
	mutable bool			totalValid;
	mutable std::vector<Value_t>	rowTotal;
	mutable std::vector<Value_t>	columnTotal;
};

typedef Histogram2D<float>  HistogramF2D;
typedef Histogram2D<double> HistogramD2D;

// wrap vectors of histograms so that CondDB can use them as top-level objects

struct VHistogramD2D {
  std::vector<PhysicsTools::Calibration::HistogramD2D>	vHist;
  std::vector<double> vValues;
};

} // namespace Calibration
} // namespace PhysicsTools

#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.icc"

#endif // CondFormats_PhysicsToolsObjects_Histogram2D_h
