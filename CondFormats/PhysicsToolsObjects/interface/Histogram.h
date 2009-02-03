#ifndef CondFormats_PhysicsToolsObjects_Histogram_h
#define CondFormats_PhysicsToolsObjects_Histogram_h

#include <vector>
#include <cmath>

namespace PhysicsTools {
namespace Calibration {

template<typename Axis_t>
struct Range {
	inline Range() {}

	template<typename OAxis_t>
	inline Range(const Range<OAxis_t> &orig) :
		min(orig.min), max(orig.max) {}

	inline Range(Axis_t min, Axis_t max) : min(min), max(max) {}

	virtual ~Range() {}

	inline Axis_t width() const { return max - min; }

	Axis_t	min, max;
};

template<typename Value_t, typename Axis_t = Value_t>
class Histogram {
    public:
	typedef PhysicsTools::Calibration::Range<Axis_t> Range;

	Histogram();

	Histogram(const Histogram &orig);

	template<typename OValue_t, typename OAxis_t>
	Histogram(const Histogram<OValue_t, OAxis_t> &orig);

	Histogram(const std::vector<Axis_t> &binULimits);

	template<typename OAxis_t>
	Histogram(const std::vector<OAxis_t> &binULimits);

	template<typename OAxis_t>
	Histogram(unsigned int nBins,
	          const PhysicsTools::Calibration::Range<OAxis_t> &range);

	Histogram(unsigned int nBins, Axis_t min, Axis_t max);

	~Histogram();

	Histogram &operator = (const Histogram &orig);

	template<typename OValue_t, typename OAxis_t>
	Histogram &operator = (const Histogram<OValue_t, OAxis_t> &orig);

	void reset();

	const std::vector<Axis_t> upperLimits() const { return binULimits; }

	Value_t binContent(int bin) const { return binValues[bin]; }
	Value_t value(Axis_t x) const { return binContent(findBin(x)); }
	Value_t normalizedValue(Axis_t x) const
	{ return binContent(findBin(x)) / normalization(); }

	Value_t binError(int bin) const { return std::sqrt(binContent(bin)); }
	Value_t error(Axis_t x) const { return binError(findBin(x)); }
	Value_t normalizedError(Axis_t x) const
	{ return std::sqrt(binContent(findBin(x))) / normalization(); }

	void setBinContent(int bin, Value_t value);
	void fill(Axis_t x, Value_t weight = 1.0);

	bool empty() const { return binValues.empty(); }
	bool hasEquidistantBins() const { return binULimits.empty(); }
	int numberOfBins() const { return binValues.size() - 2; }

	inline const std::vector<Value_t> &values() const
	{ return binValues; }

	void setValues(const std::vector<Value_t> &values);

	template<typename OValue_t>
	void setValues(const std::vector<OValue_t> &values);

	inline Range range() const { return limits; }
	Range binRange(int bin) const;

	int findBin(Axis_t x) const;
	Value_t normalization() const;

	Value_t integral(Axis_t hBound, Axis_t lBound = 0.0, int mode = 1) const;
	Value_t normalizedIntegral(Axis_t hBound, Axis_t lBound = 0.0, int mode = 1) const
	{ return integral(hBound, lBound, mode) / normalization(); }

    protected:
	std::vector<Axis_t>	binULimits;
	std::vector<Value_t>	binValues;
	Range			limits;

	// transient cache variables
	mutable Value_t		total;
	mutable bool		totalValid;
};

typedef Histogram<float>  HistogramF;
typedef Histogram<double> HistogramD;

} // namespace Calibration
} // namespace PhysicsTools

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.icc"

#endif // CondFormats_PhysicsToolsObjects_Histogram_h
