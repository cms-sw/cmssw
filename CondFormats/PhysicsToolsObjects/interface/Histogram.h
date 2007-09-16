#ifndef CondFormats_PhysicsToolsObjects_Histogram_h
#define CondFormats_PhysicsToolsObjects_Histogram_h

#include <vector>

namespace PhysicsTools {
namespace Calibration {

class Histogram {
    public:
	struct Range {
		Range() {}
		Range(const Range &orig) : min(orig.min), max(orig.max) {}
		Range(float min, float max) : min(min), max(max) {}
		~Range() {}

		inline float width() const { return max - min; }

		float min, max;
	};

	Histogram();
	Histogram(const Histogram &orig);
	Histogram(std::vector<float> binLimits);
	Histogram(unsigned int nBins, Range range);
	Histogram(unsigned int nBins, float min, float max);
	~Histogram();

	Histogram &operator = (const Histogram &orig);

	void normalize();
	void reset();

	float binContent(int bin) const { return binValues[bin]; }
	float value(float x) const { return binContent(findBin(x)); }

	void setBinContent(int bin, float value);
	void fill(float x, float weight = 1.0);

	bool hasEquidistantBins() const { return binLimits.empty(); }
	int getNBins() const { return binValues.size() - 2; }

	inline const std::vector<float> &getValueArray() const
	{ return binValues; }
	inline std::vector<float> &getValueArray()
	{ return binValues; }
	inline Range getRange() const { return range; }
	Range getBinRange(int bin) const;

	int findBin(float x) const;
	inline float getIntegral() const { return total; }

	float integral(float hBound, float lBound = 0.0, int mode = 2) const;
	float normalizedIntegral(float hBound, float lBound = 0.0, int mode = 2) const
	{ return integral(hBound, lBound, mode) / total; }

    protected:
	std::vector<float>	binLimits;
	std::vector<float>	binValues;
	Range			range;
	float			total;
};

} // namespace Calibration
} // namespace PhysicsTools

#endif // CondFormats_PhysicsToolsObjects_Histogram_h
