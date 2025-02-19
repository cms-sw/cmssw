#ifndef CondFormats_PhysicsToolsObjects_Histogram3D_h
#define CondFormats_PhysicsToolsObjects_Histogram3D_h

#include <utility>
#include <vector>
#include <cmath>

#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"

namespace PhysicsTools {
namespace Calibration {

//template<typename Value_t, typename AxisX_t = Value_t,
//         typename AxisY_t = AxisX_t>

template<typename Value_t, typename AxisX_t = Value_t,
         typename AxisY_t = AxisX_t, typename AxisZ_t = AxisX_t>


class Histogram3D {
    public:
	typedef Range<AxisX_t> RangeX;
	typedef Range<AxisY_t> RangeY;
        typedef Range<AxisZ_t> RangeZ;

	Histogram3D();

	Histogram3D(const Histogram3D &orig);

	template<typename OValue_t, typename OAxisX_t, typename OAxisY_t, typename OAxisZ_t>
	Histogram3D(const Histogram3D<OValue_t, OAxisX_t, OAxisY_t,  OAxisZ_t> &orig);

	Histogram3D(const std::vector<AxisX_t> &binULimitsX,
	            const std::vector<AxisY_t> &binULimitsY,
                    const std::vector<AxisZ_t> &binULimitsZ);

	template<typename OAxisX_t, typename OAxisY_t, typename OAxisZ_t>
	Histogram3D(const std::vector<OAxisX_t> &binULimitsX,
	            const std::vector<OAxisY_t> &binULimitsY,
                    const std::vector<OAxisZ_t> &binULimitsZ);

/*
	//TO BE CONVERTED FROM HISTO2D TO HISTO3D
	template<typename OAxisX_t, typename OAxisY_t>
	Histogram3D(const std::vector<OAxisX_t> &binULimitsX,
	            unsigned int nBinsY,
	            const PhysicsTools::Calibration::Range<OAxisY_t> &rangeY);

	template<typename OAxisX_t, typename OAxisY_t>
	Histogram3D(unsigned int nBinsX,
	            const PhysicsTools::Calibration::Range<OAxisX_t> &rangeX,
	            const std::vector<OAxisY_t> &binULimitsY);

	template<typename OAxisX_t, typename OAxisY_t>
	Histogram3D(unsigned int nBinsX,
	            const PhysicsTools::Calibration::Range<OAxisX_t> &rangeX,
	            unsigned int nBinsY,
	            const PhysicsTools::Calibration::Range<OAxisY_t> &rangeY);
*/
	Histogram3D(unsigned int nBinsX, AxisX_t minX, AxisX_t maxX,
	            unsigned int nBinsY, AxisY_t minY, AxisY_t maxY,
                    unsigned int nBinsZ, AxisZ_t minZ, AxisZ_t maxZ);

	~Histogram3D();

	Histogram3D &operator = (const Histogram3D &orig);

	template<typename OValue_t, typename OAxisX_t, typename OAxisY_t, typename OAxisZ_t>
	Histogram3D &operator = (const Histogram3D<OValue_t,
	                         OAxisX_t, OAxisY_t,  OAxisZ_t> &orig);

	void reset();

	const std::vector<AxisX_t> upperLimitsX() const { return binULimitsX; }
	const std::vector<AxisY_t> upperLimitsY() const { return binULimitsY; }
        const std::vector<AxisZ_t> upperLimitsZ() const { return binULimitsZ; }

	inline int bin3D(int binX, int binY, int binZ) const
//        { return (((binY * strideY) + binX) * strideX) + binZ; }
        { return binZ*strideX*strideY + binY*strideX + binX; }



	Value_t binContent(int bin) const { return binValues[bin]; }
	Value_t binContent(int binX, int binY, int binZ) const
	{ return binValues[bin3D(binX, binY, binZ)]; }
	Value_t value(AxisX_t x, AxisY_t y, AxisY_t z) const
	{ return binContent(findBin(x, y, z)); }
	Value_t normalizedValue(AxisX_t x, AxisY_t y, AxisZ_t z) const
	{ return binContent(findBin(x, y, z)) / normalization(); }

        //TO BE CONVERTED FROM HISTO2D TO HISTO3D
//	Value_t normalizedXValue(AxisX_t x, AxisY_t y, AxisZ_t z) const;
//	Value_t normalizedYValue(AxisX_t x, AxisY_t y, AxisZ_t z) const;
//      Value_t normalizedZValue(AxisX_t x, AxisY_t y, AxisZ_t z) const;

	Value_t binError(int bin) const { return std::sqrt(binContent(bin)); }
	Value_t binError(int binX, int binY, int binZ) const
	{ return binError(bin3D(binX, binY, binZ)); }
	Value_t error(AxisX_t x, AxisY_t y, AxisZ_t z) const
	{ return binError(findBin(x, y, z)); }
	Value_t normalizedError(AxisX_t x, AxisY_t y, AxisY_t z) const
	{ return std::sqrt(binContent(findBin(x, y, z))) / normalization(); }

        //TO BE CONVERTED FROM HISTO2D TO HISTO3D
//	Value_t normalizedXError(AxisX_t x, AxisY_t y, AxisZ_t z) const;
//	Value_t normalizedYError(AxisX_t x, AxisY_t y, AxisZ_t z) const;

	void setBinContent(int bin, Value_t value);
	void setBinContent(int binX, int binY, int binZ, Value_t value)
	{ setBinContent(bin3D(binX, binY, binZ), value); }
	void fill(AxisX_t x, AxisY_t y, AxisZ_t z, Value_t weight = 1.0);

	bool empty() const { return binValues.empty(); }
	bool hasEquidistantBinsX() const { return binULimitsX.empty(); }
	bool hasEquidistantBinsY() const { return binULimitsY.empty(); }
        bool hasEquidistantBinsZ() const { return binULimitsZ.empty(); }
	int numberOfBinsX() const { return strideX - 2; }
	int numberOfBinsY() const { return strideY - 2; }
        int numberOfBinsZ() const { return binValues.size() / (strideX * strideY) - 2; }
	int numberOfBins() const
	{ return numberOfBinsX() * numberOfBinsY() * numberOfBinsZ(); }

	inline const std::vector<Value_t> &values() const
	{ return binValues; }

	void setValues(const std::vector<Value_t> &values);

	template<typename OValue_t>
	void setValues(const std::vector<OValue_t> &values);

	inline RangeX rangeX() const { return limitsX; }
	inline RangeY rangeY() const { return limitsY; }
        inline RangeZ rangeZ() const { return limitsZ; }
	RangeX binRangeX(int binX) const;
	RangeY binRangeY(int binY) const;
        RangeZ binRangeZ(int binZ) const;

        //TO BE CONVERTED FROM HISTO2D TO HISTO3D
/*	std::pair<RangeX, RangeY> binRange(int bin) const;
	std::pair<RangeX, RangeY> binRange(int binX, int binY) const
	{ return binRange(bin2D(binX, binY)); }
*/
	int findBinX(AxisX_t x) const;
	int findBinY(AxisY_t y) const;
        int findBinZ(AxisZ_t z) const;
	int findBin(AxisX_t x, AxisY_t y, AxisZ_t z) const
	{ return bin3D(findBinX(x), findBinY(y), findBinZ(z)); }
	Value_t normalization() const;

        //TO BE CONVERTED FROM HISTO2D TO HISTO3D
//	Value_t normalizationX(int binY) const;
//	Value_t normalizationY(int binX) const;

    protected:
	unsigned int			strideX;
        unsigned int                    strideY;
	std::vector<AxisX_t>		binULimitsX;
	std::vector<AxisY_t>		binULimitsY;
        std::vector<AxisZ_t>            binULimitsZ;
	std::vector<Value_t>		binValues;
	RangeX				limitsX;
	RangeY				limitsY;
        RangeY                          limitsZ;

	// transient cache variables
	mutable Value_t			total;
	mutable bool			totalValid;
        mutable std::vector<Value_t>    sliceTotal;
	mutable std::vector<Value_t>	rowTotal;
	mutable std::vector<Value_t>	columnTotal;
};

typedef Histogram3D<float>  HistogramF3D;
typedef Histogram3D<double> HistogramD3D;


} // namespace Calibration
} // namespace PhysicsTools

#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.icc"

#endif // CondFormats_PhysicsToolsObjects_Histogram3D_h
