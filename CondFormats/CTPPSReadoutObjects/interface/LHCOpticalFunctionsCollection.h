#ifndef CondFormats_CTPPSReadoutObjects_LHCOpticalFunctionsCollection_h
#define CondFormats_CTPPSReadoutObjects_LHCOpticalFunctionsCollection_h

// Original Author:  Jan Kašpar

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSet.h"

#include <unordered_map>

class CTPPSOpticalFunctionsESSource;

/**
 \brief Collection of optical functions for two crossing angle values and various scoring planes.
**/
class LHCOpticalFunctionsCollection
{
  public:
    using mapType = std::unordered_map<unsigned int, LHCOpticalFunctionsSet>;

    LHCOpticalFunctionsCollection() {}

    ~LHCOpticalFunctionsCollection() {}

    /// (half-)crossing-angle values in urad
    double getXangle1() const { return m_xangle1; }
    double getXangle2() const { return m_xangle2; }

    const mapType& getFunctions1() const { return m_functions1; }
    const mapType& getFunctions2() const { return m_functions2; }

    void interpolateFunctions(double xangle, mapType &output) const;

    void setXangle1(double v) { m_xangle1 = v; }
    void setXangle2(double v) { m_xangle2 = v; }
  
  private:
    friend CTPPSOpticalFunctionsESSource;

    double m_xangle1, m_xangle2;

    /// map: RP id --> optical functions
    mapType m_functions1, m_functions2;
    
    COND_SERIALIZABLE;
};

#endif
