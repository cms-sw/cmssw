/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Alignment_RPDataFormats_RPAlignmentCorrectionsData
#define Alignment_RPDataFormats_RPAlignmentCorrectionsData

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"

#include <map>

namespace edm {
  class ParameterSet;
}

class AlignmentGeometry;


/**
 *\brief Container for RP alignment corrections.
 * The corrections are stored on two levels - RP and sensor. For every level,
 * there is a map: symbolic ID --> alignment correction. Sensors inherit the
 * alignment corrections for the corresponding RP, see GetFullSensorCorrection
 * method.
 **/
class RPAlignmentCorrectionsData
{
  public:
    /// map: element id -> its alignment correction
    typedef std::map<unsigned int, RPAlignmentCorrectionData> mapType;
    
  private:
    /// alignment correction maps
    mapType rps, sensors;
  
    friend class StraightTrackAlignment;
  
  public:
    RPAlignmentCorrectionsData() {}

    /// returns the map of RP alignment corrections
    const mapType& GetRPMap() const
      { return rps; }
    
    /// returns the map of sensor alignment corrections
    const mapType& GetSensorMap() const
      { return sensors; }
  
    /// returns the correction value from the RP map
    RPAlignmentCorrectionData& GetRPCorrection(unsigned int id);
  
    RPAlignmentCorrectionData GetRPCorrection(unsigned int id) const;
    
    /// returns the correction value from the sensor map
    RPAlignmentCorrectionData& GetSensorCorrection(unsigned int id);
  
    RPAlignmentCorrectionData GetSensorCorrection(unsigned int id) const;
    
    /// returns the correction for the given sensor, combining the data from RP and sensor map
    /// regarding transverse shifts, uses the x and y representation, sh_r will not be corrected!
    /// by default, RP errors shall not be summed up (see the note at FactorRPFromSensorCorrections).
    RPAlignmentCorrectionData GetFullSensorCorrection(unsigned int id, bool useRPErrors = false) const;
  
    /// sets the alignment correction for the given RP
    void SetRPCorrection(unsigned int id, const RPAlignmentCorrectionData& ac);
    
    /// sets the alignment correction for the given sensor
    void SetSensorCorrection(unsigned int id, const RPAlignmentCorrectionData& ac);
  
    /// adds (merges) a RP correction on top of the current value
    /// \param sumErrors if it is true, old and new alignment uncertainties are summed (in quadrature)
    /// if it is false, the uncertainties of the parameter (i.e. not the object) will be used
    /// With the add... switches one can control which corrections are added.
    void AddRPCorrection(unsigned int, const RPAlignmentCorrectionData&, bool sumErrors = true,
      bool addShR=true, bool addShZ=true, bool addRotZ=true);
    
    /// adds (merges) a RP correction on top of the current value
    void AddSensorCorrection(unsigned int, const RPAlignmentCorrectionData&, bool sumErrors = true,
      bool addShR=true, bool addShZ=true, bool addRotZ=true);
  
    /// adds (merges) corrections on top of the current values
    void AddCorrections(const RPAlignmentCorrectionsData &, bool sumErrors = true,
      bool addShR=true, bool addShZ=true, bool addRotZ=true);

    /// clears all alignments
    void Clear();

};

#endif

