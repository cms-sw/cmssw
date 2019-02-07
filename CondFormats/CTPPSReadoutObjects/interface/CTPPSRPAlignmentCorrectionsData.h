/****************************************************************************
 *
 * This is a part of CMS-TOTEM PPS offline software.
 * Authors:
 *  Jan Kašpar (jan.kaspar@gmail.com)
 *  Helena Malbouisson
 *  Clemencia Mora Herrera
 *
 ****************************************************************************/

#ifndef CondFormats_CTPPSReadoutObjects_CTPPSRPAlignmentCorrectionsData
#define CondFormats_CTPPSReadoutObjects_CTPPSRPAlignmentCorrectionsData

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionData.h"

#include <map>

/**
 *\brief Container for CTPPS RP alignment corrections.
 * The corrections are stored on two levels - RP and sensor. For every level,
 * there is a map: ID --> alignment correction. Sensors inherit the
 * alignment corrections from the corresponding RP, see getFullSensorCorrection
 * method.
 **/
class CTPPSRPAlignmentCorrectionsData
{
  public:
    /// map: element id -> its alignment correction
    typedef std::map<unsigned int, CTPPSRPAlignmentCorrectionData> mapType;

  private:
    /// alignment correction maps
    mapType rps_, sensors_;

    friend class StraightTrackAlignment;

  public:
    CTPPSRPAlignmentCorrectionsData() {}

    /// returns the map of RP alignment corrections
    const mapType& getRPMap() const { return rps_; }

    /// returns the map of sensor alignment corrections
    const mapType& getSensorMap() const { return sensors_; }

    /// returns the correction value from the RP map
    CTPPSRPAlignmentCorrectionData& getRPCorrection( unsigned int id );
    CTPPSRPAlignmentCorrectionData getRPCorrection( unsigned int id ) const;

    /// returns the correction value from the sensor map
    CTPPSRPAlignmentCorrectionData& getSensorCorrection( unsigned int id );
    CTPPSRPAlignmentCorrectionData getSensorCorrection( unsigned int id ) const;

    /// returns the correction for the given sensor, combining the data from RP and sensor map
    /// regarding transverse shifts, uses the x and y representation, sh_r will not be corrected!
    /// by default, RP errors shall not be summed up (strong correlation).
    CTPPSRPAlignmentCorrectionData getFullSensorCorrection( unsigned int id, bool useRPErrors = false ) const;

    /// sets the alignment correction for the given RP
    void setRPCorrection( unsigned int id, const CTPPSRPAlignmentCorrectionData& ac );

    /// sets the alignment correction for the given sensor
    void setSensorCorrection( unsigned int id, const CTPPSRPAlignmentCorrectionData& ac );

    /// adds (merges) a RP correction on top of the current value
    /// \param sumErrors if it is true, old and new alignment uncertainties are summed (in quadrature)
    /// if it is false, the uncertainties of the parameter (i.e. not the object) will be used
    /// With the add... switches one can control which corrections are added.
    void addRPCorrection( unsigned int, const CTPPSRPAlignmentCorrectionData&, bool sumErrors=true, bool addSh=true, bool addRot=true );

    /// adds (merges) a RP correction on top of the current value
    void addSensorCorrection( unsigned int, const CTPPSRPAlignmentCorrectionData&, bool sumErrors=true, bool addSh=true, bool addRot=true );

    /// adds (merges) corrections on top of the current values
    void addCorrections( const CTPPSRPAlignmentCorrectionsData &, bool sumErrors=true, bool addSh=true, bool addRot=true );

    /// clears all alignments
    void clear();

    COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& s, const CTPPSRPAlignmentCorrectionsData &corr);

#endif
