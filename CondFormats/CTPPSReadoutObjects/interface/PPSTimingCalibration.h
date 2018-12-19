/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Filip Dej
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef CondFormats_CTPPSReadoutObjects_PPSTimingCalibration_h
#define CondFormats_CTPPSReadoutObjects_PPSTimingCalibration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <vector>

class PPSTimingCalibrationESSource;
class PPSTimingCalibration
{
  public:
    /// Helper structure for storing calibration data
    struct CalibrationKey
    {
      CalibrationKey( int db = -1, int sampic = -1, int channel = -1, int cell = -1 ) :
        db( db ), sampic( sampic ), channel( channel ), cell( cell ) {}
      /// Comparison operator
      bool operator<( const CalibrationKey& rhs ) const;
      friend std::ostream& operator<<( std::ostream& os, const CalibrationKey& key );

      int db, sampic, channel, cell;
    };
    //--------------------------------------------------------------------------

    typedef std::map<CalibrationKey,std::vector<double> > ParametersMap;
    typedef std::map<CalibrationKey,std::pair<double,double> > TimingMap;

    PPSTimingCalibration() = default;
    PPSTimingCalibration( const std::string& formula, const ParametersMap& params, const TimingMap& timeinfo ) :
      formula_( formula ), parameters_( params ), timeInfo_( timeinfo ) {}
    ~PPSTimingCalibration() = default;

    std::vector<double> getParameters( int db, int sampic, int channel, int cell ) const;
    inline const std::string& getFormula() const { return formula_; }
    double getTimeOffset( int db, int sampic, int channel ) const;
    double getTimePrecision( int db, int sampic, int channel ) const;

    friend std::ostream& operator<<( std::ostream& os, const PPSTimingCalibration& data );

  private:
    friend PPSTimingCalibrationESSource;

    std::string formula_;
    ParametersMap parameters_;
    TimingMap timeInfo_;

    COND_SERIALIZABLE;
};

#endif

