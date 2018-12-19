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
      int db, sampic, channel, cell;
      /// Comparison operator
      bool operator<(const CalibrationKey& rhs) const {
        if (db == rhs.db) {
          if (sampic == rhs.sampic) {
            if (channel == rhs.channel)
              return cell < rhs.cell;
            return channel < rhs.channel;
          }
          return sampic < rhs.sampic;
        }
        return db < rhs.db;
      }
      CalibrationKey(int db = -1, int sampic = -1, int channel = -1, int cell = -1) :
        db(db), sampic(sampic), channel(channel), cell(cell) {}
      friend std::ostream& operator<<(std::ostream& os, const CalibrationKey& key) {
        return os << key.db << " " << key.sampic << " " << key.channel << " " << key.cell;
      }
    };
    //--------------------------------------------------------------------------

    typedef std::map<CalibrationKey,std::vector<double> > ParametersMap;
    typedef std::map<CalibrationKey,std::pair<double,double> > TimingMap;

    PPSTimingCalibration() = default;
    PPSTimingCalibration( const std::string& formula, const ParametersMap& params, const TimingMap& timeinfo ) :
      formula_(formula), parameters_(params), timeInfo_(timeinfo) {}
    ~PPSTimingCalibration() = default;

    inline std::vector<double> getParameters(int db, int sampic, int channel, int cell) const {
      CalibrationKey key = CalibrationKey(db, sampic, channel, cell);
      auto out = parameters_.find(key);
      if (out == parameters_.end())
        return {};
      else
        return out->second;
    }
    inline const std::string& getFormula() const { return formula_; }
    inline double getTimeOffset(int db, int sampic, int channel) const {
      CalibrationKey key = CalibrationKey(db, sampic, channel);
      auto out = timeInfo_.find(key);
      if (out == timeInfo_.end())
        return 0.;
      else
        return out->second.first;
    }
    inline double getTimePrecision(int db, int sampic, int channel) const {
      CalibrationKey key = CalibrationKey(db, sampic, channel);
      auto out = timeInfo_.find(key);
      if (out == timeInfo_.end())
        return 0.;
      else
        return out->second.second;
    }

    friend std::ostream& operator<<(std::ostream& os, const PPSTimingCalibration& data) {
      os << "FORMULA: "<< data.formula_ << "\nDB SAMPIC CHANNEL CELL PARAMETERS TIME_OFFSET\n";
      for (const auto& kv : data.parameters_) {
        os << kv.first <<" [";
        for (size_t i = 0; i < kv.second.size(); ++i)
          os << (i > 0 ? ", " : "") << kv.second.at(i);
        CalibrationKey k = kv.first;
        k.cell = -1;
        os << "] " << data.timeInfo_.at(k).first << " " <<  data.timeInfo_.at(k).second << "\n";
      }
      return os;
    }

  private:
    friend PPSTimingCalibrationESSource;

    std::string formula_;
    ParametersMap parameters_;
    TimingMap timeInfo_;

    COND_SERIALIZABLE;
};

#endif

