/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Filip Dej
 *
 ****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemTimingParser
#define RecoCTPPS_TotemRPLocal_TotemTimingParser

#include <iostream>
#include <map>
#include <vector>

class TotemTimingParser
{
  public:
    TotemTimingParser() = default;
    ~TotemTimingParser() = default;

    void parseFile(const std::string& file_name);
    void print();

    std::vector<double> getParameters(int db, int sampic, int channel, int cell) const;
    inline const std::string& getFormula() const { return formula_; }
    double getTimeOffset(int db, int sampic, int channel) const;
    double getTimePrecision(int db, int sampic, int channel) const;

    friend std::ostream& operator<<(std::ostream&, const TotemTimingParser&);

  private:
    //--------------------------------------------------------------------------
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
    };
    friend std::ostream& operator<<(std::ostream&, const TotemTimingParser::CalibrationKey&);
    //--------------------------------------------------------------------------

    std::string formula_;
    std::map<CalibrationKey,std::vector<double> > parameters_;
    std::map<CalibrationKey,std::pair<double,double> > timeInfo_;
};

#endif

