/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Filip Dej
 *
 * NOTE:
 *   Given implementation handles calibration files in JSON format,
 *   which can be generated using dedicated python script.
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingParser.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

//------------------------------------------------------------------------------
// Extracts the data from json file
// Arguments:
//   file_name: path to the calibration file

void
TotemTimingParser::parseFile(const std::string& file_name)
{
  pt::ptree node;
  pt::read_json(file_name, node);
  formula_ = node.get<std::string>("formula");

  for (pt::ptree::value_type &par : node.get_child("parameters")) {
    CalibrationKey key = CalibrationKey();
    key.db = (int) strtol(par.first.data(), nullptr, 10);

    for (pt::ptree::value_type &board : par.second) {
      key.sampic = board.second.get<int>("sampic");
      key.channel = board.second.get<int>("channel");
      double timeOffset = board.second.get<double>("time_offset");
      double timePrecision = board.second.get<double>("time_precision");
      key.cell = -1;
      timeInfo_[key] = {timeOffset, timePrecision};

      int cell_ct = 0;
      for (pt::ptree::value_type &cell : board.second.get_child("cells")) {
        std::vector<double> values;
        key.cell = cell_ct;

        for (pt::ptree::value_type& param : cell.second)
          values.push_back(std::stod(param.second.data(), nullptr));
        parameters_[key] = values;
        cell_ct++;
      }
    }
  }
}

//------------------------------------------------------------------------------

std::vector<double>
TotemTimingParser::getParameters(int db, int sampic, int channel, int cell) const
{
  CalibrationKey key = CalibrationKey(db, sampic, channel, cell);
  auto out = parameters_.find(key);
  if (out == parameters_.end())
    return {};
  else
    return out->second;
}

//------------------------------------------------------------------------------

double
TotemTimingParser::getTimeOffset(int db, int sampic, int channel) const
{
  CalibrationKey key = CalibrationKey(db, sampic, channel);
  auto out = timeInfo_.find(key);
  if (out == timeInfo_.end())
    return 0.;
  else
    return out->second.first;
}

//------------------------------------------------------------------------------

double
TotemTimingParser::getTimePrecision(int db, int sampic, int channel) const
{
  CalibrationKey key = CalibrationKey(db, sampic, channel);
  auto out = timeInfo_.find(key);
  if (out == timeInfo_.end())
    return 0.;
  else
    return out->second.second;
}

//------------------------------------------------------------------------------

std::ostream&
operator<<(std::ostream& os, const TotemTimingParser::CalibrationKey& key)
{
  return os << key.db << " " << key.sampic << " " << key.channel << " " << key.cell;
}

std::ostream&
operator<<(std::ostream& os, const TotemTimingParser& data)
{
  os << "FORMULA: "<< data.formula_ << "\nDB SAMPIC CHANNEL CELL PARAMETERS TIME_OFFSET\n";
  for (const auto& kv : data.parameters_) {
    os << kv.first <<" [";
    for (size_t i = 0; i < kv.second.size(); ++i)
      os << (i > 0 ? ", " : "") << kv.second.at(i);
    TotemTimingParser::CalibrationKey k = kv.first;
    k.cell = -1;
    os << "] " << data.timeInfo_.at(k).first << " " <<  data.timeInfo_.at(k).second << "\n";
  }
  return os;
}

