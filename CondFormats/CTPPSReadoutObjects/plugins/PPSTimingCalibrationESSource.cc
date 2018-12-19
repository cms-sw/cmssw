/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Filip Dej
 *   Laurent Forthomme
 *
 * NOTE:
 *   Given implementation handles calibration files in JSON format,
 *   which can be generated using dedicated python script.
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

//------------------------------------------------------------------------------

class PPSTimingCalibrationESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
  public:
    PPSTimingCalibrationESSource( const edm::ParameterSet& );
    ~PPSTimingCalibrationESSource() override = default;

  private:
    edm::ESProducts<std::unique_ptr<PPSTimingCalibration> > produce();
    /// Extracts calibration data from JSON file
    void parseJsonFile();

    const std::string filename_;
    PPSTimingCalibration calib_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationESSource::PPSTimingCalibrationESSource( const edm::ParameterSet& iConfig ) :
  filename_( iConfig.getParameter<edm::FileInPath>( "filename" ).fullPath() )
{
  parseJsonFile();
}

//------------------------------------------------------------------------------

void
PPSTimingCalibrationESSource::parseJsonFile()
{
  pt::ptree node;
  pt::read_json( filename_, node );

  const std::string formula = node.get<std::string>( "formula" );
  PPSTimingCalibration::ParametersMap params;
  PPSTimingCalibration::TimingMap time_info;

  for ( pt::ptree::value_type& par : node.get_child( "parameters" ) ) {
    PPSTimingCalibration::CalibrationKey key;
    key.db = (int)strtol( par.first.data(), nullptr, 10 );

    for (pt::ptree::value_type &board : par.second) {
      key.sampic = board.second.get<int>( "sampic" );
      key.channel = board.second.get<int>( "channel" );
      double timeOffset = board.second.get<double>( "time_offset" );
      double timePrecision = board.second.get<double>( "time_precision" );
      key.cell = -1;
      time_info[key] = { timeOffset, timePrecision };

      int cell_ct = 0;
      for ( pt::ptree::value_type &cell : board.second.get_child( "cells" ) ) {
        std::vector<double> values;
        key.cell = cell_ct;

        for ( pt::ptree::value_type& param : cell.second )
          values.emplace_back( std::stod( param.second.data(), nullptr ) );
        params[key] = values;
        cell_ct++;
      }
    }
  }
  calib_ = PPSTimingCalibration( formula, params, time_info );
}

