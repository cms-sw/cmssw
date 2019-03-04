/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Edoardo Bossini
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
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

//------------------------------------------------------------------------------

class PPSTimingCalibrationESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
  public:
    PPSTimingCalibrationESSource( const edm::ParameterSet& );

    edm::ESProducts<std::unique_ptr<PPSTimingCalibration> > produce( const PPSTimingCalibrationRcd& );

    static void fillDescriptions( edm::ConfigurationDescriptions& );

  private:
    enum struct DetectorType
    {
      INVALID     = 0,
      TOTEM_UFSD  = 1,
      PPS_DIAMOND = 2
    };

    void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval& ) override;

    /// Extract calibration data from JSON file (TOTEM vertical)
    std::unique_ptr<PPSTimingCalibration> parseTotemUFSDJsonFile() const;
    /// Extract calibration data from JSON file (PPS horizontal diamond)
    std::unique_ptr<PPSTimingCalibration> parsePPSDiamondJsonFile() const;

    const std::string filename_;
    DetectorType subdetector_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationESSource::PPSTimingCalibrationESSource( const edm::ParameterSet& iConfig ) :
  filename_   ( iConfig.getParameter<edm::FileInPath>( "calibrationFile" ).fullPath() ),
  subdetector_( (DetectorType)iConfig.getParameter<unsigned int>( "subDetector" ) )
{
  setWhatProduced( this );
  findingRecord<PPSTimingCalibrationRcd>();
}

//------------------------------------------------------------------------------

edm::ESProducts<std::unique_ptr<PPSTimingCalibration> >
PPSTimingCalibrationESSource::produce( const PPSTimingCalibrationRcd& )
{
  switch ( subdetector_ ) {
    case DetectorType::TOTEM_UFSD:
      return edm::es::products( parseTotemUFSDJsonFile() );
    case DetectorType::PPS_DIAMOND:
      return edm::es::products( parsePPSDiamondJsonFile() );
    default:
      throw cms::Exception( "PPSTimingCalibrationESSource" )
        << "Subdetector " << (int)subdetector_ << " not recognised!";
  }
}

//------------------------------------------------------------------------------

void
PPSTimingCalibrationESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                                              const edm::IOVSyncValue&,
                                              edm::ValidityInterval& oValidity )
{
  oValidity = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime() );
}

//------------------------------------------------------------------------------

std::unique_ptr<PPSTimingCalibration>
PPSTimingCalibrationESSource::parseTotemUFSDJsonFile() const
{
  pt::ptree mother_node;
  pt::read_json( filename_, mother_node );

  const std::string formula = mother_node.get<std::string>( "formula" );
  PPSTimingCalibration::ParametersMap params;
  PPSTimingCalibration::TimingMap time_info;

  for ( pt::ptree::value_type& par : mother_node.get_child( "parameters" ) ) {
    PPSTimingCalibration::Key key;
    key.db = (int)strtol( par.first.data(), nullptr, 10 );

    for ( pt::ptree::value_type& board : par.second ) {
      key.sampic = board.second.get<int>( "sampic" );
      key.channel = board.second.get<int>( "channel" );
      double timeOffset = board.second.get<double>( "time_offset" );
      double timePrecision = board.second.get<double>( "time_precision" );
      key.cell = -1;
      time_info[key] = { timeOffset, timePrecision };

      int cell_ct = 0;
      for ( pt::ptree::value_type& cell : board.second.get_child( "cells" ) ) {
        std::vector<double> values;
        key.cell = cell_ct;

        for ( pt::ptree::value_type& param : cell.second )
          values.emplace_back( std::stod( param.second.data(), nullptr ) );
        params[key] = values;
        cell_ct++;
      }
    }
  }
  return std::make_unique<PPSTimingCalibration>( formula, params, time_info );
}

std::unique_ptr<PPSTimingCalibration>
PPSTimingCalibrationESSource::parsePPSDiamondJsonFile() const
{
  pt::ptree mother_node;
  pt::read_json( filename_, mother_node );

  const std::string formula = mother_node.get<std::string>( "formula" );
  PPSTimingCalibration::ParametersMap params;
  PPSTimingCalibration::TimingMap time_info;

  for ( pt::ptree::value_type& par : mother_node.get_child( "Parameters.Sectors" ) ) {
    PPSTimingCalibration::Key key;
    key.db = par.second.get<int>( "sector" );

    for ( pt::ptree::value_type& st : par.second.get_child( "Stations" ) ) {
      key.sampic = st.second.get<int>( "station" );

      for ( pt::ptree::value_type& pl : st.second.get_child( "Planes" ) ) {
        key.channel = pl.second.get<int>( "plane" );

        for ( pt::ptree::value_type& ch : pl.second.get_child( "Channels" ) ) {
          key.cell = ch.second.get<int>( "channel" );
          double timeOffset = ch.second.get<double>( "time_offset" );
          double timePrecision = ch.second.get<double>( "time_precision" );
          time_info[key] = { timeOffset, timePrecision };

          std::vector<double> values;
          for ( pt::ptree::value_type& param : ch.second.get_child( "param" ) )
            values.emplace_back( std::stod( param.second.data(), nullptr ) );
          params[key] = values;
        }
      }
    }
  }
  return std::make_unique<PPSTimingCalibration>( formula, params, time_info );
}

void
PPSTimingCalibrationESSource::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>( "calibrationFile", edm::FileInPath() )
    ->setComment( "file with SAMPIC calibrations, ADC and INL; if empty or corrupted, no calibration will be applied" );
  desc.add<unsigned int>( "subDetector", (unsigned int)PPSTimingCalibrationESSource::DetectorType::INVALID )
    ->setComment( "type of sub-detector for which the calibrations are provided" );

  descriptions.add( "ppsTimingCalibrationESSource", desc );
}

DEFINE_FWK_EVENTSETUP_SOURCE( PPSTimingCalibrationESSource );

