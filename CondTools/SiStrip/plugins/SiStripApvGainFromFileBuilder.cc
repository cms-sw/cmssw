// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripApvGainFromFileBuilder
//
/**\class SiStripApvGainFromFileBuilder SiStripApvGainFromFileBuilder.cc
   Description: Created SiStripApvGain paylaods from tickmark height input ASCII files coming from
                SiStrip opto-gain scans.
*/
//
//  Original Author: A. Di Mattia
//  Contributors:    M. Musich    (modernization)
//
//  Created:  Wed, 1 Mar 2022 14:26:18 GMT
//

// STL includes
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

class SiStripApvGainFromFileBuilder : public edm::one::EDAnalyzer<> {
public:
  //enum ExceptionType = { NotConnected, ZeroGainFromScan, NegativeGainFromScan };

  typedef std::map<uint32_t, float> Gain;

  typedef struct {
    uint32_t det_id;
    uint16_t offlineAPV_id;
    int onlineAPV_id;
    int FED_id;
    int FED_ch;
    int i2cAdd;
    bool is_connected;
    bool is_scanned;
    float gain_from_scan;
    float gain_in_db;
  } Summary;

  /** Brief Constructor.
   */
  explicit SiStripApvGainFromFileBuilder(const edm::ParameterSet& iConfig);

  /** Brief Destructor performing the memory cleanup.
   */
  ~SiStripApvGainFromFileBuilder() override;

  /** Brief One dummy-event analysis to create the database record.
   */
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /** Brief framework fillDescription
   */
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> cablingToken_; /*!< ES token for the cabling */
  edm::FileInPath tfp_;        /*!< File Path for the tickmark scan with the APV gains. */
  double gainThreshold_;       /*!< Threshold for accepting the APV gain in the tickmark scan file. */
  double dummyAPVGain_;        /*!< Dummy value for the APV gain. */
  bool doGainNormalization_;   /*!< Normalize the tickmark for the APV gain. */
  bool putDummyIntoUncabled_;  /*!< Flag for putting the dummy gain in the channels not actuall cabled. */
  bool putDummyIntoUnscanned_; /*!< Flag for putting the dummy gain in the chennals not scanned. */
  bool putDummyIntoOffChannels_; /*!< Flag for putting the dummy gain in the channels that were off during the tickmark scan. */
  bool putDummyIntoBadChannels_; /*!< Flag for putting the dummy gain in the channels with negative gains. */
  bool outputMaps_;              /*!< Flag for dumping the internal maps on ASCII files. */
  bool outputSummary_;           /*!< Flag for dumping the summary of the exceptions during the DB filling. */

  const SiStripDetCabling* detCabling_; /*!< Description of detector cabling. */

  /** Brief Maps [det_id <--> gains] arranged per APV indexes.
   */
  std::vector<Gain*> gains_;          /*!< Mapping channels with positive heights. */
  std::vector<Gain*> negative_gains_; /*!< Mapping channels sending bad data. */
  std::vector<Gain*> null_gains_;     /*!< Mapping channels switched off during the scan. */

  /** Brief Collection of the channels entered in the DB without exceptions.
   * The channels whose APV gain has been input in the DB straight from the 
   * tickmark scan are collected in the summary vector. The summary list is
   * dumped in the SiStripApvGainSummary.txt at the end of the job. 
   */
  std::vector<Summary> summary_; /*!< Collection of channel with no DB filling exceptions. */

  /** Brief Collection of the exceptions encountered when filling the DB. 
   * An exception occur for all the non-cabled channels ( no gain associated
   * in the tikmark file) and for all the channels that were off ( with zero
   * gain associated) or sending corrupted data (with negative values in the
   * tickmark file). At the end of the job the exception summary is dumped in
   * SiStripApvGainExceptionSummary.txt.
   */
  std::vector<Summary> ex_summary_; /*!< Collection of DB filling exceptions. */

  /** Brief Read the ASCII file containing the tickmark gains.
   * This method reads the ASCII files that contains the tickmark heights for 
   * every APV. The heights are first translated into gains, dividing by 640,
   * then are stored into maps to be associated to the detector ids. Maps are
   * created for every APV index. 
   * Negative and Zero heights, yielding to a non physical gain, are stored 
   * into separate maps.
   *   Negative gain: channels sending bad data at the tickmark scan. 
   *   Zero gain    : channels switched off during the tickmark scan. 
   */
  void read_tickmark(void);

  /** Brief Returns the mapping among channels and gain heights for the APVs.
   * This method searchs the mapping of detector Ids <-> gains provided. If the
   * mapping exists for the requested APV it is returned; if not a new empty
   * mapping is created, inserted and retruned. The methods accepts onlineIDs
   * running from 0 to 5. 
   */
  Gain* get_map(std::vector<Gain*>* maps, int onlineAPV_id);

  /** Brief Dumps the internal mapping on a ASCII files.
   * This method dumps the detector id <-> gain maps into acii files separated
   * for each APV. The basenmae of for the acii file has to be provided as a
   * input parameter.
   */
  void output_maps(std::vector<Gain*>* maps, const char* basename) const;

  /** Brief Dump the exceptions summary on a ASCII file.
   * This method dumps the online coordinate of the channels for which  there
   * was an exception for filling the database record. Exceptions are the non
   * cabled modules, the channels that were off during the tickmark scan, the
   * channels sending corrupted data duirng the tickmark scan. These exceptions
   * have been solved putting a dummy gain into the DB record or putting a zero
   * gain.
   */
  void output_summary() const;

  /** Brief Format the output line for the channel summary.
   */
  void format_summary(std::stringstream& line, Summary summary) const;

  /** Brief Find the gain value for a pair det_id, APV_id in the internal maps.
   */
  bool gain_from_maps(uint32_t det_id, int onlineAPV_id, float& gain);
  void gain_from_maps(uint32_t det_id, uint16_t totalAPVs, std::vector<std::pair<int, float>>& gain) const;

  /** Brief Convert online APV id into offline APV id.
   */
  int online2offline(uint16_t onlineAPV_id, uint16_t totalAPVs) const;

  static constexpr float k_GainNormalizationFactor = 640.f;
  static constexpr float k_InvalidGain = 999999.f;
};

static const struct clean_up {
  void operator()(SiStripApvGainFromFileBuilder::Gain* el) {
    if (el != nullptr) {
      el->clear();
      delete el;
      el = nullptr;
    }
  }
} CleanUp;

SiStripApvGainFromFileBuilder::~SiStripApvGainFromFileBuilder() {
  for_each(gains_.begin(), gains_.end(), CleanUp);
  for_each(negative_gains_.begin(), negative_gains_.end(), CleanUp);
  for_each(null_gains_.begin(), null_gains_.end(), CleanUp);
}

SiStripApvGainFromFileBuilder::SiStripApvGainFromFileBuilder(const edm::ParameterSet& iConfig)
    : cablingToken_(esConsumes()),
      tfp_(iConfig.getParameter<edm::FileInPath>("tickFile")),
      gainThreshold_(iConfig.getParameter<double>("gainThreshold")),
      dummyAPVGain_(iConfig.getParameter<double>("dummyAPVGain")),
      doGainNormalization_(iConfig.getParameter<bool>("doGainNormalization")),
      putDummyIntoUncabled_(iConfig.getParameter<bool>("putDummyIntoUncabled")),
      putDummyIntoUnscanned_(iConfig.getParameter<bool>("putDummyIntoUnscanned")),
      putDummyIntoOffChannels_(iConfig.getParameter<bool>("putDummyIntoOffChannels")),
      putDummyIntoBadChannels_(iConfig.getParameter<bool>("putDummyIntoBadChannels")),
      outputMaps_(iConfig.getParameter<bool>("outputMaps")),
      outputSummary_(iConfig.getParameter<bool>("outputSummary")) {}

void SiStripApvGainFromFileBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  //unsigned int run=evt.id().run();

  edm::LogInfo("SiStripApvGainFromFileBuilder") << "@SUB=analyze"
                                                << "Insert SiStripApvGain Data.";
  this->read_tickmark();

  if (outputMaps_) {
    try {
      this->output_maps(&gains_, "tickmark_heights");
      this->output_maps(&negative_gains_, "negative_tickmark");
      this->output_maps(&null_gains_, "zero_tickmark");
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }

  // Retrieve the SiStripDetCabling description
  detCabling_ = &iSetup.getData(cablingToken_);

  // APV gain record to be filled with gains and delivered into the database.
  auto obj = std::make_unique<SiStripApvGain>();

  const auto& reader =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
  const auto& DetInfos = reader.getAllData();

  LogTrace("SiStripApvGainFromFileBuilder") << "  det id  |APVOF| CON |APVON| FED |FEDCH|i2cAd|tickGain|" << std::endl;

  for (const auto& it : DetInfos) {
    // check if det id is correct and if it is actually cabled in the detector
    if (it.first == 0 || it.first == 0xFFFFFFFF) {
      edm::LogError("DetIdNotGood") << "@SUB=analyze"
                                    << "Wrong det id: " << it.first << "  ... neglecting!";
      continue;
    }

    // For the cabled det_id retrieve the number of APV connected
    // to the module with the FED cabling
    uint16_t nAPVs = 0;
    const std::vector<const FedChannelConnection*> connection = detCabling_->getConnections(it.first);
    for (unsigned int ca = 0; ca < connection.size(); ca++) {
      if (connection[ca] != nullptr) {
        nAPVs += (connection[ca])->nApvs();
        break;
      }
    }

    // check consistency among FED cabling and ideal cabling, exit on error
    if (!connection.empty() && nAPVs != (uint16_t)it.second.nApvs) {
      edm::LogError("SiStripCablingError")
          << "@SUB=analyze"
          << "det id " << it.first << ": APV number from FedCabling (" << nAPVs
          << ") is different from the APV number retrieved from the ideal cabling (" << it.second.nApvs << ").";
      throw("Inconsistency on the number of APVs.");
    }

    // eventually separate the processing for the module that are fully
    // uncabled. This is worth only if we decide not tu put the record
    // in the DB for the uncabled det_id.
    //if( !detCabling_->IsConnected(it.first) ) {
    //
    //  continue;
    //}

    //Gather the APV online id
    std::vector<std::pair<int, float>> tickmark_for_detId(it.second.nApvs, std::pair<int, float>(-1, k_InvalidGain));
    for (unsigned int ca = 0; ca < connection.size(); ca++) {
      if (connection[ca] != nullptr) {
        uint16_t id1 = (connection[ca])->i2cAddr(0) % 32;
        uint16_t id2 = (connection[ca])->i2cAddr(1) % 32;
        tickmark_for_detId[online2offline(id1, it.second.nApvs)].first = id1;
        tickmark_for_detId[online2offline(id2, it.second.nApvs)].first = id2;
      }
    }
    gain_from_maps(it.first, it.second.nApvs, tickmark_for_detId);
    std::vector<float> theSiStripVector;

    // Fill the gain in the DB object, apply the logic for the dummy values
    for (unsigned short j = 0; j < it.second.nApvs; j++) {
      Summary summary;
      summary.det_id = it.first;
      summary.offlineAPV_id = j;
      summary.onlineAPV_id = tickmark_for_detId.at(j).first;
      summary.is_connected = false;
      summary.FED_id = -1;
      summary.FED_ch = -1;
      summary.i2cAdd = -1;

      for (unsigned int ca = 0; ca < connection.size(); ca++) {
        if (connection[ca] != nullptr && (connection[ca])->i2cAddr(j % 2) % 32 == summary.onlineAPV_id) {
          summary.is_connected = (connection[ca])->isConnected();
          summary.FED_id = (connection[ca])->fedId();
          summary.FED_ch = (connection[ca])->fedCh();
          summary.i2cAdd = (connection[ca])->i2cAddr(j % 2);
        }
      }

      try {
        float gain = tickmark_for_detId[j].second;
        summary.gain_from_scan = gain;
        LogTrace("SiStripApvGainFromFileBuilder")
            << it.first << "  " << std::setw(3) << j << "   " << std::setw(3) << connection.size() << "   "
            << std::setw(3) << summary.onlineAPV_id << "    " << std::setw(3) << summary.FED_id << "   " << std::setw(3)
            << summary.FED_ch << "   " << std::setw(3) << summary.i2cAdd << "   " << std::setw(7)
            << summary.gain_from_scan << std::endl;

        if (gain != k_InvalidGain) {
          summary.is_scanned = true;
          if (gain > gainThreshold_) {
            if (doGainNormalization_) {
              // divide the tickmark by the normalization factor (640)
              gain /= k_GainNormalizationFactor;
            }
            summary.gain_in_db = gain;
            if (!summary.is_connected)
              ex_summary_.push_back(summary);
            else
              summary_.push_back(summary);
          } else {
            if (gain == 0.f) {
              summary.gain_in_db = (putDummyIntoOffChannels_ ? dummyAPVGain_ : 0.f);
              ex_summary_.push_back(summary);
            } else if (gain < 0.f) {
              summary.gain_in_db = (putDummyIntoBadChannels_ ? dummyAPVGain_ : 0.f);
              ex_summary_.push_back(summary);
            }
          }
        } else {
          summary.is_scanned = false;
          if (!summary.is_connected) {
            summary.gain_in_db = (putDummyIntoUncabled_ ? dummyAPVGain_ : 0.f);
          } else {
            summary.gain_in_db = (putDummyIntoUnscanned_ ? dummyAPVGain_ : 0.f);
          }
          ex_summary_.push_back(summary);
        }

        theSiStripVector.push_back(summary.gain_in_db);
        LogTrace("SiStripApvGainFromFileBuilder") << "output gain:" << summary.gain_in_db;
      } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        edm::LogError("MappingError") << "@SUB=analyze"
                                      << "Job end prematurely.";
        return;
      }
    }

    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(it.first, range))
      edm::LogError("IndexError") << "@SUB=analyze"
                                  << "detid already exists.";
  }

  if (outputSummary_)
    output_summary();

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripApvGainRcd")) {
      mydbservice->createOneIOV<SiStripApvGain>(*obj, mydbservice->beginOfTime(), "SiStripApvGainRcd");
    } else {
      mydbservice->appendOneIOV<SiStripApvGain>(*obj, mydbservice->currentTime(), "SiStripApvGainRcd");
    }
  } else {
    edm::LogError("DBServiceNotAvailable") << "@SUB=analyze"
                                           << "DB Service is unavailable";
  }
}

void SiStripApvGainFromFileBuilder::read_tickmark() {
  // Connect file for input
  const auto& filename = tfp_.fullPath();
  std::ifstream thickmark_heights(filename.c_str());

  if (!thickmark_heights.is_open()) {
    edm::LogError("FileNotFound") << "@SUB=read_ticlmark"
                                  << "File with thickmark height file " << filename.c_str() << " cannot be opened!";
    return;
  }

  // clear internal maps
  for_each(gains_.begin(), gains_.end(), CleanUp);
  for_each(negative_gains_.begin(), negative_gains_.end(), CleanUp);
  for_each(null_gains_.begin(), null_gains_.end(), CleanUp);

  // read file and fill internal map
  uint32_t det_id = 0;
  uint32_t APV_id = 0;
  float tick_h = 0.;

  int count = -1;

  for (;;) {
    count++;
    thickmark_heights >> det_id >> APV_id >> tick_h;

    if (!(thickmark_heights.eof() || thickmark_heights.fail())) {
      if (count == 0) {
        LogTrace("Debug") << "Reading " << filename.c_str() << " for gathering the tickmark heights" << std::endl;
        LogTrace("Debug") << "|  Det Id   |  APV Id  |  Tickmark" << std::endl;
        LogTrace("Debug") << "+-----------+----------+----------" << std::endl;
      }
      LogTrace("Debug") << std::setw(11) << det_id << std::setw(8) << APV_id << std::setw(14) << tick_h << std::endl;

      // retrieve the map corresponding to the gain collection
      Gain* map = nullptr;
      if (tick_h > 0.f) {
        map = get_map(&gains_, APV_id);
      } else if (tick_h < 0.f) {
        map = get_map(&negative_gains_, APV_id);
      } else {
        // if tick_h == 0.f
        map = get_map(&null_gains_, APV_id);
      }

      // insert the gain value in the map
      if (map) {
        std::pair<Gain::iterator, bool> ret = map->insert(std::pair<uint32_t, float>(det_id, tick_h));

        if (ret.second == false) {
          edm::LogError("MapError") << "@SUB=read_tickmark"
                                    << "Cannot not insert gain for detector id " << det_id
                                    << " into the internal map: detector id already in the map.";
        }
      } else {
        edm::LogError("MapError") << "@SUB=read_tickmark"
                                  << "Cannot get the online-offline APV mapping!";
      }
    } else if (thickmark_heights.eof()) {
      edm::LogInfo("SiStripApvGainFromFileBuilder") << "@SUB=read_tickmark"
                                                    << "EOF of " << filename.c_str() << " reached.";
      break;
    } else if (thickmark_heights.fail()) {
      edm::LogError("FileiReadError") << "@SUB=read_tickmark"
                                      << "error while reading " << filename.c_str();
      break;
    }
  }

  thickmark_heights.close();
}

SiStripApvGainFromFileBuilder::Gain* SiStripApvGainFromFileBuilder::get_map(std::vector<Gain*>* maps,
                                                                            int onlineAPV_id) {
  Gain* map = nullptr;
  if (onlineAPV_id < 0 || onlineAPV_id > 5)
    return map;

  try {
    map = maps->at(onlineAPV_id);
  } catch (const std::out_of_range&) {
    if (maps->size() < static_cast<unsigned int>(onlineAPV_id))
      maps->resize(onlineAPV_id);
    maps->insert(maps->begin() + onlineAPV_id, new Gain());
    map = (*maps)[onlineAPV_id];
  }

  if (map == nullptr) {
    (*maps)[onlineAPV_id] = new Gain();
    map = (*maps)[onlineAPV_id];
  }

  return map;
}

void SiStripApvGainFromFileBuilder::output_maps(std::vector<Gain*>* maps, const char* basename) const {
  for (unsigned int APV = 0; APV < maps->size(); APV++) {
    Gain* map = (*maps)[APV];
    if (map != nullptr) {
      // open output file
      std::stringstream name;
      name << basename << "_APV" << APV << ".txt";
      std::ofstream* ofile = new std::ofstream(name.str(), std::ofstream::trunc);
      if (!ofile->is_open())
        throw "cannot open output file!";
      for (Gain::const_iterator el = map->begin(); el != map->end(); el++) {
        (*ofile) << (*el).first << "    " << (*el).second << std::endl;
      }
      ofile->close();
      delete ofile;
    }
  }
}

void SiStripApvGainFromFileBuilder::output_summary() const {
  std::ofstream* ofile = new std::ofstream("SiStripApvGainSummary.txt", std::ofstream::trunc);
  (*ofile) << "  det id  | APV | isConnected | FED |FEDCH|i2cAd|APVON| is_scanned |tickGain|gainInDB|" << std::endl;
  (*ofile) << "----------+-----+-------------+-----+-----+-----+-----+------------+--------+--------+" << std::endl;
  for (unsigned int s = 0; s < summary_.size(); s++) {
    Summary summary = summary_[s];

    std::stringstream line;

    format_summary(line, summary);

    (*ofile) << line.str() << std::endl;
  }
  ofile->close();
  delete ofile;

  ofile = new std::ofstream("SiStripApvGainExceptionSummary.txt", std::ofstream::trunc);
  (*ofile) << "  det id  | APV | isConnected | FED |FEDCH|i2cAd|APVON| is_scanned |tickGain|gainInDB|" << std::endl;
  (*ofile) << "----------+-----+-------------+-----+-----+-----+-----+------------+--------+--------+" << std::endl;
  for (unsigned int s = 0; s < ex_summary_.size(); s++) {
    Summary summary = ex_summary_[s];

    std::stringstream line;

    format_summary(line, summary);

    (*ofile) << line.str() << std::endl;
  }
  ofile->close();
  delete ofile;
}

void SiStripApvGainFromFileBuilder::format_summary(std::stringstream& line, Summary summary) const {
  std::string conn = (summary.is_connected) ? "CONNECTED" : "NOT_CONNECTED";
  std::string scan = (summary.is_scanned) ? "IN_SCAN" : "NOT_IN_SCAN";

  line << summary.det_id << "  " << std::setw(3) << summary.offlineAPV_id << "  " << std::setw(13) << conn << "   "
       << std::setw(3) << summary.FED_id << "   " << std::setw(3) << summary.FED_ch << "   " << std::setw(3)
       << summary.i2cAdd << "  " << std::setw(3) << summary.onlineAPV_id << "   " << std::setw(11) << scan << "  "
       << std::setw(7) << summary.gain_from_scan << "  " << std::setw(7) << summary.gain_in_db;
}

bool SiStripApvGainFromFileBuilder::gain_from_maps(uint32_t det_id, int onlineAPV_id, float& gain) {
  Gain* map = nullptr;

  // search det_id and APV in the good scan map
  map = get_map(&gains_, onlineAPV_id);
  if (map != nullptr) {
    Gain::const_iterator el = map->find(det_id);
    if (el != map->end()) {
      gain = el->second;
      return true;
    }
  }

  // search det_id and APV in the zero gain scan map
  map = get_map(&negative_gains_, onlineAPV_id);
  if (map != nullptr) {
    Gain::const_iterator el = map->find(det_id);
    if (el != map->end()) {
      gain = el->second;
      return true;
    }
  }

  //search det_id and APV in the negative gain scan map
  map = get_map(&null_gains_, onlineAPV_id);
  if (map != nullptr) {
    Gain::const_iterator el = map->find(det_id);
    if (el != map->end()) {
      gain = el->second;
      return true;
    }
  }

  return false;
}

void SiStripApvGainFromFileBuilder::gain_from_maps(uint32_t det_id,
                                                   uint16_t totalAPVs,
                                                   std::vector<std::pair<int, float>>& gain) const {
  std::stringstream ex_msg;
  ex_msg << "two APVs with the same online id for det id " << det_id
         << ". Please check the tick mark file or the read_tickmark routine." << std::endl;

  for (unsigned int i = 0; i < 6; i++) {
    int offlineAPV_id = online2offline(i, totalAPVs);
    try {
      Gain* map = gains_.at(i);
      if (map != nullptr) {
        Gain::const_iterator el = map->find(det_id);
        if (el != map->end()) {
          if (gain[offlineAPV_id].second != k_InvalidGain)
            throw(ex_msg.str());
          gain[offlineAPV_id].second = el->second;
        }
      }
    } catch (const std::out_of_range&) {
      // nothing to do, just pass over
    }

    try {
      Gain* map = negative_gains_.at(i);
      if (map != nullptr) {
        Gain::const_iterator el = map->find(det_id);
        if (el != map->end()) {
          if (gain[offlineAPV_id].second != k_InvalidGain)
            throw(ex_msg.str());
          gain[offlineAPV_id].second = el->second;
        }
      }
    } catch (const std::out_of_range&) {
      // nothing to do, just pass over
    }

    try {
      Gain* map = null_gains_.at(i);
      if (map != nullptr) {
        Gain::const_iterator el = map->find(det_id);
        if (el != map->end()) {
          if (gain[offlineAPV_id].second != k_InvalidGain)
            throw(ex_msg.str());
          gain[offlineAPV_id].second = el->second;
        }
      }
    } catch (const std::out_of_range&) {
      // nothing to do, just pass over
    }
  }
}

int SiStripApvGainFromFileBuilder::online2offline(uint16_t onlineAPV_id, uint16_t totalAPVs) const {
  return (onlineAPV_id >= totalAPVs) ? onlineAPV_id - 2 : onlineAPV_id;
}

void SiStripApvGainFromFileBuilder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Conditions Builder for SiStripApvGain Objects (G1 gain) from tickmark height file");
  desc.add<edm::FileInPath>("tickFile", edm::FileInPath("CondTools/SiStrip/data/tickheight.txt"));
  desc.add<double>("gainThreshold", 0.)->setComment("threshold to retain the scan vale");
  desc.add<double>("dummyAPVGain", (690. / k_GainNormalizationFactor));  // from TDR
  desc.add<bool>("doGainNormalization", false)->setComment("normalize the output gain in DB by 640");
  desc.add<bool>("putDummyIntoUncabled", false)->setComment("use default gain for uncabled APVs");
  desc.add<bool>("putDummyIntoUnscanned", false)->setComment("use default gain for unscanned APVs");
  desc.add<bool>("putDummyIntoOffChannels", false)->setComment("use default gain for APVs reading HV off modules");
  desc.add<bool>("putDummyIntoBadChannels", false)->setComment("use default gain for bad APV channels");
  desc.add<bool>("outputMaps", false)->setComment("prints ouput maps");
  desc.add<bool>("outputSummary", false)->setComment("prints output summary");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripApvGainFromFileBuilder);
