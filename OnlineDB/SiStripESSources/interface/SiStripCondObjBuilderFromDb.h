
#ifndef OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"

#include <memory>
#include <vector>
#include <string>
#include <typeinfo>
#include <cstdint>

class SiStripFecCabling;
class SiStripDetCabling;
class SiStripPedestals;
class SiStripNoises;
class SiStripQuality;
class SiStripThreshold;
class DcuDetIdMap;
class SiStripApvGain;
class SiStripLatency;
class TrackerTopology;

class SiStripCondObjBuilderFromDb {
public:
  // Typedefs
  typedef std::pair<uint32_t, FedChannelConnection> pair_apvpairconn;
  typedef std::vector<pair_apvpairconn> v_apvpairconn;
  typedef std::pair<uint32_t, v_apvpairconn> pair_detcon;
  typedef std::vector<pair_detcon> trackercon;

  typedef std::vector<pair_detcon>::iterator i_trackercon;
  typedef std::vector<pair_apvpairconn>::iterator i_apvpairconn;

  class SkipDeviceDescription {
    /* Class to hold the addresses of the devices to be skipped from gain update.
     * 0 stands for all devices at this level.
     * sistrip::invalid means this coordinate is not used. */
  public:
    SkipDeviceDescription();
    SkipDeviceDescription(const edm::ParameterSet& pset);
    bool isConsistent(const FedChannelConnection& fc) const;
    std::string dump() const;

  private:
    SiStripFecKey fec_;
    SiStripFedKey fed_;
    uint32_t detid_;
  };

  SiStripCondObjBuilderFromDb();
  SiStripCondObjBuilderFromDb(const edm::ParameterSet&, const edm::ActivityRegistry&);
  virtual ~SiStripCondObjBuilderFromDb();

  TrackerTopology* buildTrackerTopology();

  /** Returns database connection parameters. */
  inline const SiStripDbParams& dbParams() const { return db_->dbParams(); }

  /** Builds pedestals using FED descriptions and cabling info
      retrieved from configuration database. */
  void buildCondObj();
  void buildStripRelatedObjects(SiStripConfigDb* const db, const SiStripDetCabling& det_cabling);
  void buildAnalysisRelatedObjects(SiStripConfigDb* const db, const trackercon& tc);
  void buildFECRelatedObjects(SiStripConfigDb* const db, const trackercon& tc);
  void buildFEDRelatedObjects(SiStripConfigDb* const db, const trackercon& tc);

  bool checkForCompatibility(std::stringstream& input, std::stringstream& output, std::string& label);
  std::string getConfigString(const std::type_info& typeInfo);

  SiStripFedCabling* getFedCabling() {
    checkUpdate();
    return fed_cabling_;
  }
  SiStripPedestals* getPedestals() {
    checkUpdate();
    return pedestals_;
  }
  SiStripNoises* getNoises() {
    checkUpdate();
    return noises_;
  }
  SiStripThreshold* getThreshold() {
    checkUpdate();
    return threshold_;
  }
  SiStripQuality* getQuality() {
    checkUpdate();
    return quality_;
  }
  SiStripApvGain* getApvGain() {
    checkUpdate();
    return gain_;
  }
  SiStripLatency* getApvLatency() {
    checkUpdate();
    return latency_;
  }

  void getValue(SiStripFedCabling*& val) { val = getFedCabling(); }
  void getValue(SiStripPedestals*& val) { val = getPedestals(); }
  void getValue(SiStripNoises*& val) { val = getNoises(); }
  void getValue(SiStripThreshold*& val) { val = getThreshold(); }
  void getValue(SiStripQuality*& val) { val = getQuality(); }
  void getValue(SiStripBadStrip*& val) { val = new SiStripBadStrip(*(const SiStripBadStrip*)getQuality()); }
  void getValue(SiStripApvGain*& val) { val = getApvGain(); }
  void getValue(SiStripLatency*& val) { val = getApvLatency(); }

  void setLastIovGain(std::shared_ptr<SiStripApvGain> gain) { gain_last_ = gain; }

protected:
  void checkUpdate();

  /** Access to the configuration DB interface class. */
  // Build and retrieve SiStripConfigDb object using service
  edm::Service<SiStripConfigDb> db_;

  /** Container for DB connection parameters. */
  SiStripDbParams dbParams_;
  SiStripFedCabling* fed_cabling_;
  SiStripPedestals* pedestals_;
  SiStripNoises* noises_;
  SiStripThreshold* threshold_;
  SiStripQuality* quality_;
  SiStripApvGain* gain_;
  SiStripLatency* latency_;

  std::shared_ptr<SiStripApvGain> gain_last_;         // last gain object in DB
  std::vector<SkipDeviceDescription> skippedDevices;  // devices to be skipped for gain update
  std::vector<uint32_t> skippedDetIds;
  std::vector<SkipDeviceDescription>
      whitelistedDevices;  // devices whitelist for gain update: will NOT be skipped, even if in the 'skip list'
  std::vector<uint32_t> whitelistedDetIds;

  //methods used by BuildStripRelatedObjects
  bool setValuesApvLatency(SiStripLatency& latency_,
                           SiStripConfigDb* const db,
                           FedChannelConnection& ipair,
                           uint32_t detid,
                           uint16_t apvnr,
                           SiStripConfigDb::DeviceDescriptionsRange apvs);
  bool setValuesApvTiming(SiStripConfigDb* const db, FedChannelConnection& ipair);
  //  bool setValuesCabling(SiStripConfigDb* const db, FedChannelConnection &ipair, uint32_t detid);
  bool setValuesCabling(SiStripConfigDb::FedDescriptionsRange& descriptions,
                        FedChannelConnection& ipair,
                        uint32_t detid);
  bool retrieveFedDescriptions(SiStripConfigDb* const db);
  bool retrieveTimingAnalysisDescriptions(SiStripConfigDb* const db);
  vector<uint32_t> retrieveActiveDetIds(const SiStripDetCabling& det_cabling);
  vector<const FedChannelConnection*> buildConnections(const SiStripDetCabling& det_cabling, uint32_t det_id);
  uint16_t retrieveNumberAPVPairs(uint32_t det_id);

  //set and store data
  void setDefaultValuesCabling(uint16_t apvPair);
  void setDefaultValuesApvTiming(uint32_t detid, uint32_t apvPair);
  void setDefaultValuesApvLatency(SiStripLatency& latency_,
                                  const FedChannelConnection& ipair,
                                  uint32_t detid,
                                  uint16_t apvnr);
  void storePedestals(uint32_t det_id);
  void storeNoise(uint32_t det_id);
  void storeThreshold(uint32_t det_id);
  void storeQuality(uint32_t det_id);
  void storeTiming(uint32_t det_id);

  // cfi input parameters
  edm::VParameterSet m_skippedDevices;  // VPset of devices to be skipped in tickmark update
  edm::VParameterSet
      m_whitelistedDevices;  // VPset of whitelisted devices: will NOT be skipped in tickmark update (even if in the 'skip list')
  float m_tickmarkThreshold;  // threshold to accept the tickmark measurement
  float m_gaincalibrationfactor;
  float m_defaultpedestalvalue;
  float m_defaultnoisevalue;
  float m_defaultthresholdhighvalue;
  float m_defaultthresholdlowvalue;
  uint16_t m_defaultapvmodevalue;
  uint16_t m_defaultapvlatencyvalue;
  float m_defaulttickheightvalue;
  bool m_useanalysis;
  bool m_usefed;
  bool m_usefec;
  bool m_debug;
  SiStripDetInfo m_detInfo;

  //Data containers
  TrackerTopology* tTopo;
  SiStripPedestals::InputVector inputPedestals;
  SiStripNoises::InputVector inputNoises;
  SiStripThreshold::InputVector inputThreshold;
  SiStripQuality::InputVector inputQuality;
  SiStripApvGain::InputVector inputApvGain;

  // Tracker Cabling objects
  pair_apvpairconn p_apvpcon;
  v_apvpairconn v_apvpcon;
  pair_detcon p_detcon;
  trackercon v_trackercon;
};
#endif  // OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H
