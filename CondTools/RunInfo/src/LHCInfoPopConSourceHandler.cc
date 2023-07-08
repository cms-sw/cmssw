#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondTools/RunInfo/interface/LHCInfoPopConSourceHandler.h"
#include "CondTools/RunInfo/interface/OMSAccess.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/TimeStamp.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include <cmath>

namespace cond {
  static const std::pair<const char*, LHCInfo::FillType> s_fillTypeMap[] = {std::make_pair("PROTONS", LHCInfo::PROTONS),
                                                                            std::make_pair("IONS", LHCInfo::IONS),
                                                                            std::make_pair("COSMICS", LHCInfo::COSMICS),
                                                                            std::make_pair("GAP", LHCInfo::GAP)};

  static const std::pair<const char*, LHCInfo::ParticleType> s_particleTypeMap[] = {
      std::make_pair("PROTON", LHCInfo::PROTON),
      std::make_pair("PB82", LHCInfo::PB82),
      std::make_pair("AR18", LHCInfo::AR18),
      std::make_pair("D", LHCInfo::D),
      std::make_pair("XE54", LHCInfo::XE54)};

  LHCInfo::FillType fillTypeFromString(const std::string& s_fill_type) {
    for (auto const& i : s_fillTypeMap)
      if (s_fill_type == i.first)
        return i.second;
    return LHCInfo::UNKNOWN;
  }

  LHCInfo::ParticleType particleTypeFromString(const std::string& s_particle_type) {
    for (auto const& i : s_particleTypeMap)
      if (s_particle_type == i.first)
        return i.second;
    return LHCInfo::NONE;
  }

  namespace impl {

    template <>
    LHCInfo::FillType from_string(const std::string& attributeValue) {
      return from_string_impl<LHCInfo::FillType, &fillTypeFromString>(attributeValue, LHCInfo::UNKNOWN);
    }

    template <>
    LHCInfo::ParticleType from_string(const std::string& attributeValue) {
      return from_string_impl<LHCInfo::ParticleType, &particleTypeFromString>(attributeValue, LHCInfo::NONE);
    }

  }  // namespace impl
}  // namespace cond

LHCInfoPopConSourceHandler::LHCInfoPopConSourceHandler(edm::ParameterSet const& pset)
    : m_debug(pset.getUntrackedParameter<bool>("debug", false)),
      m_startTime(),
      m_endTime(),
      m_samplingInterval((unsigned int)pset.getUntrackedParameter<unsigned int>("samplingInterval", 300)),
      m_endFill(pset.getUntrackedParameter<bool>("endFill", true)),
      m_name(pset.getUntrackedParameter<std::string>("name", "LHCInfoPopConSourceHandler")),
      m_connectionString(pset.getUntrackedParameter<std::string>("connectionString", "")),
      m_ecalConnectionString(pset.getUntrackedParameter<std::string>("ecalConnectionString", "")),
      m_dipSchema(pset.getUntrackedParameter<std::string>("DIPSchema", "")),
      m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath", "")),
      m_omsBaseUrl(pset.getUntrackedParameter<std::string>("omsBaseUrl", "")),
      m_fillPayload(),
      m_prevPayload(),
      m_tmpBuffer() {
  if (pset.exists("startTime")) {
    m_startTime = boost::posix_time::time_from_string(pset.getUntrackedParameter<std::string>("startTime"));
  }
  boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
  m_endTime = now;
  if (pset.exists("endTime")) {
    m_endTime = boost::posix_time::time_from_string(pset.getUntrackedParameter<std::string>("endTime"));
    if (m_endTime > now)
      m_endTime = now;
  }
}
//L1: try with different m_dipSchema
//L2: try with different m_name
LHCInfoPopConSourceHandler::~LHCInfoPopConSourceHandler() {}

namespace LHCInfoImpl {

  struct IOVComp {
    bool operator()(const cond::Time_t& x, const std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>& y) {
      return (x < y.first);
    }
  };

  // function to search in the vector the target time
  std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>::const_iterator search(
      const cond::Time_t& val, const std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>& container) {
    if (container.empty())
      return container.end();
    auto p = std::upper_bound(container.begin(), container.end(), val, IOVComp());
    return (p != container.begin()) ? p - 1 : container.end();
  }

  bool makeFillPayload(std::unique_ptr<LHCInfo>& targetPayload, const cond::OMSServiceResult& queryResult) {
    bool ret = false;
    if (!queryResult.empty()) {
      auto row = *queryResult.begin();
      auto currentFill = row.get<unsigned short>("fill_number");
      auto bunches1 = row.get<unsigned short>("bunches_beam1");
      auto bunches2 = row.get<unsigned short>("bunches_beam2");
      auto collidingBunches = row.get<unsigned short>("bunches_colliding");
      auto targetBunches = row.get<unsigned short>("bunches_target");
      auto fillType = row.get<LHCInfo::FillType>("fill_type_runtime");
      auto particleType1 = row.get<LHCInfo::ParticleType>("fill_type_party1");
      auto particleType2 = row.get<LHCInfo::ParticleType>("fill_type_party2");
      auto intensityBeam1 = row.get<float>("intensity_beam1");
      auto intensityBeam2 = row.get<float>("intensity_beam2");
      auto energy = row.get<float>("energy");
      auto creationTime = row.get<boost::posix_time::ptime>("start_time");
      auto stableBeamStartTime = row.get<boost::posix_time::ptime>("start_stable_beam");
      auto beamDumpTime = row.get<boost::posix_time::ptime>("end_time");
      auto injectionScheme = row.get<std::string>("injection_scheme");
      targetPayload = std::make_unique<LHCInfo>();
      targetPayload->setFillNumber(currentFill);
      targetPayload->setBunchesInBeam1(bunches1);
      targetPayload->setBunchesInBeam2(bunches2);
      targetPayload->setCollidingBunches(collidingBunches);
      targetPayload->setTargetBunches(targetBunches);
      targetPayload->setFillType(fillType);
      targetPayload->setParticleTypeForBeam1(particleType1);
      targetPayload->setParticleTypeForBeam2(particleType2);
      targetPayload->setIntensityForBeam1(intensityBeam1);
      targetPayload->setIntensityForBeam2(intensityBeam2);
      targetPayload->setEnergy(energy);
      targetPayload->setCreationTime(cond::time::from_boost(creationTime));
      targetPayload->setBeginTime(cond::time::from_boost(stableBeamStartTime));
      targetPayload->setEndTime(cond::time::from_boost(beamDumpTime));
      targetPayload->setInjectionScheme(injectionScheme);
      ret = true;
    }
    return ret;
  }

}  // namespace LHCInfoImpl

size_t LHCInfoPopConSourceHandler::getLumiData(const cond::OMSService& oms,
                                               unsigned short fillId,
                                               const boost::posix_time::ptime& beginFillTime,
                                               const boost::posix_time::ptime& endFillTime) {
  auto query = oms.query("lumisections");
  query->addOutputVars({"start_time", "delivered_lumi", "recorded_lumi"});
  query->filterEQ("fill_number", fillId);
  query->filterGT("start_time", beginFillTime).filterLT("start_time", endFillTime);
  query->limit(kLumisectionsQueryLimit);
  size_t nlumi = 0;
  if (query->execute()) {
    auto res = query->result();
    for (auto r : res) {
      nlumi++;
      auto lumiTime = r.get<boost::posix_time::ptime>("start_time");
      auto delivLumi = r.get<float>("delivered_lumi");
      auto recLumi = r.get<float>("recorded_lumi");
      LHCInfo* thisLumiSectionInfo = m_fillPayload->cloneFill();
      m_tmpBuffer.emplace_back(std::make_pair(cond::time::from_boost(lumiTime), thisLumiSectionInfo));
      LHCInfo& payload = *thisLumiSectionInfo;
      payload.setDelivLumi(delivLumi);
      payload.setRecLumi(recLumi);
    }
  }
  return nlumi;
}

namespace LHCInfoImpl {
  struct LumiSectionFilter {
    LumiSectionFilter(const std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>& samples)
        : currLow(samples.begin()), currUp(samples.begin()), end(samples.end()) {
      currUp++;
    }

    void reset(const std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>& samples) {
      currLow = samples.begin();
      currUp = samples.begin();
      currUp++;
      end = samples.end();
      currentDipTime = 0;
    }

    bool process(cond::Time_t dipTime) {
      if (currLow == end)
        return false;
      bool search = false;
      if (currentDipTime == 0) {
        search = true;
      } else {
        if (dipTime == currentDipTime)
          return true;
        else {
          cond::Time_t upper = cond::time::MAX_VAL;
          if (currUp != end)
            upper = currUp->first;
          if (dipTime < upper)
            return false;
          else {
            search = true;
          }
        }
      }
      if (search) {
        while (currUp != end and currUp->first < dipTime) {
          currLow++;
          currUp++;
        }
        currentDipTime = dipTime;
        return currLow != end;
      }
      return false;
    }

    cond::Time_t currentSince() { return currLow->first; }
    LHCInfo& currentPayload() { return *currLow->second; }

    std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>::const_iterator current() { return currLow; }
    std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>::const_iterator currLow;
    std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>::const_iterator currUp;
    std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>::const_iterator end;
    cond::Time_t currentDipTime = 0;
  };
}  // namespace LHCInfoImpl

void LHCInfoPopConSourceHandler::getDipData(const cond::OMSService& oms,
                                            const boost::posix_time::ptime& beginFillTime,
                                            const boost::posix_time::ptime& endFillTime) {
  // unsure how to handle this.
  // the old implementation is not helping: apparently it is checking only the bunchconfiguration for the first diptime set of values...
  auto query1 = oms.query("diplogger/dip/acc/LHC/RunControl/CirculatingBunchConfig/Beam1");
  query1->filterGT("dip_time", beginFillTime).filterLT("dip_time", endFillTime);
  //This query is limited to 100 rows, but currently only one is used
  //If all this data is needed and saved properly the limit has to be set: query1->limit(...)
  if (query1->execute()) {
    auto res = query1->result();
    if (!res.empty()) {
      std::bitset<LHCInfo::bunchSlots + 1> bunchConfiguration1(0ULL);
      auto row = *res.begin();
      auto vbunchConf1 = row.getArray<unsigned short>("value");
      for (auto vb : vbunchConf1) {
        if (vb != 0) {
          unsigned short slot = (vb - 1) / 10 + 1;
          bunchConfiguration1[slot] = true;
        }
      }
      m_fillPayload->setBunchBitsetForBeam1(bunchConfiguration1);
    }
  }
  auto query2 = oms.query("diplogger/dip/acc/LHC/RunControl/CirculatingBunchConfig/Beam2");
  query2->filterGT("dip_time", beginFillTime).filterLT("dip_time", endFillTime);
  //This query is limited to 100 rows, but currently only one is used
  if (query2->execute()) {
    auto res = query2->result();
    if (!res.empty()) {
      std::bitset<LHCInfo::bunchSlots + 1> bunchConfiguration2(0ULL);
      auto row = *res.begin();
      auto vbunchConf2 = row.getArray<unsigned short>("value");
      for (auto vb : vbunchConf2) {
        if (vb != 0) {
          unsigned short slot = (vb - 1) / 10 + 1;
          bunchConfiguration2[slot] = true;
        }
      }
      m_fillPayload->setBunchBitsetForBeam2(bunchConfiguration2);
    }
  }

  auto query3 = oms.query("diplogger/dip/CMS/LHC/LumiPerBunch");
  query3->filterGT("dip_time", beginFillTime).filterLT("dip_time", endFillTime);
  //This query is limited to 100 rows, but currently only one is used
  if (query3->execute()) {
    auto res = query3->result();
    if (!res.empty()) {
      std::vector<float> lumiPerBX;
      auto row = *res.begin();
      auto lumiBunchInst = row.getArray<float>("lumi_bunch_inst");
      for (auto lb : lumiBunchInst) {
        if (lb != 0.) {
          lumiPerBX.push_back(lb);
        }
      }
      m_fillPayload->setLumiPerBX(lumiPerBX);
    }
  }
}

bool LHCInfoPopConSourceHandler::getCTTPSData(cond::persistency::Session& session,
                                              const boost::posix_time::ptime& beginFillTime,
                                              const boost::posix_time::ptime& endFillTime) {
  //run the fifth query against the CTPPS schema
  //Initializing the CMS_CTP_CTPPS_COND schema.
  coral::ISchema& CTPPS = session.coralSession().schema("CMS_PPS_SPECT_COND");
  //execute query for CTPPS Data
  std::unique_ptr<coral::IQuery> CTPPSDataQuery(CTPPS.newQuery());
  //FROM clause
  CTPPSDataQuery->addToTableList(std::string("PPS_LHC_MACHINE_PARAMS"));
  //SELECT clause
  CTPPSDataQuery->addToOutputList(std::string("DIP_UPDATE_TIME"));
  CTPPSDataQuery->addToOutputList(std::string("LHC_STATE"));
  CTPPSDataQuery->addToOutputList(std::string("LHC_COMMENT"));
  CTPPSDataQuery->addToOutputList(std::string("LUMI_SECTION"));
  CTPPSDataQuery->addToOutputList(std::string("XING_ANGLE_P5_X_URAD"));
  CTPPSDataQuery->addToOutputList(std::string("BETA_STAR_P5_X_M"));
  //WHERE CLAUSE
  coral::AttributeList CTPPSDataBindVariables;
  CTPPSDataBindVariables.extend<coral::TimeStamp>(std::string("beginFillTime"));
  CTPPSDataBindVariables.extend<coral::TimeStamp>(std::string("endFillTime"));
  CTPPSDataBindVariables[std::string("beginFillTime")].data<coral::TimeStamp>() = coral::TimeStamp(beginFillTime);
  CTPPSDataBindVariables[std::string("endFillTime")].data<coral::TimeStamp>() = coral::TimeStamp(endFillTime);
  std::string conditionStr = std::string("DIP_UPDATE_TIME>= :beginFillTime and DIP_UPDATE_TIME< :endFillTime");
  CTPPSDataQuery->setCondition(conditionStr, CTPPSDataBindVariables);
  //ORDER BY clause
  CTPPSDataQuery->addToOrderList(std::string("DIP_UPDATE_TIME"));
  //define query output
  coral::AttributeList CTPPSDataOutput;
  CTPPSDataOutput.extend<coral::TimeStamp>(std::string("DIP_UPDATE_TIME"));
  CTPPSDataOutput.extend<std::string>(std::string("LHC_STATE"));
  CTPPSDataOutput.extend<std::string>(std::string("LHC_COMMENT"));
  CTPPSDataOutput.extend<int>(std::string("LUMI_SECTION"));
  CTPPSDataOutput.extend<float>(std::string("XING_ANGLE_P5_X_URAD"));
  CTPPSDataOutput.extend<float>(std::string("BETA_STAR_P5_X_M"));
  CTPPSDataQuery->defineOutput(CTPPSDataOutput);
  //execute the query
  coral::ICursor& CTPPSDataCursor = CTPPSDataQuery->execute();
  cond::Time_t dipTime = 0;
  std::string lhcState = "", lhcComment = "", ctppsStatus = "";
  unsigned int lumiSection = 0;
  float crossingAngle = 0., betastar = 0.;

  bool ret = false;
  LHCInfoImpl::LumiSectionFilter filter(m_tmpBuffer);
  while (CTPPSDataCursor.next()) {
    if (m_debug) {
      std::ostringstream CTPPS;
      CTPPSDataCursor.currentRow().toOutputStream(CTPPS);
    }
    coral::Attribute const& dipTimeAttribute = CTPPSDataCursor.currentRow()[std::string("DIP_UPDATE_TIME")];
    if (!dipTimeAttribute.isNull()) {
      dipTime = cond::time::from_boost(dipTimeAttribute.data<coral::TimeStamp>().time());
      if (filter.process(dipTime)) {
        ret = true;
        coral::Attribute const& lhcStateAttribute = CTPPSDataCursor.currentRow()[std::string("LHC_STATE")];
        if (!lhcStateAttribute.isNull()) {
          lhcState = lhcStateAttribute.data<std::string>();
        }
        coral::Attribute const& lhcCommentAttribute = CTPPSDataCursor.currentRow()[std::string("LHC_COMMENT")];
        if (!lhcCommentAttribute.isNull()) {
          lhcComment = lhcCommentAttribute.data<std::string>();
        }
        coral::Attribute const& lumiSectionAttribute = CTPPSDataCursor.currentRow()[std::string("LUMI_SECTION")];
        if (!lumiSectionAttribute.isNull()) {
          lumiSection = lumiSectionAttribute.data<int>();
        }
        coral::Attribute const& crossingAngleXAttribute =
            CTPPSDataCursor.currentRow()[std::string("XING_ANGLE_P5_X_URAD")];
        if (!crossingAngleXAttribute.isNull()) {
          crossingAngle = crossingAngleXAttribute.data<float>();
        }
        coral::Attribute const& betaStarXAttribute = CTPPSDataCursor.currentRow()[std::string("BETA_STAR_P5_X_M")];
        if (!betaStarXAttribute.isNull()) {
          betastar = betaStarXAttribute.data<float>();
        }
        for (auto it = filter.current(); it != m_tmpBuffer.end(); it++) {
          // set the current values to all of the payloads of the lumi section samples after the current since
          LHCInfo& payload = *(it->second);
          payload.setCrossingAngle(crossingAngle);
          payload.setBetaStar(betastar);
          payload.setLhcState(lhcState);
          payload.setLhcComment(lhcComment);
          payload.setCtppsStatus(ctppsStatus);
          payload.setLumiSection(lumiSection);
        }
      }
    }
  }
  return ret;
}

namespace LHCInfoImpl {
  static const std::map<std::string, int> vecMap = {
      {"Beam1/beamPhaseMean", 1}, {"Beam2/beamPhaseMean", 2}, {"Beam1/cavPhaseMean", 3}, {"Beam2/cavPhaseMean", 4}};
  void setElementData(cond::Time_t since,
                      const std::string& dipVal,
                      unsigned int elementNr,
                      float value,
                      LHCInfo& payload,
                      std::set<cond::Time_t>& initList) {
    if (initList.find(since) == initList.end()) {
      payload.beam1VC().resize(LHCInfo::bunchSlots, 0.);
      payload.beam2VC().resize(LHCInfo::bunchSlots, 0.);
      payload.beam1RF().resize(LHCInfo::bunchSlots, 0.);
      payload.beam2RF().resize(LHCInfo::bunchSlots, 0.);
      initList.insert(since);
    }
    // set the current values to all of the payloads of the lumi section samples after the current since
    if (elementNr < LHCInfo::bunchSlots) {
      switch (vecMap.at(dipVal)) {
        case 1:
          payload.beam1VC()[elementNr] = value;
          break;
        case 2:
          payload.beam2VC()[elementNr] = value;
          break;
        case 3:
          payload.beam1RF()[elementNr] = value;
          break;
        case 4:
          payload.beam2RF()[elementNr] = value;
          break;
        default:
          break;
      }
    }
  }
}  // namespace LHCInfoImpl

bool LHCInfoPopConSourceHandler::getEcalData(cond::persistency::Session& session,
                                             const boost::posix_time::ptime& lowerTime,
                                             const boost::posix_time::ptime& upperTime,
                                             bool update) {
  //run the sixth query against the CMS_DCS_ENV_PVSS_COND schema
  //Initializing the CMS_DCS_ENV_PVSS_COND schema.
  coral::ISchema& ECAL = session.nominalSchema();
  //start the transaction against the fill logging schema
  //execute query for ECAL Data
  std::unique_ptr<coral::IQuery> ECALDataQuery(ECAL.newQuery());
  //FROM clause
  ECALDataQuery->addToTableList(std::string("BEAM_PHASE"));
  //SELECT clause
  ECALDataQuery->addToOutputList(std::string("CHANGE_DATE"));
  ECALDataQuery->addToOutputList(std::string("DIP_value"));
  ECALDataQuery->addToOutputList(std::string("element_nr"));
  ECALDataQuery->addToOutputList(std::string("VALUE_NUMBER"));
  //WHERE CLAUSE
  coral::AttributeList ECALDataBindVariables;
  ECALDataBindVariables.extend<coral::TimeStamp>(std::string("lowerTime"));
  ECALDataBindVariables.extend<coral::TimeStamp>(std::string("upperTime"));
  ECALDataBindVariables[std::string("lowerTime")].data<coral::TimeStamp>() = coral::TimeStamp(lowerTime);
  ECALDataBindVariables[std::string("upperTime")].data<coral::TimeStamp>() = coral::TimeStamp(upperTime);
  std::string conditionStr = std::string(
      "(DIP_value LIKE '%beamPhaseMean%' OR DIP_value LIKE '%cavPhaseMean%') AND CHANGE_DATE >= :lowerTime AND "
      "CHANGE_DATE < :upperTime");

  ECALDataQuery->setCondition(conditionStr, ECALDataBindVariables);
  //ORDER BY clause
  ECALDataQuery->addToOrderList(std::string("CHANGE_DATE"));
  ECALDataQuery->addToOrderList(std::string("DIP_value"));
  ECALDataQuery->addToOrderList(std::string("element_nr"));
  //define query output
  coral::AttributeList ECALDataOutput;
  ECALDataOutput.extend<coral::TimeStamp>(std::string("CHANGE_DATE"));
  ECALDataOutput.extend<std::string>(std::string("DIP_value"));
  ECALDataOutput.extend<unsigned int>(std::string("element_nr"));
  ECALDataOutput.extend<float>(std::string("VALUE_NUMBER"));
  //ECALDataQuery->limitReturnedRows( 14256 ); //3564 entries per vector.
  ECALDataQuery->defineOutput(ECALDataOutput);
  //execute the query
  coral::ICursor& ECALDataCursor = ECALDataQuery->execute();
  cond::Time_t changeTime = 0;
  cond::Time_t firstTime = 0;
  std::string dipVal = "";
  unsigned int elementNr = 0;
  float value = 0.;
  std::set<cond::Time_t> initializedVectors;
  LHCInfoImpl::LumiSectionFilter filter(m_tmpBuffer);
  bool ret = false;
  if (m_prevPayload.get()) {
    for (auto& lumiSlot : m_tmpBuffer) {
      lumiSlot.second->setBeam1VC(m_prevPayload->beam1VC());
      lumiSlot.second->setBeam2VC(m_prevPayload->beam2VC());
      lumiSlot.second->setBeam1RF(m_prevPayload->beam1RF());
      lumiSlot.second->setBeam2RF(m_prevPayload->beam2RF());
    }
  }
  std::map<cond::Time_t, cond::Time_t> iovMap;
  cond::Time_t lowerLumi = m_tmpBuffer.front().first;
  while (ECALDataCursor.next()) {
    if (m_debug) {
      std::ostringstream ECAL;
      ECALDataCursor.currentRow().toOutputStream(ECAL);
    }
    coral::Attribute const& changeDateAttribute = ECALDataCursor.currentRow()[std::string("CHANGE_DATE")];
    if (!changeDateAttribute.isNull()) {
      ret = true;
      boost::posix_time::ptime chTime = changeDateAttribute.data<coral::TimeStamp>().time();
      // move the first IOV found to the start of the fill interval selected
      if (changeTime == 0) {
        firstTime = cond::time::from_boost(chTime);
      }
      changeTime = cond::time::from_boost(chTime);
      cond::Time_t iovTime = changeTime;
      if (!update and changeTime == firstTime)
        iovTime = lowerLumi;
      coral::Attribute const& dipValAttribute = ECALDataCursor.currentRow()[std::string("DIP_value")];
      coral::Attribute const& valueNumberAttribute = ECALDataCursor.currentRow()[std::string("VALUE_NUMBER")];
      coral::Attribute const& elementNrAttribute = ECALDataCursor.currentRow()[std::string("element_nr")];
      if (!dipValAttribute.isNull() and !valueNumberAttribute.isNull()) {
        dipVal = dipValAttribute.data<std::string>();
        elementNr = elementNrAttribute.data<unsigned int>();
        value = valueNumberAttribute.data<float>();
        if (std::isnan(value))
          value = 0.;
        if (filter.process(iovTime)) {
          iovMap.insert(std::make_pair(changeTime, filter.current()->first));
          for (auto it = filter.current(); it != m_tmpBuffer.end(); it++) {
            LHCInfo& payload = *(it->second);
            LHCInfoImpl::setElementData(it->first, dipVal, elementNr, value, payload, initializedVectors);
          }
        }
        //}
      }
    }
  }
  if (m_debug) {
    for (auto& im : iovMap) {
      edm::LogInfo(m_name) << "Found iov=" << im.first << " (" << cond::time::to_boost(im.first) << " ) moved to "
                           << im.second << " ( " << cond::time::to_boost(im.second) << " )";
    }
  }
  return ret;
}

void LHCInfoPopConSourceHandler::addEmptyPayload(cond::Time_t iov) {
  bool add = false;
  if (m_iovs.empty()) {
    if (!m_lastPayloadEmpty)
      add = true;
  } else {
    auto lastAdded = m_iovs.rbegin()->second;
    if (lastAdded->fillNumber() != 0) {
      add = true;
    }
  }
  if (add) {
    auto newPayload = std::make_shared<LHCInfo>();
    m_iovs.insert(std::make_pair(iov, newPayload));
    m_prevPayload = newPayload;
  }
}

namespace LHCInfoImpl {
  bool comparePayloads(const LHCInfo& rhs, const LHCInfo& lhs) {
    if (rhs.fillNumber() != lhs.fillNumber())
      return false;
    if (rhs.delivLumi() != lhs.delivLumi())
      return false;
    if (rhs.recLumi() != lhs.recLumi())
      return false;
    if (rhs.instLumi() != lhs.instLumi())
      return false;
    if (rhs.instLumiError() != lhs.instLumiError())
      return false;
    if (rhs.crossingAngle() != rhs.crossingAngle())
      return false;
    if (rhs.betaStar() != rhs.betaStar())
      return false;
    if (rhs.lhcState() != rhs.lhcState())
      return false;
    if (rhs.lhcComment() != rhs.lhcComment())
      return false;
    if (rhs.ctppsStatus() != rhs.ctppsStatus())
      return false;
    return true;
  }

  size_t transferPayloads(const std::vector<std::pair<cond::Time_t, std::shared_ptr<LHCInfo>>>& buffer,
                          std::map<cond::Time_t, std::shared_ptr<LHCInfo>>& iovsToTransfer,
                          std::shared_ptr<LHCInfo>& prevPayload) {
    size_t niovs = 0;
    for (auto& iov : buffer) {
      bool add = false;
      auto payload = iov.second;
      cond::Time_t since = iov.first;
      if (iovsToTransfer.empty()) {
        add = true;
      } else {
        LHCInfo& lastAdded = *iovsToTransfer.rbegin()->second;
        if (!comparePayloads(lastAdded, *payload)) {
          add = true;
        }
      }
      if (add) {
        niovs++;
        iovsToTransfer.insert(std::make_pair(since, payload));
        prevPayload = iov.second;
      }
    }
    return niovs;
  }

}  // namespace LHCInfoImpl

void LHCInfoPopConSourceHandler::getNewObjects() {
  //reference to the last payload in the tag
  Ref previousFill;

  //if a new tag is created, transfer fake fill from 1 to the first fill for the first time
  if (tagInfo().size == 0) {
    edm::LogInfo(m_name) << "New tag " << tagInfo().name << "; from " << m_name << "::getNewObjects";
  } else {
    //check what is already inside the database
    edm::LogInfo(m_name) << "got info for tag " << tagInfo().name << ": size " << tagInfo().size
                         << ", last object valid since " << tagInfo().lastInterval.since << " ( "
                         << boost::posix_time::to_iso_extended_string(
                                cond::time::to_boost(tagInfo().lastInterval.since))
                         << " ); from " << m_name << "::getNewObjects";
  }

  cond::Time_t lastSince = tagInfo().lastInterval.since;
  if (tagInfo().isEmpty()) {
    // for a new or empty tag, an empty payload should be added on top with since=1
    addEmptyPayload(1);
    lastSince = 1;
  } else {
    edm::LogInfo(m_name) << "The last Iov in tag " << tagInfo().name << " valid since " << lastSince << "from "
                         << m_name << "::getNewObjects";
  }

  boost::posix_time::ptime executionTime = boost::posix_time::second_clock::local_time();
  cond::Time_t targetSince = 0;
  cond::Time_t endIov = cond::time::from_boost(executionTime);
  if (!m_startTime.is_not_a_date_time()) {
    targetSince = cond::time::from_boost(m_startTime);
  }
  if (lastSince > targetSince)
    targetSince = lastSince;

  edm::LogInfo(m_name) << "Starting sampling at "
                       << boost::posix_time::to_simple_string(cond::time::to_boost(targetSince));

  //retrieve the data from the relational database source
  cond::persistency::ConnectionPool connection;
  //configure the connection
  if (m_debug) {
    connection.setMessageVerbosity(coral::Debug);
  } else {
    connection.setMessageVerbosity(coral::Error);
  }
  connection.setAuthenticationPath(m_authpath);
  connection.configure();
  //create the sessions
  cond::persistency::Session session = connection.createSession(m_connectionString, false);
  cond::persistency::Session session2 = connection.createSession(m_ecalConnectionString, false);
  // fetch last payload when available
  if (!tagInfo().lastInterval.payloadId.empty()) {
    cond::persistency::Session session3 = dbSession();
    session3.transaction().start(true);
    m_prevPayload = session3.fetchPayload<LHCInfo>(tagInfo().lastInterval.payloadId);
    session3.transaction().commit();
  }

  bool iovAdded = false;
  while (true) {
    if (targetSince >= endIov) {
      edm::LogInfo(m_name) << "Sampling ended at the time "
                           << boost::posix_time::to_simple_string(cond::time::to_boost(endIov));
      break;
    }
    bool updateEcal = false;
    boost::posix_time::ptime targetTime = cond::time::to_boost(targetSince);
    boost::posix_time::ptime startSampleTime;
    boost::posix_time::ptime endSampleTime;

    cond::OMSService oms;
    oms.connect(m_omsBaseUrl);
    auto query = oms.query("fills");

    if (!m_endFill and m_prevPayload->fillNumber() and m_prevPayload->endTime() == 0ULL) {
      // execute the query for the current fill
      edm::LogInfo(m_name) << "Searching started fill #" << m_prevPayload->fillNumber();
      query->filterEQ("fill_number", m_prevPayload->fillNumber());
      bool foundFill = query->execute();
      if (foundFill)
        foundFill = LHCInfoImpl::makeFillPayload(m_fillPayload, query->result());
      if (!foundFill) {
        edm::LogError(m_name) << "Could not find fill #" << m_prevPayload->fillNumber();
        break;
      }
      updateEcal = true;
      startSampleTime = cond::time::to_boost(lastSince);
    } else {
      edm::LogInfo(m_name) << "Searching new fill after " << boost::posix_time::to_simple_string(targetTime);
      query->filterNotNull("start_stable_beam").filterNotNull("fill_number");
      if (targetTime > cond::time::to_boost(m_prevPayload->createTime())) {
        query->filterGE("start_time", targetTime);
      } else {
        query->filterGT("start_time", targetTime);
      }

      query->filterLT("start_time", m_endTime);
      if (m_endFill)
        query->filterNotNull("end_time");
      bool foundFill = query->execute();
      if (foundFill)
        foundFill = LHCInfoImpl::makeFillPayload(m_fillPayload, query->result());
      if (!foundFill) {
        edm::LogInfo(m_name) << "No fill found - END of job.";
        if (iovAdded)
          addEmptyPayload(targetSince);
        break;
      }
      startSampleTime = cond::time::to_boost(m_fillPayload->createTime());
    }
    cond::Time_t startFillTime = m_fillPayload->createTime();
    cond::Time_t endFillTime = m_fillPayload->endTime();
    unsigned short lhcFill = m_fillPayload->fillNumber();
    if (endFillTime == 0ULL) {
      edm::LogInfo(m_name) << "Found ongoing fill " << lhcFill << " created at " << cond::time::to_boost(startFillTime);
      endSampleTime = executionTime;
      targetSince = endIov;
    } else {
      edm::LogInfo(m_name) << "Found fill " << lhcFill << " created at " << cond::time::to_boost(startFillTime)
                           << " ending at " << cond::time::to_boost(endFillTime);
      endSampleTime = cond::time::to_boost(endFillTime);
      targetSince = endFillTime;
    }

    getDipData(oms, startSampleTime, endSampleTime);
    size_t nlumi = getLumiData(oms, lhcFill, startSampleTime, endSampleTime);
    edm::LogInfo(m_name) << "Found " << nlumi << " lumisections during the fill " << lhcFill;
    boost::posix_time::ptime flumiStart = cond::time::to_boost(m_tmpBuffer.front().first);
    boost::posix_time::ptime flumiStop = cond::time::to_boost(m_tmpBuffer.back().first);
    edm::LogInfo(m_name) << "First lumi starts at " << flumiStart << " last lumi starts at " << flumiStop;
    session.transaction().start(true);
    getCTTPSData(session, startSampleTime, endSampleTime);
    session.transaction().commit();
    session2.transaction().start(true);
    getEcalData(session2, startSampleTime, endSampleTime, updateEcal);
    session2.transaction().commit();
    //
    size_t niovs = LHCInfoImpl::transferPayloads(m_tmpBuffer, m_iovs, m_prevPayload);
    edm::LogInfo(m_name) << "Added " << niovs << " iovs within the Fill time";
    m_tmpBuffer.clear();
    iovAdded = true;
    if (m_prevPayload->fillNumber() and m_fillPayload->endTime() != 0ULL)
      addEmptyPayload(m_fillPayload->endTime());
  }
}

std::string LHCInfoPopConSourceHandler::id() const { return m_name; }
