#ifndef CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H
#define CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H

#include "CalibTracker/SiStripChannelGain/interface/APVGainStruct.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>
#include <vector>
#include <utility>
#include <cstdint>
#include <unordered_map>

namespace APVGain {

  typedef dqm::legacy::MonitorElement MonitorElement;

  int subdetectorId(uint32_t);
  int subdetectorId(const std::string&);
  int subdetectorSide(uint32_t, const TrackerTopology*);
  int subdetectorSide(const std::string&);
  int subdetectorPlane(uint32_t, const TrackerTopology*);
  int subdetectorPlane(const std::string&);

  std::vector<std::pair<std::string, std::string>> monHnames(std::vector<std::string>, bool, const char* tag);

  struct APVmon {
  public:
    APVmon(int v1, int v2, int v3, MonitorElement* v4)
        : m_subdetectorId(v1), m_subdetectorSide(v2), m_subdetectorPlane(v3), m_monitor(v4) {}

    int getSubdetectorId() { return m_subdetectorId; }

    int getSubdetectorSide() { return m_subdetectorSide; }

    int getSubdetectorPlane() { return m_subdetectorPlane; }

    MonitorElement* getMonitor() { return m_monitor; }

    void printAll() {
      LogDebug("APVGainHelpers") << "subDetectorID:" << m_subdetectorId << std::endl;
      LogDebug("APVGainHelpers") << "subDetectorSide:" << m_subdetectorSide << std::endl;
      LogDebug("APVGainHelpers") << "subDetectorPlane:" << m_subdetectorPlane << std::endl;
      LogDebug("APVGainHelpers") << "histoName:" << m_monitor->getName() << std::endl;
      return;
    }

  private:
    int m_subdetectorId;
    int m_subdetectorSide;
    int m_subdetectorPlane;
    MonitorElement* m_monitor;
  };

  struct APVGainHistograms {
  public:
    APVGainHistograms()
        : Charge_Vs_Index(7),
          Charge_1(),
          Charge_2(),
          Charge_3(),
          Charge_4(),
          Charge_Vs_PathlengthTIB(7),
          Charge_Vs_PathlengthTOB(7),
          Charge_Vs_PathlengthTIDP(7),
          Charge_Vs_PathlengthTIDM(7),
          Charge_Vs_PathlengthTECP1(7),
          Charge_Vs_PathlengthTECP2(7),
          Charge_Vs_PathlengthTECM1(7),
          Charge_Vs_PathlengthTECM2(7),
          NStripAPVs(0),
          NPixelDets(0),
          APVsCollOrdered(),
          APVsColl() {}

    std::vector<dqm::reco::MonitorElement*> Charge_Vs_Index;         /*!< Charge per cm for each detector id */
    std::array<std::vector<dqm::reco::MonitorElement*>, 7> Charge_1; /*!< Charge per cm per layer / wheel */
    std::array<std::vector<dqm::reco::MonitorElement*>, 7> Charge_2; /*!< Charge per cm per layer / wheel without G2 */
    std::array<std::vector<dqm::reco::MonitorElement*>, 7> Charge_3; /*!< Charge per cm per layer / wheel without G1 */
    std::array<std::vector<dqm::reco::MonitorElement*>, 7>
        Charge_4; /*!< Charge per cm per layer / wheel without G1 and G1*/

    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTIB;   /*!< Charge vs pathlength in TIB */
    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTOB;   /*!< Charge vs pathlength in TOB */
    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTIDP;  /*!< Charge vs pathlength in TIDP */
    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTIDM;  /*!< Charge vs pathlength in TIDM */
    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTECP1; /*!< Charge vs pathlength in TECP thin */
    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTECP2; /*!< Charge vs pathlength in TECP thick */
    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTECM1; /*!< Charge vs pathlength in TECP thin */
    std::vector<dqm::reco::MonitorElement*> Charge_Vs_PathlengthTECM2; /*!< Charge vs pathlength in TECP thick */
    mutable std::atomic<unsigned int> NStripAPVs;
    mutable std::atomic<unsigned int> NPixelDets;
    std::vector<std::shared_ptr<stAPVGain>> APVsCollOrdered;
    std::unordered_map<unsigned int, std::shared_ptr<stAPVGain>> APVsColl;
  };

  std::vector<MonitorElement*> FetchMonitor(std::vector<APVmon>, uint32_t, const TrackerTopology* topo = nullptr);
  std::vector<unsigned int> FetchIndices(std::map<unsigned int, APVloc>,
                                         uint32_t,
                                         const TrackerTopology* topo = nullptr);

};  // namespace APVGain

#endif
