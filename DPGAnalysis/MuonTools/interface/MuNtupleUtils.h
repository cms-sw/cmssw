#ifndef MuNtuple_MuNtupleUtils_h
#define MuNtuple_MuNtupleUtils_h

/** \class MuNtupleUtils MuNtupleUtils.h MuDPGAnalysis/MuNtuples/src/MuNtupleUtils.h
 *  
 * A set of helper classes class to handle :
 * - Handing of InputTags and tokens
 * - DB information from edm::EventSetup
 * - Conversion between L1T trigger primitive coordinates and CMS global ones
 *
 * \author C. Battilana - INFN (BO)
 * \author L. Giuducci - INFN (BO)
 *
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <array>
#include <string>

class L1MuDTChambPhDigi;
class L1Phase2MuDTPhDigi;

namespace nano_mu {

  template <class T>
  class EDTokenHandle {
  public:
    /// Constructor
    EDTokenHandle(const edm::ParameterSet& config, edm::ConsumesCollector&& collector, std::string name)
        : m_name{name}, m_inputTag{config.getParameter<edm::InputTag>(name)} {
      if (m_inputTag.label() != "none") {
        m_token = collector.template consumes<T>(m_inputTag);
      }
    }

    /// Conditional getter
    /// checks whether a token is valid and if
    /// retireving the data collection succeded
    auto conditionalGet(const edm::Event& ev) const {
      edm::Handle<T> collection;

      if (!m_token.isUninitialized() && !ev.getByToken(m_token, collection))
        edm::LogError("") << "[EDTokenHandle]::conditionalGet: " << m_inputTag.label()
                          << " collection does not exist !!!";

      return collection;
    }

  private:
    std::string m_name;
    edm::InputTag m_inputTag;
    edm::EDGetTokenT<T> m_token;
  };

  template <class T, class R, edm::Transition TR = edm::Transition::Event>
  class ESTokenHandle {
  public:
    /// Constructor
    ESTokenHandle(edm::ConsumesCollector&& collector, const std::string& label = "")
        : m_token{collector.template esConsumes<TR>(edm::ESInputTag{"", label})} {}

    ///Get Handle from ES
    void getFromES(const edm::EventSetup& environment) { m_handle = environment.getHandle(m_token); }

    /// Check validity
    bool isValid() { return m_handle.isValid(); }

    /// Return handle
    T const* operator->() { return m_handle.product(); }

  private:
    edm::ESGetToken<T, R> m_token;
    edm::ESHandle<T> m_handle;
  };

  class DTTrigGeomUtils {
  public:
    struct chambCoord {
      double pos{};
      double dir{};
    };

    /// Constructor
    DTTrigGeomUtils(edm::ConsumesCollector&& collector, bool dirInDeg = true);

    /// Return local position and direction in chamber RF - legacy
    chambCoord trigToReco(const L1MuDTChambPhDigi* trig);

    /// Return local position and direction in chamber RF - analytical method
    chambCoord trigToReco(const L1Phase2MuDTPhDigi* trig);

    /// Checks id the chamber has positive RF;
    bool hasPosRF(int wh, int sec) { return wh > 0 || (wh == 0 && sec % 4 > 1); };

    /// Update EventSetup information
    void getFromES(const edm::Run& run, const edm::EventSetup& environment) {
      m_dtGeom.getFromES(environment);
      for (int i_st = 0; i_st != 4; ++i_st) {
        const DTChamberId chId(-2, i_st + 1, 4);
        const DTChamber* chamb = m_dtGeom->chamber(chId);
        const DTSuperLayer* sl1 = chamb->superLayer(DTSuperLayerId(chId, 1));
        const DTSuperLayer* sl3 = chamb->superLayer(DTSuperLayerId(chId, 3));
        m_zsl1[i_st] = chamb->surface().toLocal(sl1->position()).z();
        m_zsl3[i_st] = chamb->surface().toLocal(sl3->position()).z();
        m_zcn[i_st] = 0.5 * (m_zsl1[i_st] + m_zsl3[i_st]);
      }
    };

  private:
    ESTokenHandle<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun> m_dtGeom;

    std::array<double, 4> m_zcn;
    std::array<double, 4> m_zsl1;
    std::array<double, 4> m_zsl3;

    static constexpr double PH1_PHI_R = 4096.;
    static constexpr double PH1_PHIB_R = 512.;

    static constexpr double PH2_PHI_R = 65536. / 0.8;
    static constexpr double PH2_PHIB_R = 2048. / 1.4;
  };

}  // namespace nano_mu

#endif
