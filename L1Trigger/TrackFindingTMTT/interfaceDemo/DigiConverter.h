#ifndef L1Trigger_TrackFindingTMTT_DigiConverter_h
#define L1Trigger_TrackFindingTMTT_DigiConverter_h

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

#include "Demonstrator/DataFormats/interface/DigiDTCStub.hpp"
#include "Demonstrator/DataFormats/interface/DigiHTStub.hpp"
#include "Demonstrator/DataFormats/interface/DigiHTMiniStub.hpp"
#include "Demonstrator/DataFormats/interface/DigiKF4Track.hpp"

namespace demo {

  /**
 * @brief      Utility class to convert simulation objects into edm did objects
 */

  class DigiConverter {
  public:
    DigiConverter(const tmtt::Settings* settings);
    ~DigiConverter() = default;

    DigiDTCStub makeDigiDTCStub(const tmtt::Stub& aDTCStub, uint32_t aDigiPhiSec) const;
    DigiHTStub makeDigiHTStub(const tmtt::Stub& aHTStub,
                              uint32_t aPhiSectorIdInNon,
                              uint32_t aEtaSectorId,
                              int cBin,
                              int aChiZ,
                              int aChiPhi,
                              bool mSel) const;
    DigiHTMiniStub makeDigiHTMiniStub(const tmtt::Stub& aHTMiniStub,
                                      uint32_t aPhiSectorIdInNon,
                                      uint32_t aEtaSectorId,
                                      int8_t cBin,
                                      int8_t mBin) const;
    DigiKF4Track makeDigiKF4Track(const tmtt::L1fittedTrack& aFitTrk) const;

  private:
    const tmtt::Settings* mSettings;  // Configuration parameters.
  };

}  // namespace demo

#endif /* __DEMONSTRATOR_DATAFORMATS_DIGICONVERTER_H__ */
