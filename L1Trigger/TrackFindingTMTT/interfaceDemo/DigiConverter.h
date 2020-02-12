#ifndef __DEMONSTRATOR_DATAFORMATS_DIGICONVERTER_H__
#define __DEMONSTRATOR_DATAFORMATS_DIGICONVERTER_H__

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
    DigiConverter( const TMTT::Settings* settings );
    ~DigiConverter() = default;

    DigiDTCStub makeDigiDTCStub( const TMTT::Stub& aDTCStub, uint32_t aDigiPhiSec ) const;
    DigiHTStub makeDigiHTStub( const TMTT::Stub& aHTStub, uint32_t aPhiSectorIdInNon, uint32_t aEtaSectorId, int cBin, int aChiZ, int aChiPhi, bool mSel) const;
    DigiHTMiniStub makeDigiHTMiniStub( const TMTT::Stub& aHTMiniStub, uint32_t aPhiSectorIdInNon, uint32_t aEtaSectorId, int8_t cBin, int8_t mBin) const;
    DigiKF4Track makeDigiKF4Track( const TMTT::L1fittedTrack& aFitTrk ) const;

  private:

    const TMTT::Settings *mSettings; // Configuration parameters.

};



} // demo

#endif /* __DEMONSTRATOR_DATAFORMATS_DIGICONVERTER_H__ */
