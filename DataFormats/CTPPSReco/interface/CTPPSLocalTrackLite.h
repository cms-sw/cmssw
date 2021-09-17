/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSLocalTrackLite
#define DataFormats_CTPPSReco_CTPPSLocalTrackLite

#include <cstdint>

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrackRecoInfo.h"

/**
 *\brief Local (=single RP) track with essential information only.
 **/
class CTPPSLocalTrackLite {
public:
  CTPPSLocalTrackLite()
      : rp_id_(0),
        x_(0.),
        x_unc_(-1.),
        y_(0.),
        y_unc_(-1.),
        tx_(999.),
        tx_unc_(-1.),
        ty_(999.),
        ty_unc_(-1.),
        chi2_norm_(-1.),
        pixel_track_reco_info_(CTPPSpixelLocalTrackReconstructionInfo::invalid),
        num_points_fit_(0),
        time_(0.),
        time_unc_(-1.) {}

  CTPPSLocalTrackLite(uint32_t pid,
                      float px,
                      float pxu,
                      float py,
                      float pyu,
                      float ptx,
                      float ptxu,
                      float pty,
                      float ptyu,
                      float pchiSquaredOverNDF,
                      CTPPSpixelLocalTrackReconstructionInfo ppixelTrack_reco_info,
                      unsigned short pNumberOfPointsUsedForFit,
                      float pt,
                      float ptu)
      : rp_id_(pid),
        x_(px),
        x_unc_(pxu),
        y_(py),
        y_unc_(pyu),
        tx_(ptx),
        tx_unc_(ptxu),
        ty_(pty),
        ty_unc_(ptyu),
        chi2_norm_(pchiSquaredOverNDF),
        pixel_track_reco_info_(ppixelTrack_reco_info),
        num_points_fit_(pNumberOfPointsUsedForFit),
        time_(pt),
        time_unc_(ptu) {}

  /// returns the RP id
  inline uint32_t rpId() const { return rp_id_; }

  /// returns the horizontal track position
  inline float x() const { return x_; }

  /// returns the horizontal track position uncertainty
  inline float xUnc() const { return x_unc_; }

  /// returns the vertical track position
  inline float y() const { return y_; }

  /// returns the vertical track position uncertainty
  inline float yUnc() const { return y_unc_; }

  /// returns the track time
  inline float time() const { return time_; }

  /// returns the track time uncertainty
  inline float timeUnc() const { return time_unc_; }

  /// returns the track horizontal angle
  inline float tx() const { return tx_; }

  /// returns the track horizontal angle uncertainty
  inline float txUnc() const { return tx_unc_; }

  /// returns the track vertical angle
  inline float ty() const { return ty_; }

  /// returns the track vertical angle uncertainty
  inline float tyUnc() const { return ty_unc_; }

  /// returns the track fit chi Squared over NDF
  inline float chiSquaredOverNDF() const { return chi2_norm_; }

  /// returns the track reconstruction info byte
  inline CTPPSpixelLocalTrackReconstructionInfo pixelTrackRecoInfo() const { return pixel_track_reco_info_; }

  /// returns the number of points used for fit
  inline unsigned short numberOfPointsUsedForFit() const { return num_points_fit_; }

protected:
  /// RP id
  uint32_t rp_id_;

  /// local track parameterization
  /// x = x0 + tx*(z-z0), y = y0 + ty*(z-z0)
  /// x0, y0, z-z0 in mm
  /// z0: position of the reference scoring plane (in the middle of the RP)

  /// horizontal hit position, mm
  float x_;
  /// uncertainty on horizontal hit position, mm
  float x_unc_;
  /// vertical hit position, mm
  float y_;
  /// uncertainty on vertical hit position, mm
  float y_unc_;
  /// horizontal angle, x = x0 + tx*(z-z0)
  float tx_;
  /// uncertainty on horizontal angle
  float tx_unc_;
  /// vertical angle, y = y0 + ty*(z-z0)
  float ty_;
  /// uncertainty on vertical angle
  float ty_unc_;
  /// fit \f$\chi^2\f$/NDF
  float chi2_norm_;

  /// Track information byte for bx-shifted runs:
  /// * notShiftedRun    -> Default value for tracks reconstructed in non-bx-shifted ROCs
  /// * allShiftedPlanes -> Track reconstructed in a bx-shifted ROC with bx-shifted planes only
  /// * noShiftedPlanes  -> Track reconstructed in a bx-shifted ROC with non-bx-shifted planes only
  /// * mixedPlanes      -> Track reconstructed in a bx-shifted ROC both with bx-shifted and non-bx-shifted planes
  /// * invalid          -> Dummy value. Assigned when pixelTrack_reco_info is not computed (i.e. non-pixel tracks)
  CTPPSpixelLocalTrackReconstructionInfo pixel_track_reco_info_;

  /// number of points used for fit
  unsigned short num_points_fit_;

  /// time information, ns
  float time_;
  /// uncertainty on time information, ns
  float time_unc_;
};

#endif
