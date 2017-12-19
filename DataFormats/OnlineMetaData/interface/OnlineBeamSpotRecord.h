#ifndef DATAFORMATS_ONLINEMETADATA_ONLINEBEAMSPOTRECORD_H
#define DATAFORMATS_ONLINEMETADATA_ONLINEBEAMSPOTRECORD_H

//---------------------------------------------------------------------------
//!  \class OnlineBeamSpotRecord
//!  \brief Class to contain online beamspot from soft FED 1022
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------


#include <cstdint>
#include <ostream>

#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"
#include "DataFormats/Provenance/interface/Timestamp.h"


class OnlineBeamSpotRecord
{
public:

  OnlineBeamSpotRecord();
  OnlineBeamSpotRecord(const online::BeamSpot_v1&);
  virtual ~OnlineBeamSpotRecord();

  /// Return the time when the beamspot was published
  const edm::Timestamp& timestamp() const { return timestamp_; }

  float x() const { return x_; }
  float y() const { return y_; }
  float z() const { return z_; }
  float dxdz() const { return dxdz_; }
  float dydz() const { return dydz_; }
  float errX() const { return errX_; }
  float errY() const { return errY_; }
  float errZ() const { return errZ_; }
  float errDxdz() const { return errDxdz_; }
  float errDydz() const { return errDydz_; }
  float widthX() const { return widthX_; }
  float widthY() const { return widthY_; }
  float sigmaZ() const { return sigmaZ_; }
  float errWidthX() const { return errWidthX_; }
  float errWidthY() const { return errWidthY_; }
  float errSigmaZ() const { return errSigmaZ_; }


private:

  edm::Timestamp timestamp_;
  float x_;
  float y_;
  float z_;
  float dxdz_;
  float dydz_;
  float errX_;
  float errY_;
  float errZ_;
  float errDxdz_;
  float errDydz_;
  float widthX_;
  float widthY_;
  float sigmaZ_;
  float errWidthX_;
  float errWidthY_;
  float errSigmaZ_;

};

/// Pretty-print operator for OnlineBeamSpotRecord
std::ostream& operator<<(std::ostream&, const OnlineBeamSpotRecord&);

#endif // DATAFORMATS_ONLINEMETADATA_ONLINEBEAMSPOTRECORD_H
