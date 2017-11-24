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
  OnlineBeamSpotRecord(const onlineMetaData::BeamSpot_v1&);
  virtual ~OnlineBeamSpotRecord();

  // Return the time when the beamspot was published
  edm::Timestamp getTimestamp() const { return timestamp_; }

  float getX() const { return x_; }
  float getY() const { return y_; }
  float getZ() const { return z_; }
  float getDxdz() const { return dxdz_; }
  float getDydz() const { return dydz_; }
  float getErrX() const { return errX_; }
  float getErrY() const { return errY_; }
  float getErrZ() const { return errZ_; }
  float getErrDxdz() const { return errDxdz_; }
  float getErrDydz() const { return errDydz_; }
  float getWidthX() const { return widthX_; }
  float getWidthY() const { return widthY_; }
  float getSigmaZ() const { return sigmaZ_; }
  float getErrWidthX() const { return errWidthX_; }
  float getErrWidthY() const { return errWidthY_; }
  float getErrSigmaZ() const { return errSigmaZ_; }


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
