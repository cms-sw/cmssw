#ifndef DATAFORMATS_TCDS_BSTRECORD_H
#define DATAFORMATS_TCDS_BSTRECORD_H

//---------------------------------------------------------------------------
//! \class BSTRecord
//!
//! \brief Class representing the Beam Synchronous Timing (BST)
//! information included in the TCDS record.
//! Beam parameters provided by BST are defined in:
//! https://edms.cern.ch/document/638899/2.0
//!
//! \author S. Di Guida - INFN and Marconi University
//! \author Remi Mommsen - Fermilab
//!
//---------------------------------------------------------------------------

#include <ostream>
#include <stdint.h>

#include "DataFormats/TCDS/interface/TCDSRaw.h"


class BSTRecord
{
public:

  enum BeamMode {
    NOMODE   =  0,
    SETUP    =  1,
    INJPILOR =  3,
    INJINTR  =  4,
    INJNOMN  =  5,
    PRERAMP  =  6,
    RAMP     =  7,
    FLATTOP  =  8,
    SQUEEZE  =  9,
    ADJUST   = 10,
    STABLE   = 11,
    UNSTABLE = 12,
    BEAMDUMP = 13,
    RAMPDOWN = 14,
    RECOVERY = 15,
    INJDUMP  = 16,
    CIRCDUMP = 17,
    ABORT    = 18,
    CYCLING  = 19,
    WBDUMP   = 20,
    NOBEAM   = 21,
  };

  BSTRecord();

  BSTRecord(const tcds::BST_v1&);

  // Microseconds since Epoch
  uint64_t const getGpsTime() const  { return gpsTime_; }
  // BST beam master
  uint8_t const getBstMaster() const { return bstMaster_; }
  // Turn count
  uint32_t const getTurnCount() const { return turnCount_; }
  // Fill number
  uint32_t const getLhcFill() const { return lhcFill_; }
  // Beam Mode
  uint16_t const getBeamMode() const { return beamMode_; }
  // Enumerator for particle type in beam 1
  uint8_t const getParticleBeam1() const { return particleBeam1_; }
  // Enumerator for particle type in beam 2
  uint8_t const getParticleBeam2() const { return particleBeam2_; }
  // Beam momentum (GeV/c)
  uint16_t const getBeamMomentum() const { return beamMomentum_; }
  // Intensity of Beam 1 (10E10 charges)
  uint32_t const getIntensityBeam1() const { return intensityBeam1_; }
  // Intensity of Beam 2 (10E10 charges)
  uint32_t const getIntensityBeam2() const { return intensityBeam2_; }

 private:

  uint64_t gpsTime_;
  uint8_t bstMaster_;
  uint32_t turnCount_;
  uint32_t lhcFill_;
  uint16_t beamMode_;
  uint8_t particleBeam1_;
  uint8_t particleBeam2_;
  uint16_t beamMomentum_;
  uint32_t intensityBeam1_;
  uint32_t intensityBeam2_;

};


/// Pretty-print operator for BSTRecord
std::ostream& operator<<(std::ostream&, const BSTRecord&);

#endif // DATAFORMATS_TCDS_BSTRECORD_H
