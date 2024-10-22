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
#include <cstdint>

#include "DataFormats/TCDS/interface/TCDSRaw.h"

class BSTRecord {
public:
  enum BeamMode {
    NOMODE = 0,
    SETUP = 1,
    INJPILOR = 3,
    INJINTR = 4,
    INJNOMN = 5,
    PRERAMP = 6,
    RAMP = 7,
    FLATTOP = 8,
    SQUEEZE = 9,
    ADJUST = 10,
    STABLE = 11,
    UNSTABLE = 12,
    BEAMDUMP = 13,
    RAMPDOWN = 14,
    RECOVERY = 15,
    INJDUMP = 16,
    CIRCDUMP = 17,
    ABORT = 18,
    CYCLING = 19,
    WBDUMP = 20,
    NOBEAM = 21,
  };

  enum Particle {
    PROTON = 0,
    LEAD = 1,
  };

  BSTRecord();

  BSTRecord(const tcds::BST_v1&);

  // Microseconds since Epoch
  uint64_t const getGpsTime() const { return m_gpstime; }

  // BST beam master
  uint8_t const getBstMaster() const { return m_bstMaster; }

  // Turn count
  uint32_t const getTurnCount() const { return m_turnCount; }

  // Fill number
  uint32_t const getLhcFill() const { return m_lhcFill; }

  // Beam Mode. The return value corresponds to BSTRecord::BeamMode
  uint16_t const getBeamMode() const { return m_beamMode; }

  // Particle type BSTRecord::Particle in beam 1
  uint8_t const getParticleBeam1() const { return m_particleBeam1; }

  // Particle type BSTRecord::Particle in beam 2
  uint8_t const getParticleBeam2() const { return m_particleBeam2; }

  // Beam momentum (GeV/c). Returns -1 if no valid value is available
  int32_t const getBeamMomentum() const { return m_beamMomentum; }

  // Intensity of Beam 1 (10E10 charges)
  uint32_t const getIntensityBeam1() const { return m_intensityBeam1; }

  // Intensity of Beam 2 (10E10 charges)
  uint32_t const getIntensityBeam2() const { return m_intensityBeam2; }

private:
  uint64_t m_gpstime;
  uint32_t m_turnCount;
  uint32_t m_lhcFill;
  uint32_t m_intensityBeam1;
  uint32_t m_intensityBeam2;
  int32_t m_beamMomentum;
  uint16_t m_beamMode;
  uint8_t m_particleBeam1;
  uint8_t m_particleBeam2;
  uint8_t m_bstMaster;
};

/// Pretty-print operator for BSTRecord
std::ostream& operator<<(std::ostream&, const BSTRecord&);

#endif  // DATAFORMATS_TCDS_BSTRECORD_H
