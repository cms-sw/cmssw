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

#include <stdint.h>

#include "DataFormats/TCDS/interface/TCDSRaw.h"


class BSTRecord
{
public:

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
