#include "DataFormats/TCDS/interface/BSTRecord.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

BSTRecord::BSTRecord()
    : m_gpstime(0),
      m_turnCount(0),
      m_lhcFill(0),
      m_intensityBeam1(0),
      m_intensityBeam2(0),
      m_beamMomentum(0),
      m_beamMode(0),
      m_particleBeam1(0),
      m_particleBeam2(0),
      m_bstMaster(0) {}

BSTRecord::BSTRecord(const tcds::BST_v1& bst)
    : m_gpstime(((uint64_t)(bst.gpstimehigh) << 32) | bst.gpstimelow),
      m_turnCount(((uint32_t)(bst.turnCountHigh) << 16) | bst.turnCountLow),
      m_lhcFill(((uint32_t)(bst.lhcFillHigh) << 16) | bst.lhcFillLow),
      m_intensityBeam1(bst.intensityBeam1),
      m_intensityBeam2(bst.intensityBeam2),
      m_beamMomentum(bst.beamMomentum),
      m_beamMode(bst.beamMode),
      m_particleBeam1(bst.particleTypes & 0xFF),
      m_particleBeam2(bst.particleTypes >> 8),
      m_bstMaster(bst.bstMaster >> 8) {
  if (m_beamMomentum == 65535)  // Invalid value
    m_beamMomentum = -1;
  else if (m_lhcFill >= 5698)  // scale factor changed from 1GeV/LSB to 120MeV/LSB
    m_beamMomentum *= 0.120;
}

std::ostream& operator<<(std::ostream& s, const BSTRecord& record) {
  s << "BST record:" << std::endl;
  s << "   GpsTime:            " << record.getGpsTime() << std::endl;
  s << "   BstMaster:          " << (uint16_t)record.getBstMaster() << std::endl;
  s << "   TurnCount:          " << record.getTurnCount() << std::endl;
  s << "   LhcFill:            " << record.getLhcFill() << std::endl;
  s << "   BeamMode:           " << record.getBeamMode() << std::endl;
  s << "   ParticleBeam1:      " << (uint16_t)record.getParticleBeam1() << std::endl;
  s << "   ParticleBeam2:      " << (uint16_t)record.getParticleBeam2() << std::endl;
  s << "   BeamMomentum:       " << record.getBeamMomentum() << " GeV" << std::endl;
  s << "   IntensityBeam1:     " << record.getIntensityBeam1() << " 10E10" << std::endl;
  s << "   IntensityBeam2:     " << record.getIntensityBeam2() << " 10E10" << std::endl;

  return s;
}
