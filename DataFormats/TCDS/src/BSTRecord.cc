#include "DataFormats/TCDS/interface/BSTRecord.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

BSTRecord::BSTRecord() :
  gpsTime_(0),
  bstMaster_(0),
  turnCount_(0),
  lhcFill_(0),
  beamMode_(0),
  particleBeam1_(0),
  particleBeam2_(0),
  beamMomentum_(0),
  intensityBeam1_(0),
  intensityBeam2_(0)
{}


BSTRecord::BSTRecord(const tcds::BST_v1& bst) :
  gpsTime_(((uint64_t)(bst.gpstimehigh)<<32)|bst.gpstimelow),
  bstMaster_(bst.bstMaster >> 8),
  turnCount_(((uint32_t)(bst.turnCountHigh)<<16)|bst.turnCountLow),
  lhcFill_(((uint32_t)(bst.lhcFillHigh)<<16)|bst.lhcFillLow),
  beamMode_(bst.beamMode),
  particleBeam1_(bst.particleTypes & 0xFF),
  particleBeam2_(bst.particleTypes >> 8),
  beamMomentum_(bst.beamMomentum),
  intensityBeam1_(bst.intensityBeam1),
  intensityBeam2_(bst.intensityBeam2)
{}


std::ostream& operator<<(std::ostream& s, const BSTRecord& record)
{
  s << "BST record:" << std::endl;
  s << "   GpsTime:            " << record.getGpsTime() << std::endl;
  s << "   BstMaster:          " << record.getBstMaster() << std::endl;
  s << "   TurnCount:          " << record.getTurnCount() << std::endl;
  s << "   LhcFill:            " << record.getLhcFill() << std::endl;
  s << "   BeamMode:           " << record.getBeamMode() << std::endl;
  s << "   ParticleBeam1:      " << record.getParticleBeam1() << std::endl;
  s << "   ParticleBeam2:      " << record.getParticleBeam2() << std::endl;
  s << "   BeamMomentum:       " << record.getBeamMomentum() << " GeV" << std::endl;
  s << "   IntensityBeam1:     " << record.getIntensityBeam1() << " 10E10" << std::endl;
  s << "   IntensityBeam2:     " << record.getIntensityBeam2() << " 10E10" << std::endl;

  return s;
}
