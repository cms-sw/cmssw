/** \class BSTRecord
 *
 * Class representing the Beam Synchronous Timing (BST)
 * information included in the TCDS record.
 * Beam parameters provided by BST are defined in:
 * https://edms.cern.ch/document/638899/2.0
 *
 * \author S. Di Guida - INFN and Marconi University
 */

class BSTRecord {
 public:
  BSTRecord():
    m_gpstime(0),
    m_turnCount(0), m_lhcFill(0),
    m_intensityBeam1(0), m_intensityBeam2(0),
    m_beamMode(0),
    m_beamMomentum(0),
    m_bstMaster(0),
    m_particleBeam1(0), m_particleBeam2(0) {}

  void set(unsigned long long gpstime, unsigned char bstMaster,
           unsigned int turnCount, unsigned int lhcFill,
           unsigned short beamMode, unsigned char particleBeam1, unsigned char particleBeam2,
           unsigned short beamMomentum, unsigned int intensityBeam1, unsigned int intensityBeam2) {
    m_gpstime = gpstime;
    m_bstMaster = bstMaster;
    m_turnCount = turnCount;
    m_lhcFill = lhcFill;
    m_beamMode = beamMode;
    m_particleBeam1 = particleBeam1;
    m_particleBeam2 = particleBeam2;
    m_beamMomentum = beamMomentum;
    m_intensityBeam1 = intensityBeam1;
    m_intensityBeam2 = intensityBeam2;
  }
  // Microseconds since Epoch
  unsigned long long const gpstime() const  { return m_gpstime; }
  // BST beam master
  unsigned char const bstMaster() const { return m_bstMaster; }
  // Turn count
  unsigned int const turnCount() const { return m_turnCount; }
  // Fill number
  unsigned int const lhcFill() const { return m_lhcFill; }
  // Beam Mode
  unsigned short const beamMode() const { return m_beamMode; }
  // Enumerator for particle type in beam 1
  unsigned char const particleBeam1() const { return m_particleBeam1; }
  // Enumerator for particle type in beam 2
  unsigned char const particleBeam2() const { return m_particleBeam2; }
  // Beam momentum (GeV/c)
  unsigned short const beamMomentum() const { return m_beamMomentum; }
  // Intensity of Beam 1 (10E10 charges)
  unsigned int const intensityBeam1() const { return m_intensityBeam1; }
  // Intensity of Beam 2 (10E10 charges)
  unsigned int const intensityBeam2() const { return m_intensityBeam2; }

 private:
  unsigned long long m_gpstime;
  unsigned int m_turnCount, m_lhcFill;
  unsigned int m_intensityBeam1, m_intensityBeam2;
  unsigned short m_beamMode;
  unsigned short m_beamMomentum;
  unsigned char m_bstMaster;
  unsigned char m_particleBeam1, m_particleBeam2;
};
