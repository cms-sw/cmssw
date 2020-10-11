//-------------------------------------------------
//
/**  \class L1MuBMTrack
 *
 *   L1 Muon Track Candidate
 *
 *
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TRACK_H
#define L1MUBM_TRACK_H

//---------------
// C++ Headers --
//---------------

#include <iosfwd>
#include <string>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackAssParam.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMAddressArray.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegPhi.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegEta.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMTrack;

typedef std::vector<L1MuBMTrack> L1MuBMTrackCollection;

class L1MuBMTrack : public l1t::RegionalMuonCand {
public:
  /// default constructor
  L1MuBMTrack();

  /// constructor
  L1MuBMTrack(const L1MuBMSecProcId&);

  /// copy constructor
  L1MuBMTrack(const L1MuBMTrack&);

  /// destructor
  ~L1MuBMTrack() override;

  /// reset muon candidate
  void reset();

  /// get name of object
  inline std::string name() const { return m_name; }

  /// get pt-code (5 bits)
  inline unsigned int pt() const { return hwPt(); }

  /// get phi-code (8 bits)
  inline unsigned int phi() const { return hwPhi(); }

  /// get eta-code (6 bits)
  inline int eta() const { return hwEta(); }

  /// get fine eta bit
  inline bool fineEtaBit() const { return hwHF(); }

  /// get charge (1 bit)
  inline int charge() const { return hwSign(); }

  /// get quality
  inline unsigned int quality() const { return hwQual(); }

  /// get track-class
  inline TrackClass tc() const { return m_tc; }

  /// is it an empty  muon candidate?
  inline bool empty() const { return m_empty; }

  /// return Sector Processor in which the muon candidate was found
  inline const L1MuBMSecProcId& spid() const { return m_spid; }

  /// get address-array for this muon candidate
  inline L1MuBMAddressArray address() const { return m_addArray; }

  /// get relative address of a given station
  inline int address(int stat) const { return m_addArray.station(stat); }

  /// get the bunch crossing for this muon candidate
  inline int bx() const { return m_bx; }

  /// return number of phi track segments used to form the muon candidate
  inline int numberOfTSphi() const { return m_tsphiList.size(); }

  /// return number of eta track segments used to form the muon candidate
  inline int numberOfTSeta() const { return m_tsetaList.size(); }

  /// return all phi track segments of the muon candidate
  const std::vector<L1MuBMTrackSegPhi>& getTSphi() const { return m_tsphiList; }

  /// return start phi track segment of muon candidate
  const L1MuBMTrackSegPhi& getStartTSphi() const;

  /// return end phi track segment of muon candidate
  const L1MuBMTrackSegPhi& getEndTSphi() const;

  /// return all eta track segments of the muon candidate
  const std::vector<L1MuBMTrackSegEta>& getTSeta() const { return m_tsetaList; }

  /// return start eta track segment of muon candidate
  const L1MuBMTrackSegEta& getStartTSeta() const;

  /// return end eta track segment of muon candidate
  const L1MuBMTrackSegEta& getEndTSeta() const;

  /// enable muon candidate
  inline void enable() {
    m_empty = false;
    setTFIdentifiers(this->spid().sector(), l1t::tftype::bmtf);
  };

  /// disable muon candidate
  inline void disable() { m_empty = true; }

  /// set name of object
  inline void setName(std::string name) { m_name = name; }

  /// set track-class of muon candidate
  inline void setTC(TrackClass tc) { m_tc = tc; }

  /// set phi-code of muon candidate
  inline void setPhi(int phi) { setHwPhi(phi); }

  /// set eta-code of muon candidate
  void setEta(int eta);

  /// set fine eta bit
  inline void setFineEtaBit() { setHwHF(true); }

  /// set pt-code of muon candidate
  inline void setPt(int pt) { setHwPt(pt); }

  /// set charge of muon candidate
  inline void setCharge(int charge) { setHwSign(charge); }

  /// set charge of muon candidate
  inline void setBx(int bx) { m_bx = bx; }

  /// set quality of muon candidate
  inline void setQuality(unsigned int quality) { setHwQual(quality); }

  /// set relative addresses of muon candidate
  inline void setAddresses(const L1MuBMAddressArray& addr) { m_addArray = addr; }

  /// set phi track segments used to form the muon candidate
  void setTSphi(const std::vector<const L1MuBMTrackSegPhi*>& tsList);

  /// set eta track segments used to form the muon candidate
  void setTSeta(const std::vector<const L1MuBMTrackSegEta*>& tsList);

  /// assignment operator
  L1MuBMTrack& operator=(const L1MuBMTrack&);

  /// equal operator
  bool operator==(const L1MuBMTrack&) const;

  /// unequal operator
  bool operator!=(const L1MuBMTrack&) const;

  /// print parameters of muon candidate
  void print() const;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1MuBMTrack&);

  /// define a rank for muon candidates
  static bool rank(const L1MuBMTrack* first, const L1MuBMTrack* second) {
    unsigned short int rank_f = 0;  // rank of first
    unsigned short int rank_s = 0;  // rank of second
    if (first)
      rank_f = first->pt() + 512 * first->quality();
    if (second)
      rank_s = second->pt() + 512 * second->quality();
    return rank_f > rank_s;
  }

private:
  L1MuBMSecProcId m_spid;  // which SP found the track
  std::string m_name;
  bool m_empty;
  TrackClass m_tc;
  int m_bx;

  L1MuBMAddressArray m_addArray;
  std::vector<L1MuBMTrackSegPhi> m_tsphiList;
  std::vector<L1MuBMTrackSegEta> m_tsetaList;
};

#endif
