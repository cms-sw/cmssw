// Class for muon tracks in EMTF - AWB 04.01.16
// Mostly copied from L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h

#ifndef DataFormats_L1TMuon_EMTFTrack_h
#define DataFormats_L1TMuon_EMTFTrack_h

#include <cstdint>
#include <vector>

#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFRoad.h"
#include "DataFormats/L1TMuon/interface/EMTF/SP.h"

namespace l1t {

  namespace emtf {
    // Want a scoped enum, not necessarily a strongly-typed enum.
    // This is because of all the casting that will be required throughout legacy code
    // (usage will likely rely on implicit integer casting)
    enum class EMTFCSCStation { ME1 = 0, ME2, ME3, ME4 };
    enum class EMTFCSCSection { ME1sub1 = 0, ME1sub2, ME2, ME3, ME4 };
  }  // namespace emtf

  struct EMTFPtLUT {
    uint64_t address;
    uint16_t mode;
    uint16_t theta;
    uint16_t st1_ring2;
    uint16_t eta;
    uint16_t delta_ph[6];  // index: 0=12, 1=13, 2=14, 3=23, 4=24, 5=34
    uint16_t delta_th[6];  // ^
    uint16_t sign_ph[6];   // ^
    uint16_t sign_th[6];   // ^
    uint16_t cpattern[4];  // index: 0=ME1, 1=ME2, 2=ME3, 3=ME4
    uint16_t csign[4];     // index: 0=ME1, 1=ME2, 2=ME3, 3=ME4
    uint16_t slope[4];     // index: 0=ME1, 1=ME2, 2=ME3, 3=ME4
    uint16_t fr[4];        // ^
    uint16_t bt_vi[5];     // index: 0=ME1sub1, 1=ME1sub2, 2=ME2, 3=ME3, 4=ME4
    uint16_t bt_hi[5];     // ^
    uint16_t bt_ci[5];     // ^
    uint16_t bt_si[5];     // ^
  };

  class EMTFTrack {
  public:
    EMTFTrack()
        : _PtLUT(),
          endcap(-99),
          sector(-99),
          sector_idx(-99),
          mode(-99),
          mode_CSC(0),
          mode_RPC(0),
          mode_GEM(0),
          mode_neighbor(0),
          mode_inv(-99),
          rank(-99),
          winner(-99),
          charge(-99),
          bx(-99),
          first_bx(-99),
          second_bx(-99),
          pt(-99),
          pt_XML(-99),
          pt_dxy(-99),
          dxy(-99),
          zone(-99),
          ph_num(-99),
          ph_q(-99),
          theta_fp(-99),
          theta(-99),
          eta(-99),
          phi_fp(-99),
          phi_loc(-99),
          phi_glob(-999),
          gmt_pt(-99),
          gmt_pt_dxy(-99),
          gmt_dxy(-99),
          gmt_phi(-999),
          gmt_eta(-999),
          gmt_quality(-99),
          gmt_charge(-99),
          gmt_charge_valid(-99),
          track_num(-99),
          numHits(-99){};

    virtual ~EMTFTrack(){};

    void ImportSP(const emtf::SP _SP, int _sector);
    // void ImportPtLUT( int _mode, unsigned long _address );

    void clear_Hits() {
      _Hits.clear();
      numHits = 0;
      mode_CSC = 0;
      mode_RPC = 0;
      mode_GEM = 0;
      mode_neighbor = 0;
    }

    void push_Hit(const EMTFHit& hit) {
      _Hits.push_back(hit);
      numHits = _Hits.size();
      if (hit.Is_CSC())
        mode_CSC |= (1 << (4 - hit.Station()));
      if (hit.Is_RPC())
        mode_RPC |= (1 << (4 - hit.Station()));
      if (hit.Is_GEM())
        mode_GEM |= (1 << (4 - hit.Station()));
      if (hit.Neighbor())
        mode_neighbor |= (1 << (4 - hit.Station()));
    }

    void set_Hits(const EMTFHitCollection& hits) {
      clear_Hits();
      for (const auto& hit : hits)
        push_Hit(hit);
    }

    void clear_HitIdx() { _HitIdx.clear(); }
    void push_HitIdx(unsigned int bits) { _HitIdx.push_back(bits); }
    void set_HitIdx(const std::vector<unsigned int>& bits) { _HitIdx = bits; }

    int NumHits() const { return numHits; }
    EMTFHitCollection Hits() const { return _Hits; }
    std::vector<unsigned int> HitIdx() const { return _HitIdx; }

    void set_PtLUT(EMTFPtLUT bits) { _PtLUT = bits; }
    EMTFPtLUT PtLUT() const { return _PtLUT; }

    void set_endcap(int bits) { endcap = bits; }
    void set_sector(int bits) { sector = bits; }
    void set_sector_idx(int bits) { sector_idx = bits; }
    void set_mode(int bits) { mode = bits; }
    void set_mode_inv(int bits) { mode_inv = bits; }
    void set_rank(int bits) { rank = bits; }
    void set_winner(int bits) { winner = bits; }
    void set_charge(int bits) { charge = bits; }
    void set_bx(int bits) { bx = bits; }
    void set_first_bx(int bits) { first_bx = bits; }
    void set_second_bx(int bits) { second_bx = bits; }
    void set_pt(float val) { pt = val; }
    void set_pt_XML(float val) { pt_XML = val; }
    void set_pt_dxy(float val) { pt_dxy = val; }
    void set_dxy(float val) { dxy = val; }
    void set_zone(int bits) { zone = bits; }
    void set_ph_num(int bits) { ph_num = bits; }
    void set_ph_q(int bits) { ph_q = bits; }
    void set_theta_fp(int bits) { theta_fp = bits; }
    void set_theta(float val) { theta = val; }
    void set_eta(float val) { eta = val; }
    void set_phi_fp(int bits) { phi_fp = bits; }
    void set_phi_loc(float val) { phi_loc = val; }
    void set_phi_glob(float val) { phi_glob = val; }
    void set_gmt_pt(int bits) { gmt_pt = bits; }
    void set_gmt_pt_dxy(int bits) { gmt_pt_dxy = bits; }
    void set_gmt_dxy(int bits) { gmt_dxy = bits; }
    void set_gmt_phi(int bits) { gmt_phi = bits; }
    void set_gmt_eta(int bits) { gmt_eta = bits; }
    void set_gmt_quality(int bits) { gmt_quality = bits; }
    void set_gmt_charge(int bits) { gmt_charge = bits; }
    void set_gmt_charge_valid(int bits) { gmt_charge_valid = bits; }
    void set_track_num(int bits) { track_num = bits; }

    int Endcap() const { return endcap; }
    int Sector() const { return sector; }
    int Sector_idx() const { return sector_idx; }
    int Mode() const { return mode; }
    int Mode_CSC() const { return mode_CSC; }
    int Mode_RPC() const { return mode_RPC; }
    int Mode_GEM() const { return mode_GEM; }
    int Mode_neighbor() const { return mode_neighbor; }
    int Mode_inv() const { return mode_inv; }
    int Rank() const { return rank; }
    int Winner() const { return winner; }
    int Charge() const { return charge; }
    int BX() const { return bx; }
    int First_BX() const { return first_bx; }
    int Second_BX() const { return second_bx; }
    float Pt() const { return pt; }
    float Pt_XML() const { return pt_XML; }
    float Pt_dxy() const { return pt_dxy; }
    float Dxy() const { return dxy; }
    int Zone() const { return zone; }
    int Ph_num() const { return ph_num; }
    int Ph_q() const { return ph_q; }
    int Theta_fp() const { return theta_fp; }
    float Theta() const { return theta; }
    float Eta() const { return eta; }
    int Phi_fp() const { return phi_fp; }
    float Phi_loc() const { return phi_loc; }
    float Phi_glob() const { return phi_glob; }
    int GMT_pt() const { return gmt_pt; }
    int GMT_pt_dxy() const { return gmt_pt_dxy; }
    int GMT_dxy() const { return gmt_dxy; }
    int GMT_phi() const { return gmt_phi; }
    int GMT_eta() const { return gmt_eta; }
    int GMT_quality() const { return gmt_quality; }
    int GMT_charge() const { return gmt_charge; }
    int GMT_charge_valid() const { return gmt_charge_valid; }
    int Track_num() const { return track_num; }

  private:
    EMTFHitCollection _Hits;
    std::vector<unsigned int> _HitIdx;

    EMTFPtLUT _PtLUT;

    int endcap;      //    +/-1.  For ME+ and ME-.
    int sector;      //  1 -  6.
    int sector_idx;  //  0 - 11.  0 - 5 for ME+, 6 - 11 for ME-.
    int mode;        //  0 - 15.
    int mode_CSC;    //  0 - 15, CSC-only
    int mode_RPC;    //  0 - 15, RPC-only
    int mode_GEM;  //  0 - 15, GEM-only // TODO: verify if needed when including GEM, also start the good habit of documenting these
    int mode_neighbor;  // 0 - 15, only neighbor hits
    int mode_inv;       // 15 -  0.
    int rank;           //  0 - 127  (Range? - AWB 03.03.17)
    int winner;         //  0 -  2.  (Range? - AWB 03.03.17)
    int charge;         //    +/-1.  For physical charge (reversed from GMT convention)
    int bx;             // -3 - +3.
    int first_bx;       // -3 - +3.
    int second_bx;      // -3 - +3.
    float pt;           //  0 - 255
    float pt_XML;       //  0 - 999
    float pt_dxy;       //  0 - 127
    float dxy;          //  0 -  3
    int zone;           //  0 -  3.
    int ph_num;
    int ph_q;
    int theta_fp;    //  0 - 127
    float theta;     //  0 - 90.
    float eta;       //  +/-2.5.
    int phi_fp;      // 0 - 4920
    float phi_loc;   // -22 - 60  (Range? - AWB 03.03.17)
    float phi_glob;  //  +/-180.
    int gmt_pt;
    int gmt_pt_dxy;
    int gmt_dxy;
    int gmt_phi;
    int gmt_eta;
    int gmt_quality;
    int gmt_charge;
    int gmt_charge_valid;
    int track_num;  //  0 - ??.  (Range? - AWB 03.03.17)
    int numHits;    //  1 -  4.

  };  // End of class EMTFTrack

  // Define a vector of EMTFTrack
  typedef std::vector<EMTFTrack> EMTFTrackCollection;

}  // End of namespace l1t

#endif /* define DataFormats_L1TMuon_EMTFTrack_h */
