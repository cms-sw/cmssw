// Class for RPC trigger primitives sent from CPPF to EMTF
// Author Alejandro Segura -- Universidad de los Andes

#ifndef DataFormats_L1TMuon_CPPFDigi_h
#define DataFormats_L1TMuon_CPPFDigi_h

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <vector>

namespace l1t {

  class CPPFDigi {
  public:
    CPPFDigi()
        : bx_(-99),
          phi_int_(-99),
          theta_int_(-99),
          valid_(-99),
          board_(-99),
          channel_(-99),
          emtf_sector_(-99),
          emtf_link_(-99),
          first_strip_(-99),
          cluster_size_(-99),
          phi_glob_(-99),
          theta_glob_(-99){};

    explicit CPPFDigi(const RPCDetId& rpcId0,
                      int bx0,
                      int phi_int0,
                      int theta_int0,
                      int valid0,
                      int board0,
                      int channel0,
                      int emtf_sector0,
                      int emtf_link0,
                      int first_strip0,
                      int cluster_size0,
                      float phi_glob0,
                      float theta_glob0);

    virtual ~CPPFDigi(){};

    virtual CPPFDigi* clone() const;

    CPPFDigi(const RPCDetId& rpcId0, int bx0);
    CPPFDigi(const RPCDetId& rpcId0, int bx0, int theta_int0, int phi_int0);

    bool operator<(const CPPFDigi& rhs) const;

    RPCDetId rpcId() const { return rpcId_; }
    int bx() const { return bx_; }
    int phi_int() const { return phi_int_; }
    int theta_int() const { return theta_int_; }
    int valid() const { return valid_; }
    int board() const { return board_; }
    int channel() const { return channel_; }
    int emtf_sector() const { return emtf_sector_; }
    int emtf_link() const { return emtf_link_; }
    int first_strip() const { return first_strip_; }
    int cluster_size() const { return cluster_size_; }
    float phi_glob() const { return phi_glob_; }
    float theta_glob() const { return theta_glob_; }

  private:
    RPCDetId rpcId_;  // RPC detector ID (includes endcap, ring, station, sector, and chamber)
    int bx_;          // Bunch crossing, signed, centered at 0
    int phi_int_;     // Local integer phi value within an EMTF sector, represents 1/15 degree
    int theta_int_;   // Integer theta value in EMTF scale, represents 36.5/32 degree
    int valid_;
    int board_;         // CPPF board, 1 - 4 in each endcap
    int channel_;       // CPPF output link, 0 - 8 in each board
    int emtf_sector_;   // EMTF sector, 1 - 6 in each endcap
    int emtf_link_;     // EMTF input link, 0 - 6 in each sector
    int first_strip_;   // Lowest-numbered strip in the cluster
    int cluster_size_;  // Number of strips in the cluster
    float phi_glob_;    // Global phi coordinate in degrees, from -180 to 180
    float theta_glob_;  // Global theta coordinate in degrees, from 0 to 90

  };  // End of class CPPFDigi

  // Define a collection of CPPFDigis
  typedef std::vector<CPPFDigi> CPPFDigiCollection;

}  // End of namespace l1t

/// The ostream operator
std::ostream& operator<<(std::ostream& o, const l1t::CPPFDigi& cppf);

#endif /* #define DataFormats_L1TMuon_CPPFDigi_h */
