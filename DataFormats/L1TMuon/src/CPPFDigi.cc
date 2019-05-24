// Class for RPC trigger primitives sent from CPPF to EMTF
// Author Alejandro Segura -- Universidad de los Andes

#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include <iostream>

namespace l1t {

  CPPFDigi::CPPFDigi(const RPCDetId& rpcId0, int bx0)
      : rpcId_(rpcId0),
        bx_(bx0),
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
        theta_glob_(-99) {}

  CPPFDigi::CPPFDigi(const RPCDetId& rpcId0, int bx0, int theta_int0, int phi_int0)
      : rpcId_(rpcId0),
        bx_(bx0),
        phi_int_(phi_int0),
        theta_int_(theta_int0),
        valid_(-99),
        board_(-99),
        channel_(-99),
        emtf_sector_(-99),
        emtf_link_(-99),
        first_strip_(-99),
        cluster_size_(-99),
        phi_glob_(-99),
        theta_glob_(-99) {}

  CPPFDigi::CPPFDigi(const RPCDetId& rpcId0,
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
                     float theta_glob0)
      : rpcId_(rpcId0),
        bx_(bx0),
        phi_int_(phi_int0),
        theta_int_(theta_int0),
        valid_(valid0),
        board_(board0),
        channel_(channel0),
        emtf_sector_(emtf_sector0),
        emtf_link_(emtf_link0),
        first_strip_(first_strip0),
        cluster_size_(cluster_size0),
        phi_glob_(phi_glob0),
        theta_glob_(theta_glob0) {}

  CPPFDigi* CPPFDigi::clone() const { return new CPPFDigi(*this); }

  bool CPPFDigi::operator<(const CPPFDigi& rhs) const {
    return (rpcId().rawId() < rhs.rpcId().rawId() ||
            (!(rhs.rpcId().rawId() < rpcId().rawId()) &&
             (bx() < rhs.bx() ||
              (!(rhs.bx() < bx()) &&
               (theta_int() < rhs.theta_int() || (!(rhs.theta_int() < theta_int()) && phi_int() < rhs.phi_int()))))));
  }

}  // End namespace l1t

std::ostream& operator<<(std::ostream& o, const l1t::CPPFDigi& cppf) {
  o << "Local integer phi: " << cppf.phi_int();
  o << "Local integer theta: " << cppf.theta_int();
  return o;
}
