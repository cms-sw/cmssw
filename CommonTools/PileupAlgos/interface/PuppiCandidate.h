#ifndef CommonTools_PileupAlgos_PuppiCandidate
#define CommonTools_PileupAlgos_PuppiCandidate

#include "fastjet/PseudoJet.hh"

const double pseudojet_invalid_eta = -1e200;

// Extension of fastjet::PseudoJet that caches eta and some other info
// Puppi uses register to decide between NHs, PV CHs, and PU CHs.
class PuppiCandidate : public fastjet::PseudoJet {
  public:
    using fastjet::PseudoJet::PseudoJet;
    double pseudorapidity() const { _ensure_valid_eta(); return _eta; }
    double eta() const { return pseudorapidity(); }
    void _ensure_valid_eta() const { if(_eta==pseudojet_invalid_eta) _eta = fastjet::PseudoJet::pseudorapidity(); }
    void set_info(int puppi_register) { puppi_register_ = puppi_register; }
    inline int puppi_register() const { return puppi_register_; }
  private:
    mutable double _eta = pseudojet_invalid_eta;
    int puppi_register_;
};

#endif
