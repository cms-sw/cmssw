#ifndef CommonTools_PileupAlgos_PuppiCandidate
#define CommonTools_PileupAlgos_PuppiCandidate

#include "fastjet/PseudoJet.hh"

// Extension of fastjet::PseudoJet that caches eta and some other info
// Puppi uses register to decide between NHs, PV CHs, and PU CHs.
class PuppiCandidate : public fastjet::PseudoJet {
  public:
    using fastjet::PseudoJet::PseudoJet;
    double pseudorapidity() const { _ensure_valid_eta(); return _eta; }
    double eta() const { return pseudorapidity(); }
    void _ensure_valid_eta() const { if(_eta==fastjet::pseudojet_invalid_rap) _eta = fastjet::PseudoJet::pseudorapidity(); }
    void set_info(int puppi_register) { _puppi_register = puppi_register; }
    inline int puppi_register() const { return _puppi_register; }
  private:
    // variable names in fastjet style
    mutable double _eta = fastjet::pseudojet_invalid_rap;
    int _puppi_register;
};

#endif
