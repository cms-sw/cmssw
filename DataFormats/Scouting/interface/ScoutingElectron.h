#ifndef DataFormats_ScoutingElectron_h
#define DataFormats_ScoutingElectron_h

#include <vector>

// Class for holding electron information, for use in data scouting
// IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class ScoutingElectron
{
    public:
        //constructor with values for all data fields
        ScoutingElectron(float pt, float eta, float phi, float m, float d0, float dz, float dEtaIn,
                         float dPhiIn, float sigmaIetaIeta, float hOverE, float ooEMOop,
                         int missingHits, int charge, float ecalIso, float hcalIso, float trackIso):
            pt_(pt), eta_(eta), phi_(phi), m_(m), d0_(d0), dz_(dz), dEtaIn_(dEtaIn),
            dPhiIn_(dPhiIn), sigmaIetaIeta_(sigmaIetaIeta), hOverE_(hOverE), ooEMOop_(ooEMOop),
            missingHits_(missingHits), charge_(charge), ecalIso_(ecalIso), hcalIso_(hcalIso),
            trackIso_(trackIso) {}
        //default constructor
        ScoutingElectron(): pt_(0), eta_(0), phi_(0), m_(0), d0_(0), dz_(0), dEtaIn_(0), dPhiIn_(0),
            sigmaIetaIeta_(0), hOverE_(0), ooEMOop_(0), missingHits_(0), charge_(0), ecalIso_(0),
            hcalIso_(0), trackIso_(0) {}

        //accessor functions
        float pt() const { return pt_; }
        float eta() const { return eta_; }
        float phi() const { return phi_; }
        float m() const { return m_; }
        float d0() const { return d0_; }
        float dz() const { return dz_; }
        float dEtaIn() const { return dEtaIn_; }
        float dPhiIn() const { return dPhiIn_; }
        float sigmaIetaIeta() const { return sigmaIetaIeta_; }
        float hOverE() const { return hOverE_; }
        float ooEMOop() const { return ooEMOop_; }
        int missingHits() const { return missingHits_; }
        int charge() const { return charge_; }
        float ecalIso() const { return ecalIso_; }
        float hcalIso() const { return hcalIso_; }
        float trackIso() const { return trackIso_; }

    private:
        float pt_;
        float eta_;
        float phi_;
        float m_;
        float d0_;
        float dz_;
        float dEtaIn_;
        float dPhiIn_;
        float sigmaIetaIeta_;
        float hOverE_;
        float ooEMOop_;
        int missingHits_;
        int charge_;
        float ecalIso_;
        float hcalIso_;
        float trackIso_;
};

typedef std::vector<ScoutingElectron> ScoutingElectronCollection;

#endif
