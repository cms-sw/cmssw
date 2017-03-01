#ifndef DataFormats_ScoutingPhoton_h
#define DataFormats_ScoutingPhoton_h

#include <vector>

// Class for holding photon information, for use in data scouting
// IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class ScoutingPhoton
{
    public:
        //constructor with values for all data fields
        ScoutingPhoton(float pt, float eta, float phi, float m, float sigmaIetaIeta, float hOverE,
                       float ecalIso, float hcalIso):
            pt_(pt), eta_(eta), phi_(phi), m_(m), sigmaIetaIeta_(sigmaIetaIeta), hOverE_(hOverE),
            ecalIso_(ecalIso), hcalIso_(hcalIso) {}
        //default constructor
        ScoutingPhoton(): pt_(0), eta_(0), phi_(0), m_(0), sigmaIetaIeta_(0), hOverE_(0),
            ecalIso_(0), hcalIso_(0) {}

        //accessor functions
        float pt() const { return pt_; }
        float eta() const { return eta_; }
        float phi() const { return phi_; }
        float m() const { return m_; }
        float sigmaIetaIeta() const { return sigmaIetaIeta_; }
        float hOverE() const { return hOverE_; }
        float ecalIso() const { return ecalIso_; }
        float hcalIso() const { return hcalIso_; }

    private:
        float pt_;
        float eta_;
        float phi_;
        float m_;
        float sigmaIetaIeta_;
        float hOverE_;
        float ecalIso_;
        float hcalIso_;
};

typedef std::vector<ScoutingPhoton> ScoutingPhotonCollection;

#endif
