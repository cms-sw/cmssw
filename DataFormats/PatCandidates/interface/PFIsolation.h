#ifndef DataFormats_PatCandidates_interface_PFIsolation_h
#define DataFormats_PatCandidates_interface_PFIsolation_h

/*
  \class    pat::PFIsolation PFIsolation.h "DataFormats/PatCandidates/interface/PFIsolation.h"
  \brief Basic class to store components of pf-isolation for pf candidates
  \author   Bennett Marsh
*/


namespace pat {

    class PFIsolation{
    public:
        PFIsolation() :
            chiso_(9999.), nhiso_(9999.), 
            phiso_(9999.), puiso_(9999.) {}

        PFIsolation(float ch, float nh, float ph, float pu) :
            chiso_(ch), nhiso_(nh),
            phiso_(ph), puiso_(pu) {}

        ~PFIsolation() {}

        PFIsolation& operator=(const PFIsolation& iso) {
            chiso_ = iso.chiso_;
            nhiso_ = iso.nhiso_;
            phiso_ = iso.phiso_;
            puiso_ = iso.puiso_;
            return *this;
        }

        float chargedHadronIso()   const { return chiso_; }
        float neutralHadronIso()   const { return nhiso_; }
        float photonIso()          const { return phiso_; }
        float puChargedHadronIso() const { return puiso_; }

    private:
        float chiso_; // charged hadrons from PV
        float nhiso_; // neutral hadrons
        float phiso_; // photons
        float puiso_; // pileup contribution (charged hadrons not from PV)

    };

}

#endif
