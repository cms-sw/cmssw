#ifndef __l1microgmtmatchquallut_h
#define __l1microgmtmatchquallut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1t {
    enum cancel_t {
        bmtf_bmtf, omtf_bmtf_pos, omtf_emtf_pos, omtf_omtf_pos, emtf_emtf_pos, omtf_bmtf_neg, omtf_emtf_neg, omtf_omtf_neg, emtf_emtf_neg
    };


    class MicroGMTMatchQualLUT : public MicroGMTLUT {
      public:
        MicroGMTMatchQualLUT() : MicroGMTLUT() {};
        explicit MicroGMTMatchQualLUT(l1t::LUT* lut) : MicroGMTLUT(lut) {};
        virtual ~MicroGMTMatchQualLUT() {};

        virtual int lookup(int etaFine, int dEta, int dPhi) const = 0;

        int getDeltaEtaWidth() const { return m_dEtaRedInWidth; }
        int getDeltaPhiWidth() const { return m_dPhiRedInWidth; }
      protected:
        int m_dEtaRedMask; 
        int m_dPhiRedMask; 
        int m_dEtaRedInWidth;
        int m_dPhiRedInWidth;

        double m_etaScale;
        double m_phiScale;

        double m_maxDR;
        double m_fEta;
        double m_fPhi;

        cancel_t m_cancelType;
    };

    // LUT class for LUTs without eta fine bit, the eta fine bit in the lookup function is ignored
    class MicroGMTMatchQualSimpleLUT : public MicroGMTMatchQualLUT {
      public:
        MicroGMTMatchQualSimpleLUT() {};
        explicit MicroGMTMatchQualSimpleLUT(const std::string&, const double maxDR, const double fEta, const double fPhi, cancel_t cancelType);
        explicit MicroGMTMatchQualSimpleLUT(l1t::LUT* lut, cancel_t cancelType);
        virtual ~MicroGMTMatchQualSimpleLUT() {};

        int lookup(int etaFine, int dEta, int dPhi) const;
        virtual int lookupPacked(int in) const;
        int hashInput(int dEta, int dPhi) const;
        void unHashInput(int input, int& dEta, int& dPhi) const;
    };

    // LUT class for LUTs with eta fine bit
    class MicroGMTMatchQualFineLUT : public MicroGMTMatchQualLUT {
      public:
        MicroGMTMatchQualFineLUT() {};
        explicit MicroGMTMatchQualFineLUT(const std::string&, const double maxDR, const double fEta, const double fEtaCoarse, const double fPhi, cancel_t cancelType);
        explicit MicroGMTMatchQualFineLUT(l1t::LUT* lut, cancel_t cancelType);
        virtual ~MicroGMTMatchQualFineLUT() {};

        int lookup(int etaFine, int dEta, int dPhi) const;
        virtual int lookupPacked(int in) const;
        int hashInput(int etaFine, int dEta, int dPhi) const;
        void unHashInput(int input, int& etaFine, int& dEta, int& dPhi) const;
      private:
        int m_etaFineMask;
        double m_fEtaCoarse;
    };
}
#endif /* defined(__l1microgmtmatchquallut_h) */
