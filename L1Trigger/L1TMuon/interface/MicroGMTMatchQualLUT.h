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
        MicroGMTMatchQualLUT ();
        explicit MicroGMTMatchQualLUT (const edm::ParameterSet&, std::string, cancel_t cancelType);
        virtual ~MicroGMTMatchQualLUT ();

        int lookup(int dEta, int dPhi) const;
        virtual int lookupPacked(int in) const;
        int hashInput(int dEta, int dPhi) const;
        void unHashInput(int input, int& dEta, int& dPhi) const;

        int getDeltaEtaWidth() const { return m_dEtaRedInWidth; }
        int getDeltaPhiWidth() const { return m_dPhiRedInWidth; }
      private:
        int m_dEtaRedMask; 
        int m_dPhiRedMask; 
        int m_dEtaRedInWidth;
        int m_dPhiRedInWidth;

        double m_etaScale;
        double m_phiScale;

        cancel_t m_cancelType;
    };
}
#endif /* defined(__l1microgmtmatchquallut_h) */