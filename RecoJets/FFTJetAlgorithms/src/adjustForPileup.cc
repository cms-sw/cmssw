#include "RecoJets/FFTJetAlgorithms/interface/adjustForPileup.h"

namespace fftjetcms {
    math::XYZTLorentzVector adjustForPileup(
        const math::XYZTLorentzVector& jet,
        const math::XYZTLorentzVector& pileup,
        const bool subtractPileupAs4Vec)
    {
        const double pt = jet.Pt();
        if (pt > 0.0)
        {
            const double pileupPt = pileup.Pt();
            const double ptFactor = (pt - pileupPt)/pt;
            if (ptFactor <= 0.0)
                return math::XYZTLorentzVector();
            else if (subtractPileupAs4Vec)
            {
                const math::XYZTLorentzVector subtracted(jet - pileup);
                const double e = subtracted.E();
                if (e <= 0.0)
                    return math::XYZTLorentzVector();
                else
                {
                    // Avoid negative jet masses
                    const double px = subtracted.Px();
                    const double py = subtracted.Py();
                    const double pz = subtracted.Pz();
                    if (e*e < px*px + py*py + pz*pz)
                        // It is not clear what is the best thing to do here.
                        // For now, revert to Pt scaling.
                        return jet*ptFactor;
                    else
                        return subtracted;
                }
            }
            else
                return jet*ptFactor;
        }
        else
            return jet;
    }
}
