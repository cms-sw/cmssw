#include "regionizer_base_ref.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

l1ct::RegionizerEmulator::~RegionizerEmulator() {}

void l1ct::RegionizerEmulator::run(const RegionizerDecodedInputs & in, std::vector<PFInputRegion> & out) const {
    for (const auto & sec : in.track) {
        for (const auto & tk : sec) {
            float glbEta  = sec.region.globalEta(tk.floatEta()), glbPhi = sec.region.globalPhi(tk.floatPhi());
            float glbEtaV = sec.region.globalEta(tk.floatVtxEta()), glbPhiV = sec.region.globalPhi(tk.floatVtxPhi());
            for (auto & r : out) {
                if (r.region.contains(glbEta,glbPhi) || r.region.contains(glbEtaV,glbPhiV)) {
                    r.track.push_back(tk);
                    r.track.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
                    r.track.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
                }
            }
        }
    }

    for (const auto & sec : in.hadcalo) {
        for (const auto & c : sec) {
            float glbEta  = sec.region.globalEta(c.floatEta()), glbPhi = sec.region.globalPhi(c.floatPhi());
            for (auto & r : out) {
                if (r.region.contains(glbEta,glbPhi)) {
                    r.hadcalo.push_back(c);
                    r.hadcalo.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
                    r.hadcalo.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
                }
            }
        }
    }

    for (const auto & sec : in.emcalo) {
        for (const auto & c : sec) {
            float glbEta  = sec.region.globalEta(c.floatEta()), glbPhi = sec.region.globalPhi(c.floatPhi());
            for (auto & r : out) {
                if (r.region.contains(glbEta,glbPhi)) {
                    r.emcalo.push_back(c);
                    r.emcalo.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
                    r.emcalo.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
                }
            }
        }
    }

    for (const auto & mu : in.muon) {
        float glbEta  = mu.floatEta(), glbPhi = mu.floatPhi();
        for (auto & r : out) {
            if (r.region.contains(glbEta,glbPhi)) {
                r.muon.push_back(mu);
                r.muon.back().hwEta = l1ct::Scales::makeEta(r.region.localEta(glbEta));
                r.muon.back().hwPhi = l1ct::Scales::makePhi(r.region.localPhi(glbPhi));
            }
        }
    }

}
