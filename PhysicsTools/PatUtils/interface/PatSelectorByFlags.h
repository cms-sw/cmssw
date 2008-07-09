#ifndef PhysicsTools_PatUtils_interface_PatSelectorByFlags_h
#define PhysicsTools_PatUtils_interface_PatSelectorByFlags_h

#include "DataFormats/PatCandidates/interface/Flags.h"

namespace pat {
    class SelectorByFlags {
        public:
            SelectorByFlags() : mask_(0) { }
            SelectorByFlags(uint32_t    maskToTest) : mask_(~maskToTest) {}
            SelectorByFlags(const std::string &bitToTest)  ;
            SelectorByFlags(const std::vector<std::string> bitsToTest) ;
            bool operator()(const reco::Particle &p) const { return pat::Flags::test(p, mask_); }
            bool operator()(const reco::Particle *p) const { return (p == 0 ? false : pat::Flags::test(*p, mask_)); }
        private:
            uint32_t mask_;
    };
}
#endif
