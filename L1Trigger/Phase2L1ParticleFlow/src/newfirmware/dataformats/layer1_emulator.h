#ifndef FIRMWARE_dataformats_layer1_emulator_h
#define FIRMWARE_dataformats_layer1_emulator_h

#include <fstream>
#include <vector>
#include "layer1_objs.h"
#include "pf.h"
#include "puppi.h"

namespace l1t { class PFTrack; class PFCluster; class PFCandidate; class Muon; }

namespace l1ct {
    
    struct HadCaloObjEmu : public HadCaloObj {
        const l1t::PFCluster *src;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };

    struct EmCaloObjEmu : public EmCaloObj { 
        const l1t::PFCluster *src;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };

    struct TkObjEmu : public TkObj {
        uint16_t hwChi2, hwStubs;
        const l1t::PFTrack *src;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };

    struct MuObjEmu : public MuObj {
        const l1t::Muon *src;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };

    struct PFChargedObjEmu : public PFChargedObj {
        const l1t::PFCluster *srcCluster;
        const l1t::PFTrack *srcTrack;
        const l1t::Muon *srcMu;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };

    struct PFNeutralObjEmu : public PFNeutralObj {
        const l1t::PFCluster *srcCluster;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };

    struct PuppiObjEmu : public PuppiObj {
        const l1t::PFCluster *srcCluster;
        const l1t::PFTrack *srcTrack;
        const l1t::Muon *srcMu;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };


    struct PFRegionEmu : public PFRegion {
        float etaCenter, etaMin, etaMax, phiCenter, phiHalfWidth;
        float etaExtra, phiExtra;

        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };
   
    struct PVObjEmu {
        z0_t hwZ0;
        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };

    struct PFInputRegion {
        PFRegionEmu region;
        std::vector<HadCaloObjEmu> hadcalo;
        std::vector<EmCaloObjEmu> emcalo;
        std::vector<TkObjEmu> track;
        std::vector<MuObjEmu> muon;

        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    }; 

    struct OutputRegion {
        std::vector<PFChargedObjEmu> pfcharged;
        std::vector<PFNeutralObjEmu> pfphoton;
        std::vector<PFNeutralObjEmu> pfneutral;
        std::vector<PFChargedObjEmu> pfmuon;
        std::vector<PuppiObjEmu> puppi;

        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };


    struct Event {
        uint32_t run, lumi; uint64_t event;
        std::vector<PFInputRegion> pfinputs;
        std::vector<PVObjEmu> pvs;
        std::vector<OutputRegion> out;

        Event() : run(0), lumi(0), event(0) {}

        bool read(std::fstream & from) ;
        bool write(std::fstream & to) const ;
    };
  
    template<typename T1, typename T2>
    void toFirmware(const std::vector<T1> &in, unsigned int NMAX, T2 out[/*NMAX*/]) {
        unsigned int n = std::min<unsigned>(in.size(), NMAX);
        for (unsigned int i = 0; i < n; ++i) out[i] = in[i];
        for (unsigned int i = n; i < NMAX; ++i) out[i].clear();
    }

} // namespace

#endif
