// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include <unordered_set>

#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthHelper.h"
#include "PhysicsTools/TruthInfo/interface/GenGraphBuild.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"

namespace truth {

  GenBuild buildFromHepMC2(HepMC::GenEvent const& ev) {
    GenBuild gb;

    std::unordered_set<int> seenV;
    std::unordered_set<int> seenP;

    gb.particlePdgIdByBarcode.reserve(ev.particles_size() * 2);
    gb.particleStatusByBarcode.reserve(ev.particles_size() * 2);
    gb.particleBarcodeByIndex.reserve(ev.particles_size());

    for (auto v = ev.vertices_begin(); v != ev.vertices_end(); ++v) {
      if (*v == nullptr)
        continue;

      const int vbc = (*v)->barcode();

      if (seenV.insert(vbc).second)
        gb.vtxBarcodes.push_back(vbc);

      for (auto po = (*v)->particles_out_const_begin(); po != (*v)->particles_out_const_end(); ++po) {
        if (*po == nullptr)
          continue;

        const int pbc = (*po)->barcode();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.vtxToPart.emplace_back(vbc, pbc);
      }

      for (auto pi = (*v)->particles_in_const_begin(); pi != (*v)->particles_in_const_end(); ++pi) {
        if (*pi == nullptr)
          continue;

        const int pbc = (*pi)->barcode();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.partToVtx.emplace_back(pbc, vbc);
      }
    }

    MCTruthHelper<HepMC::GenParticle> mcTruthHelper;
    for (auto p = ev.particles_begin(); p != ev.particles_end(); ++p) {
      if (*p == nullptr)
        continue;

      const int pbc = (*p)->barcode();

      gb.particleBarcodeByIndex.push_back(pbc);
      gb.particlePdgIdByBarcode.emplace(pbc, (*p)->pdg_id());
      gb.particleStatusByBarcode.emplace(pbc, static_cast<int16_t>((*p)->status()));

      reco::GenStatusFlags flags;
      mcTruthHelper.fillGenStatusFlags(**p, flags);
      gb.particleStatusFlagsByBarcode.emplace(pbc, static_cast<uint16_t>(flags.flags_.to_ulong()));

      if (seenP.insert(pbc).second)
        gb.partBarcodes.push_back(pbc);
    }

    return gb;
  }

  GenBuild buildFromHepMC3(HepMC3::GenEvent const& ev) {
    GenBuild gb;

    std::unordered_set<int> seenV;
    std::unordered_set<int> seenP;

    gb.particlePdgIdByBarcode.reserve(ev.particles().size() * 2);
    gb.particleStatusByBarcode.reserve(ev.particles().size() * 2);
    gb.particleBarcodeByIndex.reserve(ev.particles().size());

    for (auto const& vptr : ev.vertices()) {
      if (!vptr)
        continue;

      const int vbc = vptr->id();

      if (seenV.insert(vbc).second)
        gb.vtxBarcodes.push_back(vbc);

      for (auto const& po : vptr->particles_out()) {
        if (!po)
          continue;

        const int pbc = po->id();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.vtxToPart.emplace_back(vbc, pbc);
      }

      for (auto const& pi : vptr->particles_in()) {
        if (!pi)
          continue;

        const int pbc = pi->id();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.partToVtx.emplace_back(pbc, vbc);
      }
    }

    for (auto const& pptr : ev.particles()) {
      if (!pptr)
        continue;

      const int pbc = pptr->id();

      gb.particleBarcodeByIndex.push_back(pbc);
      gb.particlePdgIdByBarcode.emplace(pbc, pptr->pid());
      gb.particleStatusByBarcode.emplace(pbc, static_cast<int16_t>(pptr->status()));

      if (seenP.insert(pbc).second)
        gb.partBarcodes.push_back(pbc);
    }

    return gb;
  }

}  // namespace truth
