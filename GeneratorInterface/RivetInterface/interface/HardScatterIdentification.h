#ifndef GeneratorInterface_RivetInterface_HardScatterIdentification_h
#define GeneratorInterface_RivetInterface_HardScatterIdentification_h

#include "Rivet/Particle.hh"
#include "Rivet/Tools/ParticleUtils.hh"

namespace HepMC3 {
  class GenEvent;
}

namespace Rivet {
  namespace HTXSUtils {

    inline bool hasChildWithPDG(const ConstGenParticlePtr &ptcl, int pdgID) {
      if (!ptcl || !ptcl->end_vertex())
        return false;
      for (const auto &child : Particle(*ptcl).children()) {
        if (child.pid() == pdgID)
          return true;
      }
      return false;
    }

    inline bool hasParentWithPDG(const ConstGenParticlePtr &ptcl, int pdgID) {
      if (!ptcl)
        return false;
      const ConstGenVertexPtr prodVtx = ptcl->production_vertex();
      if (!prodVtx)
        return false;
      for (const auto &parent : HepMCUtils::particles(prodVtx, Relatives::PARENTS)) {
        if (parent->pdg_id() == pdgID)
          return true;
      }
      return false;
    }

    struct HardScatterResult {
      ConstGenVertexPtr vertex = nullptr;
      ConstGenParticlePtr higgs = nullptr;
      unsigned int higgsCount = 0;
    };

    inline HardScatterResult identifyHardScatter(const HepMC3::GenEvent *genEvent) {
      HardScatterResult result;
      if (!genEvent)
        return result;

      for (const ConstGenParticlePtr &ptcl : HepMCUtils::particles(genEvent)) {
        // a) Reject all non-Higgs particles
        if (!PID::isHiggs(ptcl->pdg_id()))
          continue;
        // b) select only the final Higgs boson copy, prior to decay
        if (ptcl->end_vertex() && !hasChildWithPDG(ptcl, PID::HIGGS)) {
          result.higgs = ptcl;
          ++result.higgsCount;
        }
        // c) HepMC3 does not provide signal_process_vertex anymore
        //    set hard-scatter vertex based on first Hiigs boson
        if (!result.vertex && ptcl->production_vertex() && !hasParentWithPDG(ptcl, PID::HIGGS))
          result.vertex = ptcl->production_vertex();
      }

      return result;
    }

  }  // namespace HTXSUtils
}  // namespace Rivet

#endif
