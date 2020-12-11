// lheh5.h
#ifndef GeneratorInterface_LHEInterface_LHEH5_H
#define GeneratorInterface_LHEInterface_LHEH5_H

#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "highfive/H5File.hpp"
#include "highfive/H5DataSet.hpp"

namespace lheh5 {

  struct Particle {
    int id, status, mother1, mother2, color1, color2;
    double px, py, pz, e, m, lifetime, spin;
    // id        .. IDUP
    // color1/2  .. ICOLUP firt/second
    // mother1/2 .. MOTHUP first/second
    // status    .. ISTUP
    // px ... m  .. PUP[..]
    // lifetime  .. VTIMUP  (UP ... user process)
    // spin      .. SPINUP
  };

  struct EventHeader {
    // Event info
    int nparticles;  // corr to NUP
    int pid;         // this is all LHAu-::setProcess
    double weight;
    size_t trials;
    double scale;
    double rscale;
    double fscale;
    double aqed;
    double aqcd;
    int npLO;
    int npNLO;
  };

  struct Events {
    // Lookup
    std::vector<size_t> _vstart;
    std::vector<size_t> _vend;
    // Particles
    std::vector<int> _vid;
    std::vector<int> _vstatus;
    std::vector<int> _vmother1;
    std::vector<int> _vmother2;
    std::vector<int> _vcolor1;
    std::vector<int> _vcolor2;
    std::vector<double> _vpx;
    std::vector<double> _vpy;
    std::vector<double> _vpz;
    std::vector<double> _ve;
    std::vector<double> _vm;
    std::vector<double> _vlifetime;
    std::vector<double> _vspin;
    // Event info
    std::vector<int> _vnparticles;
    std::vector<int> _vpid;
    std::vector<double> _vweight;
    std::vector<size_t> _vtrials;
    std::vector<double> _vscale;
    std::vector<double> _vrscale;
    std::vector<double> _vfscale;
    std::vector<double> _vaqed;
    std::vector<double> _vaqcd;
    std::vector<int> _vnpLO;
    std::vector<int> _vnpNLO;
    size_t _particle_offset;

    Particle mkParticle(size_t idx) const;
    std::vector<Particle> mkEvent(size_t ievent) const;
    EventHeader mkEventHeader(int ievent) const;
  };

  struct Events2 {
    // Lookup
    std::vector<size_t> _vstart;
    // Particles
    std::vector<int> _vid;
    std::vector<int> _vstatus;
    std::vector<int> _vmother1;
    std::vector<int> _vmother2;
    std::vector<int> _vcolor1;
    std::vector<int> _vcolor2;
    std::vector<double> _vpx;
    std::vector<double> _vpy;
    std::vector<double> _vpz;
    std::vector<double> _ve;
    std::vector<double> _vm;
    std::vector<double> _vlifetime;
    std::vector<double> _vspin;
    // Event info
    std::vector<int> _vnparticles;
    std::vector<int> _vpid;
    std::vector<double> _vweight;
    std::vector<size_t> _vtrials;
    std::vector<double> _vscale;
    std::vector<double> _vrscale;
    std::vector<double> _vfscale;
    std::vector<double> _vaqed;
    std::vector<double> _vaqcd;
    int npLO;
    int npNLO;
    size_t _particle_offset;

    Particle mkParticle(size_t idx) const;
    std::vector<Particle> mkEvent(size_t ievent) const;
    EventHeader mkEventHeader(int ievent) const;
  };

  Events readEvents(HighFive::Group& g_index,
                    HighFive::Group& g_particle,
                    HighFive::Group& g_event,
                    size_t first_event,
                    size_t n_events);
  Events2 readEvents(
      HighFive::Group& g_particle, HighFive::Group& g_event, size_t first_event, size_t n_events, int npLO, int npNLO);
  std::ostream& operator<<(std::ostream& os, Particle const& p);
  std::ostream& operator<<(std::ostream& os, EventHeader const& eh);
}  // namespace lheh5

#endif
