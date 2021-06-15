#include "GeneratorInterface/LHEInterface/interface/lheh5.h"

namespace lheh5 {

  inline std::ostream& operator<<(std::ostream& os, Particle const& p) {
    os << "\tpx: " << p.px << " py: " << p.py << " pz: " << p.pz << " e: " << p.e << "\n";
    return os;
  }

  inline std::ostream& operator<<(std::ostream& os, EventHeader const& eh) {
    os << "\tnparticles: " << eh.nparticles << " procid: " << eh.pid << " weight: " << eh.weight
       << " trials: " << eh.trials << "\n";
    os << "\tscale: " << eh.scale << " rscale: " << eh.rscale << " fscale: " << eh.fscale << " aqed: " << eh.aqed
       << " aqcd: " << eh.aqcd << "\n";
    os << "\tnpLO: " << eh.npLO << " npNLO: " << eh.npNLO << "\n";
    return os;
  }

  Particle Events::mkParticle(size_t idx) const {
    return {_vid[idx],
            _vstatus[idx],
            _vmother1[idx],
            _vmother2[idx],
            _vcolor1[idx],
            _vcolor2[idx],
            _vpx[idx],
            _vpy[idx],
            _vpz[idx],
            _ve[idx],
            _vm[idx],
            _vlifetime[idx],
            _vspin[idx]};
  }

  std::vector<Particle> Events::mkEvent(size_t ievent) const {
    std::vector<Particle> _E;
    // NOTE we need to subtract the particle offset here as the
    // particle properties run from 0 and not the global index when using batches
    for (size_t id = _vstart[ievent] - _particle_offset; id < _vend[ievent] - _particle_offset; ++id) {
      _E.push_back(mkParticle(id));
    }

    // Make sure beam particles are ordered according to convention i.e. first particle has positive z-momentum
    if (_E[0].pz < 0) {
      std::swap<Particle>(_E[0], _E[1]);
    }

    return _E;
  }

  EventHeader Events::mkEventHeader(int iev) const {
    return {
        _vnparticles[iev],
        _vpid[iev],
        _vweight[iev],
        _vtrials[iev],
        _vscale[iev],
        _vrscale[iev],
        _vfscale[iev],
        _vaqed[iev],
        _vaqcd[iev],
        _vnpLO[iev],
        _vnpNLO[iev],
    };
  }

  // Read function, returns an Events struct
  Events readEvents(HighFive::Group& g_index,
                    HighFive::Group& g_particle,
                    HighFive::Group& g_event,
                    size_t first_event,
                    size_t n_events) {
    // Lookup
    std::vector<size_t> _vstart, _vend;
    // Particles
    std::vector<int> _vid, _vstatus, _vmother1, _vmother2, _vcolor1, _vcolor2;
    std::vector<double> _vpx, _vpy, _vpz, _ve, _vm, _vlifetime, _vspin;
    // Event info
    std::vector<int> _vnparticles, _vpid, _vnpLO, _vnpNLO;
    std::vector<size_t> _vtrials;
    std::vector<double> _vweight, _vscale, _vrscale, _vfscale, _vaqed, _vaqcd;

    //double starttime, endtime;
    //starttime = MPI_Wtime();
    // Lookup
    HighFive::DataSet _start = g_index.getDataSet("start");
    HighFive::DataSet _end = g_index.getDataSet("end");
    // Particles
    HighFive::DataSet _id = g_particle.getDataSet("id");
    HighFive::DataSet _status = g_particle.getDataSet("status");
    HighFive::DataSet _mother1 = g_particle.getDataSet("mother1");
    HighFive::DataSet _mother2 = g_particle.getDataSet("mother2");
    HighFive::DataSet _color1 = g_particle.getDataSet("color1");
    HighFive::DataSet _color2 = g_particle.getDataSet("color2");
    HighFive::DataSet _px = g_particle.getDataSet("px");
    HighFive::DataSet _py = g_particle.getDataSet("py");
    HighFive::DataSet _pz = g_particle.getDataSet("pz");
    HighFive::DataSet _e = g_particle.getDataSet("e");
    HighFive::DataSet _m = g_particle.getDataSet("m");
    HighFive::DataSet _lifetime = g_particle.getDataSet("lifetime");
    HighFive::DataSet _spin = g_particle.getDataSet("spin");
    // Event info
    HighFive::DataSet _nparticles = g_event.getDataSet("nparticles");
    HighFive::DataSet _pid = g_event.getDataSet("pid");
    HighFive::DataSet _weight = g_event.getDataSet("weight");
    HighFive::DataSet _trials = g_event.getDataSet("trials");
    HighFive::DataSet _scale = g_event.getDataSet("scale");
    HighFive::DataSet _rscale = g_event.getDataSet("rscale");
    HighFive::DataSet _fscale = g_event.getDataSet("fscale");
    HighFive::DataSet _aqed = g_event.getDataSet("aqed");
    HighFive::DataSet _aqcd = g_event.getDataSet("aqcd");
    HighFive::DataSet _npLO = g_event.getDataSet("npLO");
    HighFive::DataSet _npNLO = g_event.getDataSet("npNLO");

    //endtime   = MPI_Wtime();
    //printf("DS took %f seconds\n", endtime-starttime);
    std::vector<size_t> offset_e = {first_event};
    std::vector<size_t> readsize_e = {n_events};
    //_vstart.reserve(n_events);
    //_vend.reserve(n_events);

    _start.select(offset_e, readsize_e).read(_vstart);
    _end.select(offset_e, readsize_e).read(_vend);
    std::vector<size_t> offset_p = {_vstart.front()};
    std::vector<size_t> readsize_p = {_vend.back() - _vstart.front()};

    int RESP = _vend.back() - _vstart.front();
    _vid.reserve(RESP);
    _vstatus.reserve(RESP);
    _vmother1.reserve(RESP);
    _vmother2.reserve(RESP);
    _vcolor1.reserve(RESP);
    _vcolor2.reserve(RESP);
    _vpx.reserve(RESP);
    _vpy.reserve(RESP);
    _vpz.reserve(RESP);
    _ve.reserve(RESP);
    _vm.reserve(RESP);
    _vlifetime.reserve(RESP);
    _vspin.reserve(RESP);

    _vnparticles.reserve(n_events);
    _vpid.reserve(n_events);
    _vweight.reserve(n_events);
    _vtrials.reserve(n_events);
    _vscale.reserve(n_events);
    _vrscale.reserve(n_events);
    _vfscale.reserve(n_events);
    _vaqed.reserve(n_events);
    _vaqcd.reserve(n_events);
    _vnpLO.reserve(n_events);
    _vnpNLO.reserve(n_events);

    //starttime = MPI_Wtime();
    // This is using HighFive's read
    _id.select(offset_p, readsize_p).read(_vid);
    _status.select(offset_p, readsize_p).read(_vstatus);
    _mother1.select(offset_p, readsize_p).read(_vmother1);
    _mother2.select(offset_p, readsize_p).read(_vmother2);
    _color1.select(offset_p, readsize_p).read(_vcolor1);
    _color2.select(offset_p, readsize_p).read(_vcolor2);
    _px.select(offset_p, readsize_p).read(_vpx);
    _py.select(offset_p, readsize_p).read(_vpy);
    _pz.select(offset_p, readsize_p).read(_vpz);
    _e.select(offset_p, readsize_p).read(_ve);
    _m.select(offset_p, readsize_p).read(_vm);
    _lifetime.select(offset_p, readsize_p).read(_vlifetime);
    _spin.select(offset_p, readsize_p).read(_vspin);

    //endtime   = MPI_Wtime();
    //printf("SELP took %f seconds\n", endtime-starttime);
    //starttime = MPI_Wtime();
    _nparticles.select(offset_e, readsize_e).read(_vnparticles);
    _pid.select(offset_e, readsize_e).read(_vpid);
    _weight.select(offset_e, readsize_e).read(_vweight);
    _trials.select(offset_e, readsize_e).read(_vtrials);
    _scale.select(offset_e, readsize_e).read(_vscale);
    _rscale.select(offset_e, readsize_e).read(_vrscale);
    _fscale.select(offset_e, readsize_e).read(_vfscale);
    _aqed.select(offset_e, readsize_e).read(_vaqed);
    _aqcd.select(offset_e, readsize_e).read(_vaqcd);
    _npLO.select(offset_e, readsize_e).read(_vnpLO);
    _npNLO.select(offset_e, readsize_e).read(_vnpNLO);
    //endtime   = MPI_Wtime();
    //printf("SELE took %f seconds\n", endtime-starttime);

    return {
        std::move(_vstart),      std::move(_vend),    std::move(_vid),     std::move(_vstatus),   std::move(_vmother1),
        std::move(_vmother2),    std::move(_vcolor1), std::move(_vcolor2), std::move(_vpx),       std::move(_vpy),
        std::move(_vpz),         std::move(_ve),      std::move(_vm),      std::move(_vlifetime), std::move(_vspin),
        std::move(_vnparticles), std::move(_vpid),    std::move(_vweight), std::move(_vtrials),   std::move(_vscale),
        std::move(_vrscale),     std::move(_vfscale), std::move(_vaqed),   std::move(_vaqcd),     std::move(_vnpLO),
        std::move(_vnpNLO),      offset_p[0],
    };
  }

  Particle Events2::mkParticle(size_t idx) const {
    return {_vid[idx],
            _vstatus[idx],
            _vmother1[idx],
            _vmother2[idx],
            _vcolor1[idx],
            _vcolor2[idx],
            _vpx[idx],
            _vpy[idx],
            _vpz[idx],
            _ve[idx],
            _vm[idx],
            _vlifetime[idx],
            _vspin[idx]};
  }

  std::vector<Particle> Events2::mkEvent(size_t ievent) const {
    std::vector<Particle> _E;
    // NOTE we need to subtract the particle offset here as the
    // particle properties run from 0 and not the global index when using batches
    size_t partno = _vstart[ievent] - _particle_offset;
    for (int id = 0; id < _vnparticles[ievent]; ++id) {
      _E.push_back(mkParticle(partno));
      partno++;
    }

    // Make sure beam particles are ordered according to convention i.e. first particle has positive z-momentum
    if (_E[0].pz < 0) {
      std::swap<Particle>(_E[0], _E[1]);
    }

    return _E;
  }

  EventHeader Events2::mkEventHeader(int iev) const {
    return {
        _vnparticles[iev],
        _vpid[iev],
        _vweight[iev],
        _vtrials[iev],
        _vscale[iev],
        _vrscale[iev],
        _vfscale[iev],
        _vaqed[iev],
        _vaqcd[iev],
        npLO,
        npNLO,
    };
  }

  // Read function, returns an Events struct --- this is for the new structure
  Events2 readEvents(
      HighFive::Group& g_particle, HighFive::Group& g_event, size_t first_event, size_t n_events, int npLO, int npNLO) {
    // Lookup
    std::vector<size_t> _vstart;
    // Particles
    std::vector<int> _vid, _vstatus, _vmother1, _vmother2, _vcolor1, _vcolor2;
    std::vector<double> _vpx, _vpy, _vpz, _ve, _vm, _vlifetime, _vspin;
    // Event info
    std::vector<int> _vnparticles, _vpid;
    std::vector<size_t> _vtrials;
    std::vector<double> _vweight, _vscale, _vrscale, _vfscale, _vaqed, _vaqcd;

    //double starttime, endtime;
    // Lookup
    HighFive::DataSet _start = g_event.getDataSet("start");
    // Particles
    HighFive::DataSet _id = g_particle.getDataSet("id");
    HighFive::DataSet _status = g_particle.getDataSet("status");
    HighFive::DataSet _mother1 = g_particle.getDataSet("mother1");
    HighFive::DataSet _mother2 = g_particle.getDataSet("mother2");
    HighFive::DataSet _color1 = g_particle.getDataSet("color1");
    HighFive::DataSet _color2 = g_particle.getDataSet("color2");
    HighFive::DataSet _px = g_particle.getDataSet("px");
    HighFive::DataSet _py = g_particle.getDataSet("py");
    HighFive::DataSet _pz = g_particle.getDataSet("pz");
    HighFive::DataSet _e = g_particle.getDataSet("e");
    HighFive::DataSet _m = g_particle.getDataSet("m");
    HighFive::DataSet _lifetime = g_particle.getDataSet("lifetime");
    HighFive::DataSet _spin = g_particle.getDataSet("spin");
    // Event info
    HighFive::DataSet _nparticles = g_event.getDataSet("nparticles");
    HighFive::DataSet _pid = g_event.getDataSet("pid");
    HighFive::DataSet _weight = g_event.getDataSet("weight");
    HighFive::DataSet _trials = g_event.getDataSet("trials");
    HighFive::DataSet _scale = g_event.getDataSet("scale");
    HighFive::DataSet _rscale = g_event.getDataSet("rscale");
    HighFive::DataSet _fscale = g_event.getDataSet("fscale");
    HighFive::DataSet _aqed = g_event.getDataSet("aqed");
    HighFive::DataSet _aqcd = g_event.getDataSet("aqcd");

    std::vector<size_t> offset_e = {first_event};
    std::vector<size_t> readsize_e = {n_events};

    // We now know the first event to read
    _start.select(offset_e, readsize_e).read(_vstart);

    // That's the first particle
    std::vector<size_t> offset_p = {_vstart.front()};
    // The last particle is last entry in start + nparticles of that event
    _vnparticles.reserve(n_events);
    _nparticles.select(offset_e, readsize_e).read(_vnparticles);

    size_t RESP = _vstart.back() - _vstart.front() + _vnparticles.back();
    std::vector<size_t> readsize_p = {RESP};

    _vid.reserve(RESP);
    _vstatus.reserve(RESP);
    _vmother1.reserve(RESP);
    _vmother2.reserve(RESP);
    _vcolor1.reserve(RESP);
    _vcolor2.reserve(RESP);
    _vpx.reserve(RESP);
    _vpy.reserve(RESP);
    _vpz.reserve(RESP);
    _ve.reserve(RESP);
    _vm.reserve(RESP);
    _vlifetime.reserve(RESP);
    _vspin.reserve(RESP);

    _vpid.reserve(n_events);
    _vweight.reserve(n_events);
    _vtrials.reserve(n_events);
    _vscale.reserve(n_events);
    _vrscale.reserve(n_events);
    _vfscale.reserve(n_events);
    _vaqed.reserve(n_events);
    _vaqcd.reserve(n_events);

    // This is using HighFive's read
    _id.select(offset_p, readsize_p).read(_vid);
    _status.select(offset_p, readsize_p).read(_vstatus);
    _mother1.select(offset_p, readsize_p).read(_vmother1);
    _mother2.select(offset_p, readsize_p).read(_vmother2);
    _color1.select(offset_p, readsize_p).read(_vcolor1);
    _color2.select(offset_p, readsize_p).read(_vcolor2);
    _px.select(offset_p, readsize_p).read(_vpx);
    _py.select(offset_p, readsize_p).read(_vpy);
    _pz.select(offset_p, readsize_p).read(_vpz);
    _e.select(offset_p, readsize_p).read(_ve);
    _m.select(offset_p, readsize_p).read(_vm);
    _lifetime.select(offset_p, readsize_p).read(_vlifetime);
    _spin.select(offset_p, readsize_p).read(_vspin);

    _pid.select(offset_e, readsize_e).read(_vpid);
    _weight.select(offset_e, readsize_e).read(_vweight);
    _trials.select(offset_e, readsize_e).read(_vtrials);
    _scale.select(offset_e, readsize_e).read(_vscale);
    _rscale.select(offset_e, readsize_e).read(_vrscale);
    _fscale.select(offset_e, readsize_e).read(_vfscale);
    _aqed.select(offset_e, readsize_e).read(_vaqed);
    _aqcd.select(offset_e, readsize_e).read(_vaqcd);

    return {
        std::move(_vstart),
        std::move(_vid),
        std::move(_vstatus),
        std::move(_vmother1),
        std::move(_vmother2),
        std::move(_vcolor1),
        std::move(_vcolor2),
        std::move(_vpx),
        std::move(_vpy),
        std::move(_vpz),
        std::move(_ve),
        std::move(_vm),
        std::move(_vlifetime),
        std::move(_vspin),
        std::move(_vnparticles),
        std::move(_vpid),
        std::move(_vweight),
        std::move(_vtrials),
        std::move(_vscale),
        std::move(_vrscale),
        std::move(_vfscale),
        std::move(_vaqed),
        std::move(_vaqcd),
        npLO,
        npNLO,
        offset_p[0],
    };
  }

}  // namespace lheh5
