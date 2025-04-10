// -*- C++ -*-
//
// HepMCTraits.h is a part of ThePEG - Toolkit for HEP Event Generation
// Copyright (C) 1999-2019 Leif Lonnblad
//
// ThePEG is licenced under version 3 of the GPL, see COPYING for details.
// Please respect the MCnet academic guidelines, see GUIDELINES for details.
//
#ifndef ThePEG_HepMCTraits_H
#define ThePEG_HepMCTraits_H

#include "HepMC3/GenEvent.h"
namespace HepMC3 {
  class GenParticle;
  class GenVertex;
  class GenPdfInfo;
}  // namespace HepMC3

namespace ThePEG {

  /**
   * HepMCTraitsBase is a convenient base class for specializing the
   * HepMCTraits class to deal with different flavours of HepMC in the
   * HepMCConverter class. The default version will work for the CLHEP
   * implementation of HepMC. To use the HepMCConverter class for any
   * flavour of HepMC you have to specialize the HepMCTraits class
   * accordingly, possibly inheriting the functionality from the
   * HepMCTraitsBase class and only overriding the functions and
   * typedefs which are different. For the CLHEP flavour of HepMC you
   * only need to do <code>template&lt;&gt; struct
   * HepMCTraits&lt;HepMC::GenEvent&gt;: public
   * HepMCTraitsBase&lt;HepMC::GenEvent,HepMC::GenParticle,HepMC::GenVertex,
   * HepMC::Polarization&gt; {};</code> somewhere inside the ThePEG
   * namespace.  The boolean template argument determines whether the
   * HepMC implementation is specifying units or not.
   */
  template <typename HepMCEventT,
            typename HepMCParticleT,
            typename HepMCParticlePtrT,
            typename HepMCVertexT,
            typename HepMCVertexPtrT,
            typename HepMCPolarizationT,
            typename HepMCPdfInfoT>

  struct HepMCTraitsBase {
    /** Typedef of the particle class. */
    typedef HepMCParticleT ParticleT;

    /** Typedef of the event class. */
    typedef HepMCEventT EventT;

    /** Typedef of the vertex class. */
    typedef HepMCVertexT VertexT;

    /** Typedef of the polarization class. */
    typedef HepMCPolarizationT PolarizationT;

    /** Typedef of the PdfInfo class. */
    typedef HepMCPdfInfoT PdfInfoT;

    /** Typedef of a particle pointer */
    typedef HepMCParticlePtrT ParticlePtrT;

    /** Typedef of a vertex pointer */
    typedef HepMCVertexPtrT VertexPtrT;

    /** Create an event object with number \a evno and \a weight. */
    static EventT* newEvent(long evno, double weight, const map<string, double>& optionalWeights) {
      EventT* e = new EventT();
      e->set_event_number(evno);
      std::vector<std::string> wnames;
      std::vector<double> wvalues;

      wnames.push_back("Default");
      wvalues.push_back(weight);
      for (map<string, double>::const_iterator w = optionalWeights.begin(); w != optionalWeights.end(); ++w) {
        wnames.push_back(w->first);
        wvalues.push_back(w->second);
      }
      e->run_info()->set_weight_names(wnames);
      e->weights() = wvalues;

#ifdef HEPMC_HAS_NAMED_WEIGHTS
      for (size_t i = 0; i < wnames.size(); i++)
        e->weights()[wnames[i]] = wvalues[i];
#else
      e->weights() = wvalues;
#endif

      return e;
    }

    /** Reset event weight and number of a re-used GenEvent. */
    static void resetEvent(EventT* e, long evno, double weight, const map<string, double>& optionalWeights) {
      e->set_event_number(evno);
      e->weights().clear();
      std::vector<std::string> wnames;
      std::vector<double> wvalues;

      wnames.push_back("Default");
      wvalues.push_back(weight);
      for (map<string, double>::const_iterator w = optionalWeights.begin(); w != optionalWeights.end(); ++w) {
        wnames.push_back(w->first);
        wvalues.push_back(w->second);
      }
      e->run_info()->set_weight_names(wnames);
      e->weights() = wvalues;

#ifdef HEPMC_HAS_NAMED_WEIGHTS
      for (size_t i = 0; i < wnames.size(); i++)
        e->weights()[wnames[i]] = wvalues[i];
#else
      e->weights() = wvalues;
#endif
    }

    /**
     * Return true if this version of HepMC accept user-defined units.
     */
    static bool hasUnits() {
#ifdef HEPMC_HAS_UNITS
      return true;
#else
      return false;
#endif
    }

    /**
     * Return the energy unit used in the installed version of HepMC.
     */
    static Energy defaultEnergyUnit() {
#ifndef HEPMC_HAS_UNITS
      return GeV;
#else
      return HepMC3::Units::default_momentum_unit() == HepMC3::Units::GEV ? GeV : MeV;
#endif
    }

    /**
     * Return the length unit used in the installed version of HepMC.
     */
    static Length defaultLengthUnit() {
#ifndef HEPMC_HAS_UNITS
      return millimeter;
#else
      return HepMC3::Units::default_length_unit() == HepMC3::Units::MM ? millimeter : 10.0 * millimeter;
#endif
    }

    /**
     * Return the momentum unit used by a given GenEvent object. If
     * HepMC does not support units this must return GeV.
     */
    static Energy momentumUnit(const EventT& e) {
#ifdef HEPMC_HAS_UNITS
      return e.momentum_unit() == HepMC3::Units::MEV ? MeV : GeV;
#else
      return GeV;
#endif
    }

    /**
     * Return the length unit used by a given GenEvent object. If
     * HepMC does not support units this must return millimeter.
     */
    static Length lengthUnit(const EventT& e) {
#ifdef HEPMC_HAS_UNITS
      return e.length_unit() == HepMC3::Units::CM ? centimeter : millimeter;
#else
      return millimeter;
#endif
    }

    /**
     * Set the units to be used by the given GenEvent object. If
     * HepMC does not support units this should be a no-op.
     */
#ifdef HEPMC_HAS_UNITS
    static void setUnits(EventT& e, Energy momu, Length lenu) {
      e.use_units(momu == MeV ? HepMC3::Units::MEV : HepMC3::Units::GEV,
                  lenu == centimeter ? HepMC3::Units::CM : HepMC3::Units::MM);
    }
#else
    static void setUnits(EventT&, Energy, Length) {}
#endif

    /** Set the \a scale, \f$\alpha_S\f$ (\a aS) and \f$\alpha_{EM}\f$
     (\a aEM) for the event \a e. The scale will be scaled with \a
     unit before given to the GenEvent. */
    static void setScaleAndAlphas(EventT& e, Energy2 scale, double aS, double aEM, Energy unit) {
      e.set_event_scale(sqrt(scale) / unit);
      e.set_alphaQCD(aS);
      e.set_alphaQED(aEM);
    }

    /** Set the primary vertex, \a v, for the event \a e. */
    static void setSignalProcessVertex(EventT& e, VertexPtrT v) { e.set_signal_process_vertex(v); }

    /** Set a vertex, \a v, for the event \a e. */
    static void addVertex(EventT& e, VertexPtrT v) { e.add_vertex(v); }

    /** Create a new particle object with momentum \a p, PDG number \a
     id and status code \a status. The momentum will be scaled with
     \a unit which according to the HepMC documentation should be
     GeV. */
    static ParticlePtrT newParticle(const Lorentz5Momentum& p, long id, int status, Energy unit) {
      // Note that according to the documentation the momentum is stored in a
      // HepLorentzVector in GeV (event though the CLHEP standard is MeV).
      LorentzVector<double> p_scalar = p / unit;
      ParticlePtrT genp = new ParticleT(p_scalar, id, status);
      genp->setGeneratedMass(p.mass() / unit);
      return genp;
    }

    /** Set the polarization directions, \a the and \a phi, for particle
     \a p. */
    static void setPolarization(ParticleT& genp, double the, double phi) {
      genp.set_polarization(PolarizationT(the, phi));
    }

    /** Set the colour line (with index \a indx) to \a coline for
     particle \a p. */
    static void setColourLine(ParticleT& p, int indx, int coline) { p.set_flow(indx, coline); }

    /** Create a new vertex. */
    static VertexPtrT newVertex() { return new VertexT(); }

    /** Add an incoming particle, \a p, to the vertex, \a v. */
    static void addIncoming(VertexT& v, ParticlePtrT p) { v.add_particle_in(p); }

    /** Add an outgoing particle, \a p, to the vertex, \a v. */
    static void addOutgoing(VertexT& v, ParticlePtrT p) { v.add_particle_out(p); }

    /** Set the position \a p for the vertex, \a v. The length will be
     scaled with \a unit which normally should be millimeters. */
    static void setPosition(VertexT& v, const LorentzPoint& p, Length unit) {
      LorentzVector<double> p_scaled = p / unit;
      v.set_position(p_scaled);
    }

    /** Set the beam particles for the event.*/
    static void setBeamParticles(EventT& e, ParticlePtrT p1, ParticlePtrT p2) {
      e.set_beam_particles(p1, p2);
      p1->set_status(4);
      p2->set_status(4);
    }

    /** Set the PDF info for the event. */
#ifdef HEPMC_HAS_PDF_INFO
    static void setPdfInfo(EventT& e, int id1, int id2, double x1, double x2, double scale, double xf1, double xf2) {
      HepMC::GenPdfInfoPtr pdfinfo = std::make_shared<HepMC::GenPdfInfo>();
      pdfinfo->set(id1, id2, x1, x2, scale, xf1, xf2);
      e.set_pdf_info(pdfinfo);
    }
#else
    static void setPdfInfo(EventT&, int, int, double, double, double, double, double) {}
#endif

    /** Set the cross section info for the event. */
#ifdef HEPMC_HAS_CROSS_SECTION
    static void setCrossSection(EventT& ev, double xs, double xserr) {
      std::shared_ptr<HepMC::GenCrossSection> x = std::make_shared<HepMC::GenCrossSection>();
      x->set_cross_section(xs, xserr);
      ev.set_cross_section(x);
    }
#else
    static void setCrossSection(EventT&, double, double) {}
#endif
  };

  /**
   * The HepMCTraits class is used to deal with different flavours of
   * HepMC in the HepMCConverter class. To use the HepMCConverter class
   * for any flavour of HepMC you have to specialize the
   * HepMCTraits class accordingly, possibly inheriting the
   * functionality from the HepMCTraitsBase class and only overriding
   * the functions and typedefs which are different. For the CLHEP
   * flavour of HepMC you only need to do <code>template&lt;&gt; struct
   * HepMCTraits&lt;HepMC::GenEvent&gt;: public
   * HepMCTraitsBase&lt;HepMC::GenEvent,HepMC::GenParticle,HepMC::GenVertex,
   * HepMC::Polarization,HepMC::PdfInfo&gt; {};</code> somewhere inside the ThePEG
   * namespace.
   */
  template <typename HepMCEventT>
  struct HepMCTraits {};

}  // namespace ThePEG

#endif
