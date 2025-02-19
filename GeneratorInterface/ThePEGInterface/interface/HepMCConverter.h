// -*- C++ -*-
//
// HepMCConverter.h is a part of ThePEG - Toolkit for HEP Event Generation
// Copyright (C) 1999-2007 Leif Lonnblad
//
// ThePEG is licenced under version 2 of the GPL, see COPYING for details.
// Please respect the MCnet academic guidelines, see GUIDELINES for details.
//
#ifndef ThePEG_HepMCConverter_H
#define ThePEG_HepMCConverter_H
// This is the declaration of the HepMCConverter class.

#include <ThePEG/Config/ThePEG.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/Vectors/HepMCTraits.h>

namespace ThePEG {


/**
 * The HepMCConverter defines only one public static function which
 * converts a ThePEG::Event object to a
 * <code>HepMC::GenEvent</code>. All mother-daughter relationships and
 * colour information is preserved.
 *
 * @see Event
 * @see Particle
 *
 * @author Leif Lönnblad
 */
template <typename HepMCEventT, typename Traits = HepMCTraits<HepMCEventT> >
class HepMCConverter  {

public:

  /** @cond EXCEPTIONCLASSES */
  /** Exception class for HepMCConverter. */
  struct HepMCConverterException: public Exception {};
  /** @endcond */

  /**
   * Help class to represent a temporary vertex which can be
   * converted to a GenVertex.
   */
  struct Vertex {
    /** Particles going in to the vertex. */
    tcParticleSet in;
    /** Particles going out of the vertex. */
    tcParticleSet out;
  };

  /** Forward typedefs from Traits class. */
  typedef typename Traits::ParticleT GenParticle;
  /** Forward typedefs from Traits class. */
  typedef typename Traits::EventT GenEvent;
  /** Forward typedefs from Traits class. */
  typedef typename Traits::VertexT GenVertex;
  /** Forward typedefs from Traits class. */
  typedef typename Traits::PdfInfoT PdfInfo;
  /** Map ThePEG particles to HepMC particles. */
  typedef map<tcPPtr,GenParticle*> ParticleMap;
  /** Map ThePEG colour lines to HepMC colour indices. */
  typedef map<tcColinePtr,long> FlowMap;
  /** Map ThePEG particles to vertices. */
  typedef map<tcPPtr,Vertex*> VertexMap;
  /** Map vertices to GenVertex */
  typedef map<const Vertex *, GenVertex*> GenVertexMap;

public:

  /**
   * Convert a ThePEG::Event to a HepMC::GenEvent. The caller is
   * responsible for deleting the constructed GenEvent object. If \a
   * nocopies is true, only final copies of particles connected with
   * Particle::previous() and Particle::next() will be entered in the
   * HepMC::GenEvent. In the GenEvent object, the energy/momentum
   * variables will be in units of \a eunit and lengths variables in
   * units of \a lunit.
   */
  static GenEvent * convert(const Event & ev, bool nocopies = false,
			    Energy eunit = Traits::defaultEnergyUnit(),
			    Length lunit = Traits::defaultLengthUnit());

  /**
   * Convert a ThePEG::Event to a HepMC::GenEvent. The caller supplies
   * a GenEvent object, \a gev, which will be filled. If \a nocopies
   * is true, only final copies of particles connected with
   * Particle::previous() and Particle::next() will be entered in the
   * HepMC::GenEvent. In the GenEvent object, the energy/momentum
   * variables will be in units of \a eunit and lengths variables in
   * units of \a lunit.
   */
  static void
  convert(const Event & ev, GenEvent & gev, bool nocopies,
	  Energy eunit, Length lunit);

  /**
   * Convert a ThePEG::Event to a HepMC::GenEvent. The caller supplies
   * a GenEvent object, \a gev, which will be filled. If \a nocopies
   * is true, only final copies of particles connected with
   * Particle::previous() and Particle::next() will be entered in the
   * HepMC::GenEvent. In the GenEvent object, the energy/momentum
   * variables will be in units of \a eunit and lengths variables in
   * units of \a lunit.
   */
  static void
  convert(const Event & ev, GenEvent & gev, bool nocopies = false);

private:

  /**
   * The proper constructors are private. The class is only
   * instantiated within the convert method.
   */
  HepMCConverter(const Event & ev, bool nocopies, Energy eunit, Length lunit);

  /**
   * The proper constructors are private. The class is only
   * instantiated within the convert method.
   */
  HepMCConverter(const Event & ev, GenEvent & gev, bool nocopies,
		 Energy eunit, Length lunit);

  /**
   * Common init function used by the constructors.
   */
  void init(const Event & ev, bool nocopies);

  /**
   * Default constructor is unimplemented and private and should never be used.
   */
  HepMCConverter();

  /**
   * Copy constructor is unimplemented and private and should never be used.
   */
  HepMCConverter(const HepMCConverter &);

  /**
   * Assignment is unimplemented and private and should never be used.
   */
  HepMCConverter & operator=(const HepMCConverter &);

private:

  /**
   * Create a GenParticle from a ThePEG Particle.
   */
  GenParticle * createParticle(tcPPtr p) const;

  /**
   * Join the decay vertex of the parent with the decay vertex of the
   * child.
   */
  void join(tcPPtr parent, tcPPtr child);

  /**
   * Create a GenVertex from a temporary Vertex.
   */
  GenVertex * createVertex(Vertex * v); 

  /**
   * Create and set a PdfInfo object for the event
   */
  void setPdfInfo(const Event & e);

  /**
   * Sort vertices topologically
   */
  void sortTopologically(tcPVector & allv);
private:

  /**
   * The constructed GenEvent.
   */
  GenEvent * geneve;

  /**
   * The translation table between the ThePEG particles and the
   * GenParticles.
   */
  ParticleMap pmap;

  /**
   * The translation table between ThePEG ColourLine objects and HepMC
   * Flow indices.
   */
  FlowMap flowmap;

  /**
   * All temporary vertices created.
   */
  vector<Vertex> vertices;

  /**
   * The mapping of particles to their production vertices.
   */
  VertexMap prov;

  /**
   * The mapping of particles to their decy vertices.
   */
  VertexMap decv;

  /**
   * The mapping between temporary vertices and the created GenVertex Objects.
   */
  GenVertexMap vmap;

  /**
   * The energy unit to be used in the GenEvent.
   */
  Energy energyUnit;

  /**
   * The length unit to be used in the GenEvent.
   */
  Length lengthUnit;

  /**
    * The primary event handler
    */
  tcEHPtr eh;
};

	template<> struct HepMCTraits<HepMC::GenEvent> :
		public HepMCTraitsBase<
			HepMC::GenEvent, HepMC::GenParticle,
			HepMC::GenVertex, HepMC::Polarization,
			HepMC::PdfInfo> {};
}


#endif /* ThePEG_HepMCConverter_H */
