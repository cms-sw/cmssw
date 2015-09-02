// HepMC2.h is a part of the PYTHIA event generator.
// Copyright (C) 2014 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Mikhail Kirsanov, Mikhail.Kirsanov@cern.ch
// Header file and function definitions for the Pythia8ToHepMC class,
// which converts a PYTHIA event record to the standard HepMC format.

#ifndef Pythia8_HepMCA2_H
#define Pythia8_HepMCA2_H

#include <vector>
#include "HepMC/IO_BaseClass.h"
#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include "HepMC/Units.h"
#include "Pythia8/Pythia.h"

namespace HepMC {

//==========================================================================

// The Pythia8ToHepMCA class.

class Pythia8ToHepMCA : public IO_BaseClass {

public:

  // Constructor and destructor.
  Pythia8ToHepMCA() : m_internal_event_number(0),
    m_print_inconsistency(true), m_free_parton_warnings(true),
    m_crash_on_problem(false),   m_convert_gluon_to_0(false) {;}
  virtual ~Pythia8ToHepMCA() {;}

  // Alternative method to convert Pythia events into HepMC ones.
  bool append_event( Pythia8::Event& pyev, GenEvent* evt, GenParticle* rootpart,
    int ibarcode = -1, Pythia8::Info* pyinfo = 0, Pythia8::Settings* pyset = 0);

  // Read out values for some switches.
  bool print_inconsistency()  const {return m_print_inconsistency;}
  bool free_parton_warnings() const {return m_free_parton_warnings;}
  bool crash_on_problem()     const {return m_crash_on_problem;}
  bool convert_gluon_to_0()   const {return m_convert_gluon_to_0;}

  // Set values for some switches.
  void set_print_inconsistency(bool b = true)  {m_print_inconsistency = b;}
  void set_free_parton_warnings(bool b = true) {m_free_parton_warnings = b;}
  void set_crash_on_problem(bool b = false)    {m_crash_on_problem = b;}
  void set_convert_gluon_to_0(bool b = false)  {m_convert_gluon_to_0 = b;}

private:

  // Following methods are not implemented for this class.
  virtual bool fill_next_event( GenEvent*  ) { return 0; }
  virtual void write_event( const GenEvent* ) {;}

  // Use of copy constructor is not allowed.
  Pythia8ToHepMCA( const Pythia8ToHepMCA& ) : IO_BaseClass() {;}

  // Data members.
  int  m_internal_event_number;
  bool m_print_inconsistency, m_free_parton_warnings,
       m_crash_on_problem, m_convert_gluon_to_0;

};

//==========================================================================

// Main method to append PYTHIA event to HepMC event.
// Read one event from Pythia8, append GenEvent
// and return T/F = success/failure.

inline bool Pythia8ToHepMCA::append_event( Pythia8::Event& pyev, GenEvent* evt, GenParticle* rootpart,
  int ibarcode, Pythia8::Info* pyinfo, Pythia8::Settings* pyset) {

  // 1. Error if no event passed.
  if (!evt) {
    std::cerr << "Pythia8ToHepMCA::fill_next_event error - passed null event."
              << std::endl;
    return 0;
  }

  // Conversion factors from Pythia units GeV and mm to HepMC ones.
  double momFac = HepMC::Units::conversion_factor(HepMC::Units::GEV,
    evt->momentum_unit());
  double lenFac = HepMC::Units::conversion_factor(HepMC::Units::MM,
    evt->length_unit());
    
  int NewBarcode = evt->particles_size();
  if (ibarcode > -1) NewBarcode = ibarcode;

  GenVertex* prod_vtx0 = new GenVertex();
  prod_vtx0->add_particle_in( rootpart );
  evt->add_vertex( prod_vtx0 );

  // 2. Create a particle instance for each entry and fill a map, and
  // a vector which maps from the particle index to the GenParticle address.
  std::vector<GenParticle*> hepevt_particles( pyev.size() );
  for (int i = 2; i < pyev.size(); ++i) {

    // Fill the particle.

    hepevt_particles[i] = new GenParticle(
      FourVector( momFac * pyev[i].px(), momFac * pyev[i].py(),
                  momFac * pyev[i].pz(), momFac * pyev[i].e()  ),
      pyev[i].id(), pyev[i].statusHepMC() );
    if (ibarcode !=0) NewBarcode++;
    hepevt_particles[i]->suggest_barcode(NewBarcode);
    hepevt_particles[i]->set_generated_mass( momFac * pyev[i].m() );

    // Colour flow uses index 1 and 2.
    int colType = pyev[i].colType();
    if (colType ==  1 || colType == 2)
      hepevt_particles[i]->set_flow(1, pyev[i].col());
    if (colType == -1 || colType == 2)
      hepevt_particles[i]->set_flow(2, pyev[i].acol());
  }

  // 3. Loop over particles AGAIN, this time creating vertices.
  // We build the production vertex for each entry in hepevt.
  // The HEPEVT pointers are bi-directional, so gives decay vertices as well.
  for (int i = 2; i < pyev.size(); ++i) {
    GenParticle* p = hepevt_particles[i];

    // 3a. Search to see if a production vertex already exists.
    std::vector<int> mothers = pyev[i].motherList();

    unsigned int imother = 0;
    int mother = -1; // note that in Pythia8 there is a particle number 0!
    if ( !mothers.empty() ) mother = mothers[imother];
    GenVertex* prod_vtx = p->production_vertex();
    while ( !prod_vtx && mother > 0 ) {
      if(mother == 1) {
        prod_vtx = rootpart->end_vertex();
      } else {
        prod_vtx = hepevt_particles[mother]->end_vertex();
      }
      if ( prod_vtx ) prod_vtx->add_particle_out( p );
      mother = ( ++imother < mothers.size() ) ? mothers[imother] : -1;
    }

    // 3b. If no suitable production vertex exists - and the particle has
    // at least one mother or position information to store - make one.
    FourVector prod_pos( lenFac * pyev[i].xProd(), lenFac * pyev[i].yProd(),
                         lenFac * pyev[i].zProd(), lenFac * pyev[i].tProd() );
    if ( !prod_vtx && ( mothers.size() > 0 || prod_pos != FourVector() ) ) {
      prod_vtx = new GenVertex();
      prod_vtx->add_particle_out( p );
      evt->add_vertex( prod_vtx );
    }

    // 3c. If prod_vtx doesn't already have position specified, fill it.
    if ( prod_vtx && prod_vtx->position() == FourVector() )
      prod_vtx->set_position( prod_pos );

    // 3d. loop over mothers to make sure their end_vertices are consistent.
    imother = 0;
    mother = -1;
    if ( !mothers.empty() ) mother = mothers[imother];
    while ( prod_vtx && mother > 0 ) {

      // If end vertex of the mother isn't specified, do it now.
      GenParticle* ppp;
      if(mother == 1) ppp = rootpart;
      if(mother > 1) ppp = hepevt_particles[mother];

      if ( !ppp->end_vertex() ) {
        prod_vtx->add_particle_in( ppp );

      // Problem scenario: the mother already has a decay vertex which
      // differs from the daughter's production vertex. This means there is
      // internal inconsistency in the HEPEVT event record. Print an error.
      // Note: we could provide a fix by joining the two vertices with a
      // dummy particle if the problem arises often.
      } else if (ppp->end_vertex() != prod_vtx ) {
       if ( m_print_inconsistency ) std::cerr
          << "HepMC::Pythia8ToHepMC: inconsistent mother/daugher "
          << "information in Pythia8 event " << std::endl
          << "i = " << i << " mother = " << mother
          << "\n This warning can be turned off with the "
          << "Pythia8ToHepMC::print_inconsistency switch." << std::endl;
      }

      // End of vertex-setting loops.
      mother = ( ++imother < mothers.size() ) ? mothers[imother] : -1;
    }
  }

  // If hadronization switched on then no final coloured particles.
  bool doHadr = (pyset == 0) ? m_free_parton_warnings
    : pyset->flag("HadronLevel:all") && pyset->flag("HadronLevel:Hadronize");

  // 4. Check for particles which come from nowhere, i.e. are without
  // mothers or daughters. These need to be attached to a vertex, or else
  // they will never become part of the event.
  for (int i = 2; i < pyev.size(); ++i) {
    if ( !hepevt_particles[i]->end_vertex() &&
         !hepevt_particles[i]->production_vertex() ) {
      std::cerr << "hanging particle " << i << std::endl;
      GenVertex* prod_vtx = new GenVertex();
      prod_vtx->add_particle_out( hepevt_particles[i] );
      evt->add_vertex( prod_vtx );
    }

    // Also check for free partons (= gluons and quarks; not diquarks?).
    if ( doHadr && m_free_parton_warnings ) {
      if ( hepevt_particles[i]->pdg_id() == 21 &&
        !hepevt_particles[i]->end_vertex() ) {
        std::cerr << "gluon without end vertex " << i << std::endl;
        if ( m_crash_on_problem ) exit(1);
      }
      if ( abs(hepevt_particles[i]->pdg_id()) <= 6 &&
        !hepevt_particles[i]->end_vertex()         ) {
        std::cerr << "quark without end vertex " << i << std::endl;
        if ( m_crash_on_problem ) exit(1);
      }
    }
  }

  // Done.
  return true;

}

//==========================================================================

} // end namespace HepMC

#endif  // end Pythia8_HepMCA2_H
