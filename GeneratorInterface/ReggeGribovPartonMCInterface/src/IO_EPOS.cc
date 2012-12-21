
// EPOS IO class

#include "GeneratorInterface/ReggeGribovPartonMCInterface/interface/IO_EPOS.h"
#include "HepMC/GenEvent.h"
#include <cstdio>       // needed for formatted output using sprintf 

using namespace HepMC;

namespace EPOS {

    unsigned int EPOS_Wrapper::s_sizeof_int = 4;
    unsigned int EPOS_Wrapper::s_sizeof_real = sizeof(double);
    unsigned int EPOS_Wrapper::s_max_number_entries = 99900;

    IO_EPOS::IO_EPOS() : m_trust_mothers_before_daughters(1),
                             m_trust_both_mothers_and_daughters(0),
                             m_print_inconsistency_errors(1),
                             m_trust_beam_particles(true)
    {}

    IO_EPOS::~IO_EPOS(){}

    void IO_EPOS::print( std::ostream& ostr ) const { 
        ostr << "IO_EPOS: reads an event from the FORTRAN EPOS g "
             << "common block. \n" 
             << " trust_mothers_before_daughters = " 
             << m_trust_mothers_before_daughters
             << " trust_both_mothers_and_daughters = "
             << m_trust_both_mothers_and_daughters
             << ", print_inconsistency_errors = " 
             << m_print_inconsistency_errors << std::endl;
    }

    bool IO_EPOS::fill_next_event( HepMC::GenEvent* evt ) {
        //
        // 1. test that evt pointer is not null and set event number
        if ( !evt ) {
            std::cerr 
                << "IO_EPOS::fill_next_event error - passed null event." 
                << std::endl;
            return false;
        }
        evt->set_event_number( EPOS_Wrapper::event_number() );
        //
        // 2. create a particle instance for each EPOS entry and fill a map
        //    create a vector which maps from the EPOS particle index to the 
        //    GenParticle address
        //    (+1 in size accounts for hepevt_particle[0] which is unfilled)
        std::vector<HepMC::GenParticle*> hepevt_particle( 
                                        EPOS_Wrapper::number_entries()+1 );
        hepevt_particle[0] = 0;
	//intentionally skipping last particle in event record which is the nucleus
        for ( int i1 = 1; i1 < EPOS_Wrapper::number_entries(); ++i1 ) {
            hepevt_particle[i1] = build_particle(i1);
        }

	HepMC::GenVertex*  primaryVertex = new HepMC::GenVertex(HepMC::FourVector(0,0,0,0),0);
	evt->add_vertex(primaryVertex);
	if(!evt->signal_process_vertex()) evt->set_signal_process_vertex(primaryVertex);
	
        std::set<HepMC::GenVertex*> new_vertices;
        //
        // Here we assume that the first two particles in the list 
        // are the incoming beam particles.
        if( trust_beam_particles() ) {
	  evt->set_beam_particles( hepevt_particle[1], hepevt_particle[2] );
        }
        //
        // 3.+4. loop over EPOS particles AGAIN, this time creating vertices
        for ( int i = 1; i < EPOS_Wrapper::number_entries(); ++i ) {
            // We go through and build EITHER the production or decay 
            // vertex for each entry in hepevt, depending on the switch
            // m_trust_mothers_before_daughters (new 2001-02-28)
            // Note: since the EPOS pointers are bi-directional, it is
            //
            // 3. Build the production_vertex (if necessary)
            if ( m_trust_mothers_before_daughters || 
                 m_trust_both_mothers_and_daughters ) {
                build_production_vertex( i, hepevt_particle, evt );
            }
            //
            // 4. Build the end_vertex (if necessary) 
            //    Identical steps as for production vertex
            if ( !m_trust_mothers_before_daughters || 
                 m_trust_both_mothers_and_daughters ) {
                build_end_vertex( i, hepevt_particle, evt );
            }
        }
        // 5.             01.02.2000
        // handle the case of particles in EPOS which come from nowhere -
        //  i.e. particles without mothers or daughters.
        //  These particles need to be attached to a vertex, or else they
        //  will never become part of the event. check for this situation
        for ( int i3 = 1; i3 < EPOS_Wrapper::number_entries(); ++i3 ) {
            if ( !hepevt_particle[i3]->end_vertex() && 
                        !hepevt_particle[i3]->production_vertex() ) {
                HepMC::GenVertex* prod_vtx = new GenVertex();
                prod_vtx->add_particle_out( hepevt_particle[i3] );
                evt->add_vertex( prod_vtx );
            }
        }
        return true;
    }

    void IO_EPOS::write_event( const GenEvent* evt ) {
        //
        if ( !evt ) return;
        //
        // map all particles onto a unique index
        std::vector<HepMC::GenParticle*> index_to_particle(
            EPOS_Wrapper::max_number_entries()+1 );
        index_to_particle[0]=0;
        std::map<HepMC::GenParticle*,int> particle_to_index;
        int particle_counter=0;
        for ( HepMC::GenEvent::vertex_const_iterator v = evt->vertices_begin();
              v != evt->vertices_end(); ++v ) {
            // all "mothers" or particles_in are kept adjacent in the list
            // so that the mother indices in hepevt can be filled properly
            for ( HepMC::GenVertex::particles_in_const_iterator p1 
                      = (*v)->particles_in_const_begin();
                  p1 != (*v)->particles_in_const_end(); ++p1 ) {
                ++particle_counter;
                if ( particle_counter > 
                     EPOS_Wrapper::max_number_entries() ) break; 
                index_to_particle[particle_counter] = *p1;
                particle_to_index[*p1] = particle_counter;
            }
            // daughters are entered only if they aren't a mother of 
            // another vtx
            for ( HepMC::GenVertex::particles_out_const_iterator p2 
                      = (*v)->particles_out_const_begin();
                  p2 != (*v)->particles_out_const_end(); ++p2 ) {
                if ( !(*p2)->end_vertex() ) {
                    ++particle_counter;
                    if ( particle_counter > 
                         EPOS_Wrapper::max_number_entries() ) {
                        break;
                    }
                    index_to_particle[particle_counter] = *p2;
                    particle_to_index[*p2] = particle_counter;
                }
            }
        }
        if ( particle_counter > EPOS_Wrapper::max_number_entries() ) {
            particle_counter = EPOS_Wrapper::max_number_entries();
        }
        //      
        // fill the EPOS event record
        EPOS_Wrapper::set_event_number( evt->event_number() );
        EPOS_Wrapper::set_number_entries( particle_counter );
        for ( int i = 1; i <= particle_counter; ++i ) {
            EPOS_Wrapper::set_status( i, index_to_particle[i]->status() );
            EPOS_Wrapper::set_id( i, index_to_particle[i]->pdg_id() );
            FourVector m = index_to_particle[i]->momentum();
            EPOS_Wrapper::set_momentum( i, m.px(), m.py(), m.pz(), m.e() );
            EPOS_Wrapper::set_mass( i, index_to_particle[i]->generatedMass() );
            // there should ALWAYS be particles in any vertex, but some generators
            // are making non-kosher HepMC events
            if ( index_to_particle[i]->production_vertex() && 
                 index_to_particle[i]->production_vertex()->particles_in_size()) {
                FourVector p = index_to_particle[i]->
                                     production_vertex()->position();
                EPOS_Wrapper::set_position( i, p.x(), p.y(), p.z(), p.t() );
                int num_mothers = index_to_particle[i]->production_vertex()->
                                  particles_in_size();
                int first_mother = find_in_map( particle_to_index,
                                                *(index_to_particle[i]->
                                                  production_vertex()->
                                                  particles_in_const_begin()));
                int last_mother = first_mother + num_mothers - 1;
                if ( first_mother == 0 ) last_mother = 0;
                EPOS_Wrapper::set_parents( i, first_mother, last_mother );
            } else {
                EPOS_Wrapper::set_position( i, 0, 0, 0, 0 );
                EPOS_Wrapper::set_parents( i, 0, 0 );
            }
            EPOS_Wrapper::set_children( i, 0, 0 );
        }
    }

    void IO_EPOS::build_production_vertex(int i, 
                                            std::vector<HepMC::GenParticle*>& 
                                            hepevt_particle,
                                            GenEvent* evt ) {
        HepMC::GenParticle* p = hepevt_particle[i];
        // a. search to see if a production vertex already exists
        int mother = EPOS_Wrapper::first_parent(i);
        HepMC::GenVertex* prod_vtx = p->production_vertex();
        while ( !prod_vtx && mother > 0 ) {
            prod_vtx = hepevt_particle[mother]->end_vertex();
            if ( prod_vtx ) prod_vtx->add_particle_out( p );
            // increment mother for next iteration
            if ( ++mother > EPOS_Wrapper::last_parent(i) ) mother = 0;
        }
        // b. if no suitable production vertex exists - and the particle
        // has atleast one mother or position information to store - 
        // make one
        HepMC::FourVector prod_pos( EPOS_Wrapper::x(i), EPOS_Wrapper::y(i), 
                                   EPOS_Wrapper::z(i), EPOS_Wrapper::t(i) 
                                 ); 
        if ( !prod_vtx && (EPOS_Wrapper::number_parents(i)>0 
                           || prod_pos!=FourVector(0,0,0,0)) )
        {
            prod_vtx = new HepMC::GenVertex();
            prod_vtx->add_particle_out( p );
            evt->add_vertex( prod_vtx );
        }
        // c. if prod_vtx doesn't already have position specified, fill it
        if ( prod_vtx && prod_vtx->position()==FourVector(0,0,0,0) ) {
            prod_vtx->set_position( prod_pos );
        }
        // d. loop over mothers to make sure their end_vertices are
        //     consistent
        mother = EPOS_Wrapper::first_parent(i);
        while ( prod_vtx && mother > 0 ) {
            if ( !hepevt_particle[mother]->end_vertex() ) {
                // if end vertex of the mother isn't specified, do it now
                prod_vtx->add_particle_in( hepevt_particle[mother] );
            } else if (hepevt_particle[mother]->end_vertex() != prod_vtx ) {
                // problem scenario --- the mother already has a decay
                // vertex which differs from the daughter's produciton 
                // vertex. This means there is internal
                // inconsistency in the EPOS event record. Print an
                // error
                // Note: we could provide a fix by joining the two 
                //       vertices with a dummy particle if the problem
                //       arrises often with any particular generator.
                if ( m_print_inconsistency_errors ) std::cerr
                    << "HepMC::IO_EPOS: inconsistent mother/daugher "
                    << "information in EPOS event " 
                    << EPOS_Wrapper::event_number()
                    << ". \n I recommend you try "
                    << "inspecting the event first with "
                    << "\n\tEPOS_Wrapper::check_hepevt_consistency()"
                    << "\n This warning can be turned off with the "
                    << "IO_EPOS::print_inconsistency_errors switch."
                    << std::endl;
            }
            if ( ++mother > EPOS_Wrapper::last_parent(i) ) mother = 0;
        }
    }

    void IO_EPOS::build_end_vertex
    ( int i, std::vector<HepMC::GenParticle*>& hepevt_particle, GenEvent* evt ) 
    {
        //    Identical steps as for build_production_vertex
        HepMC::GenParticle* p = hepevt_particle[i];
        // a.
        int daughter = EPOS_Wrapper::first_child(i);
        HepMC::GenVertex* end_vtx = p->end_vertex();
        while ( !end_vtx && daughter > 0 ) {
            end_vtx = hepevt_particle[daughter]->production_vertex();
            if ( end_vtx ) end_vtx->add_particle_in( p );
            if ( ++daughter > EPOS_Wrapper::last_child(i) ) daughter = 0;
        }
        // b. (different from 3c. because EPOS particle can not know its
        //        decay position )
        if ( !end_vtx && EPOS_Wrapper::number_children(i)>0 ) {
            end_vtx = new GenVertex();
            end_vtx->add_particle_in( p );
            evt->add_vertex( end_vtx );
        }
        // c+d. loop over daughters to make sure their production vertices 
        //    point back to the current vertex.
        //    We get the vertex position from the daughter as well.
        daughter = EPOS_Wrapper::first_child(i);
        while ( end_vtx && daughter > 0 ) {
            if ( !hepevt_particle[daughter]->production_vertex() ) {
                // if end vertex of the mother isn't specified, do it now
                end_vtx->add_particle_out( hepevt_particle[daughter] );
                // 
                // 2001-03-29 M.Dobbs, fill vertex the position.
                if ( end_vtx->position()==FourVector(0,0,0,0) ) {
		  // again mm to cm conversion 
                    FourVector prod_pos( EPOS_Wrapper::x(daughter), 
                                               EPOS_Wrapper::y(daughter), 
                                               EPOS_Wrapper::z(daughter), 
                                               EPOS_Wrapper::t(daughter) 
                        );
                    if ( prod_pos != FourVector(0,0,0,0) ) {
                        end_vtx->set_position( prod_pos );
                    }
                }
            } else if (hepevt_particle[daughter]->production_vertex() 
                       != end_vtx){
                // problem scenario --- the daughter already has a prod
                // vertex which differs from the mother's end 
                // vertex. This means there is internal
                // inconsistency in the EPOS event record. Print an
                // error
                if ( m_print_inconsistency_errors ) std::cerr
                    << "HepMC::IO_EPOS: inconsistent mother/daugher "
                    << "information in EPOS event " 
                    << EPOS_Wrapper::event_number()
                    << ". \n I recommend you try "
                    << "inspecting the event first with "
                    << "\n\tEPOS_Wrapper::check_hepevt_consistency()"
                    << "\n This warning can be turned off with the "
                    << "IO_EPOS::print_inconsistency_errors switch."
                    << std::endl;
            }
            if ( ++daughter > EPOS_Wrapper::last_child(i) ) daughter = 0;
        }
        if ( !p->end_vertex() && !p->production_vertex() ) {
            // Added 2001-11-04, to try and handle Isajet problems.
            build_production_vertex( i, hepevt_particle, evt );
        }
    }

    HepMC::GenParticle* IO_EPOS::build_particle( int index ) {
        // 
        HepMC::GenParticle* p 
            = new GenParticle( FourVector( EPOS_Wrapper::px(index), 
                                                 EPOS_Wrapper::py(index), 
                                                 EPOS_Wrapper::pz(index), 
                                                 EPOS_Wrapper::e(index) ),
                               EPOS_Wrapper::id(index), 
                               EPOS_Wrapper::status(index) );
        p->setGeneratedMass( EPOS_Wrapper::m(index) );
        p->suggest_barcode( index );
        return p;
    }

    int IO_EPOS::find_in_map( const std::map<HepMC::GenParticle*,int>& m, 
                                HepMC::GenParticle* p) const {
        std::map<HepMC::GenParticle*,int>::const_iterator iter = m.find(p);
        if ( iter == m.end() ) return 0;
        return iter->second;
    }

} // HepMC



