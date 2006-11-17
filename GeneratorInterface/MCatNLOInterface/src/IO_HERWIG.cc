//////////////////////////////////////////////////////////////////////////
// Matt.Dobbs@Cern.CH, October 2002
// Herwig 6.400 IO class
//////////////////////////////////////////////////////////////////////////

#include "IO_HERWIG.h"
#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/Vector/LorentzVector.h"
#include <cstdio>       // needed for formatted output using sprintf 


#define FourVector HepLorentzVector

namespace HepMC {

    IO_HERWIG::IO_HERWIG() : m_trust_mothers_before_daughters(false),
			     m_trust_both_mothers_and_daughters(true),
			     m_print_inconsistency_errors(true),
			     m_no_gaps_in_barcodes(true),
			     m_herwig_to_pdg_id(100,0)
    {
	// These arrays are copied from Lynn Garren's stdhep 5.01.
	//   see http://www-pat.fnal.gov/stdhep.html
	// Translation from HERWIG particle ID's to PDG particle ID's.
	m_herwig_to_pdg_id[1] =1; 
	m_herwig_to_pdg_id[2] =2;
	m_herwig_to_pdg_id[3] =3;
	m_herwig_to_pdg_id[4] =4;
	m_herwig_to_pdg_id[5] =5;
	m_herwig_to_pdg_id[6] =6;
	m_herwig_to_pdg_id[7] =7;
	m_herwig_to_pdg_id[8] =8;
       
	m_herwig_to_pdg_id[11] =11;
	m_herwig_to_pdg_id[12] =12;
	m_herwig_to_pdg_id[13] =13;
	m_herwig_to_pdg_id[14] =14;
	m_herwig_to_pdg_id[15] =15;
	m_herwig_to_pdg_id[16] =16;
       
	m_herwig_to_pdg_id[21] =21;
	m_herwig_to_pdg_id[22] =22;
	m_herwig_to_pdg_id[23] =23;
	m_herwig_to_pdg_id[24] =24;
	m_herwig_to_pdg_id[25] =25;
	m_herwig_to_pdg_id[26] =51; // <--
       
	m_herwig_to_pdg_id[32] =32;
	m_herwig_to_pdg_id[35] =35;
	m_herwig_to_pdg_id[36] =36;
	m_herwig_to_pdg_id[37] =37;
	m_herwig_to_pdg_id[39] =39;
       
	m_herwig_to_pdg_id[81] =81;
	m_herwig_to_pdg_id[82] =82;
	m_herwig_to_pdg_id[83] =83;
	m_herwig_to_pdg_id[84] =84;
	m_herwig_to_pdg_id[85] =85;
	m_herwig_to_pdg_id[86] =86;
	m_herwig_to_pdg_id[87] =87;
	m_herwig_to_pdg_id[88] =88;
	m_herwig_to_pdg_id[89] =89;
	m_herwig_to_pdg_id[90] =90;
       
	m_herwig_to_pdg_id[91] =91;
	m_herwig_to_pdg_id[92] =92;
	m_herwig_to_pdg_id[93] =93;
	m_herwig_to_pdg_id[94] =94;
	m_herwig_to_pdg_id[95] =95;
	m_herwig_to_pdg_id[96] =96;
	m_herwig_to_pdg_id[97] =97;
	m_herwig_to_pdg_id[98] =9920022; // <--
	m_herwig_to_pdg_id[99] =9922212; // <--

	// These particle ID's have no antiparticle, so aren't allowed.
	m_no_antiparticles.insert(-21);
	m_no_antiparticles.insert(-22);
	m_no_antiparticles.insert(-23);
	m_no_antiparticles.insert(-25);
	m_no_antiparticles.insert(-51);
	m_no_antiparticles.insert(-35);
	m_no_antiparticles.insert(-36);
    }

    IO_HERWIG::~IO_HERWIG(){}

    void IO_HERWIG::print( std::ostream& ostr ) const { 
        ostr << "IO_HERWIG: reads an event from the FORTRAN Herwig HEPEVT "
             << "common block. \n" 
	     << " trust_mothers_before_daughters = " 
	     << m_trust_mothers_before_daughters
	     << " trust_both_mothers_and_daughters = "
	     << m_trust_both_mothers_and_daughters
	     << " print_inconsistency_errors = " 
	     << m_print_inconsistency_errors << std::endl;
    }

    bool IO_HERWIG::fill_next_event( GenEvent* evt ) {
	// read one event from the Herwig HEPEVT common block and fill GenEvent
	// return T/F =success/failure
	//
	// 0. Test that evt pointer is not null and set event number
	if ( !evt ) {
	    std::cerr 
		<< "IO_HERWIG::fill_next_event error - passed null event." 
		<< std::endl;
	    return 0;
	}

	// 1. First we have to fix the HEPEVT input, which is all mucked up for
	//    herwig.
	repair_hepevt();

	evt->set_event_number( HEPEVT_Wrapper::event_number() );
	//
	// 2. create a particle instance for each HEPEVT entry and fill a map
	//    create a vector which maps from the HEPEVT particle index to the 
	//    GenParticle address
	//    (+1 in size accounts for hepevt_particle[0] which is unfilled)
	std::vector<GenParticle*> hepevt_particle( 
	                                HEPEVT_Wrapper::number_entries()+1 );
	hepevt_particle[0] = 0;
	for ( int i1 = 1; i1 <= HEPEVT_Wrapper::number_entries(); ++i1 ) {
	    hepevt_particle[i1] = build_particle(i1);
	}
	std::set<GenVertex*> new_vertices;
	//
	// 3. We need to take special care with the hard process
	// vertex.  The problem we are trying to avoid is when the
	// partons entering the hard process also have daughters from
	// the parton shower. When this happens, each one can get its
	// own decay vertex, making it difficult to join them
	// later. We handle it by joining them together first, then
	// the other daughters get added on later.
	// Find the partons entering the hard vertex (status codes 121, 122).
	int index_121 = 0;
	int index_122 = 0;
	for ( int i = 1; i <=HEPEVT_Wrapper::number_entries(); i++ ) {
	    if ( HEPEVT_Wrapper::status(i)==121 ) index_121=i;
	    if ( HEPEVT_Wrapper::status(i)==122 ) index_122=i;
	    if ( index_121!=0 && index_122!=0 ) break;
	}
	if ( index_121 && index_122 ) {
	    GenVertex* hard_vtx = new GenVertex();
	    hard_vtx->add_particle_in( hepevt_particle[index_121] );
	    hard_vtx->add_particle_in( hepevt_particle[index_122] );
	    // evt->add_vertex( hard_vtx ); // not necessary, its done in 
	                                    // set_signal_process_vertex
	    evt->set_signal_process_vertex( hard_vtx );
	}
	//
	// 4. loop over HEPEVT particles AGAIN, this time creating vertices
	for ( int i = 1; i <= HEPEVT_Wrapper::number_entries(); ++i ) {
	    // We go through and build EITHER the production or decay 
	    // vertex for each entry in hepevt, depending on the switch
	    // m_trust_mothers_before_daughters (new 2001-02-28)
	    // Note: since the HEPEVT pointers are bi-directional, it is
	    ///      sufficient to do one or the other.
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
	// handle the case of particles in HEPEVT which come from nowhere -
	//  i.e. particles without mothers or daughters.
	//  These particles need to be attached to a vertex, or else they
	//  will never become part of the event. check for this situation.
	for ( int i3 = 1; i3 <= HEPEVT_Wrapper::number_entries(); ++i3 ) {
	    // Herwig also has some non-physical entries in HEPEVT
	    // like CMS, HARD, and CONE. These are flagged by
	    // repair_hepevt by making their status and id zero. We
	    // delete those particles here.
	    if ( hepevt_particle[i3] && !hepevt_particle[i3]->parent_event()
		 && !hepevt_particle[i3]->pdg_id()
		 && !hepevt_particle[i3]->status() ) {
		//std::cout << "IO_HERWIG::fill_next_event is deleting null "
		//	  << "particle" << std::endl;
		//hepevt_particle[i3]->print();
		delete hepevt_particle[i3];
	    } else if ( hepevt_particle[i3] && 
			!hepevt_particle[i3]->end_vertex() && 
			!hepevt_particle[i3]->production_vertex() ) {
		GenVertex* prod_vtx = new GenVertex();
		prod_vtx->add_particle_out( hepevt_particle[i3] );
		evt->add_vertex( prod_vtx );
	    }
	}
	return true;
    }

    void IO_HERWIG::build_production_vertex(int i, 
					    std::vector<GenParticle*>& 
					    hepevt_particle,
					    GenEvent* evt ) {
	// 
	// for particle in HEPEVT with index i, build a production vertex
	// if appropriate, and add that vertex to the event
	GenParticle* p = hepevt_particle[i];
	// a. search to see if a production vertex already exists
	int mother = HEPEVT_Wrapper::first_parent(i);
	GenVertex* prod_vtx = p->production_vertex();
	while ( !prod_vtx && mother > 0 ) {
	    prod_vtx = hepevt_particle[mother]->end_vertex();
	    if ( prod_vtx ) prod_vtx->add_particle_out( p );
	    // increment mother for next iteration
	    if ( ++mother > HEPEVT_Wrapper::last_parent(i) ) mother = 0;
	}
	// b. if no suitable production vertex exists - and the particle
	// has atleast one mother or position information to store - 
	// make one
	FourVector prod_pos( HEPEVT_Wrapper::x(i), HEPEVT_Wrapper::y(i), 
				   HEPEVT_Wrapper::z(i), HEPEVT_Wrapper::t(i) 
	                         ); 
	if ( !prod_vtx && (HEPEVT_Wrapper::number_parents(i)>0 
			   || prod_pos!=FourVector(0,0,0,0)) )
	{
	    prod_vtx = new GenVertex();
	    prod_vtx->add_particle_out( p );
	    evt->add_vertex( prod_vtx ); 
	}
	// c. if prod_vtx doesn't already have position specified, fill it
	if ( prod_vtx && prod_vtx->position()==FourVector(0,0,0,0) ) {
	    prod_vtx->set_position( prod_pos );
	}
	// d. loop over mothers to make sure their end_vertices are
	//     consistent
	mother = HEPEVT_Wrapper::first_parent(i);
	while ( prod_vtx && mother > 0 ) {
	    if ( !hepevt_particle[mother]->end_vertex() ) {
		// if end vertex of the mother isn't specified, do it now
		prod_vtx->add_particle_in( hepevt_particle[mother] );
	    } else if (hepevt_particle[mother]->end_vertex() != prod_vtx ) {
		// problem scenario --- the mother already has a decay
		// vertex which differs from the daughter's produciton 
		// vertex. This means there is internal
		// inconsistency in the HEPEVT event record. Print an
		// error
		// Note: we could provide a fix by joining the two 
		//       vertices with a dummy particle if the problem
		//       arrises often with any particular generator.
		if ( m_print_inconsistency_errors ) {
		  std::cerr
		    << "HepMC::IO_HERWIG: inconsistent mother/daugher "
		    << "information in HEPEVT event " 
		    << HEPEVT_Wrapper::event_number()
		    << ". \n I recommend you try "
		    << "inspecting the event first with "
		    << "\n\tHEPEVT_Wrapper::check_hepevt_consistency()"
		    << "\n This warning can be turned off with the "
		    << "IO_HERWIG::print_inconsistency_errors switch."
		    << std::endl;
		  hepevt_particle[mother]->print(std::cerr);
		  std::cerr
		    << "problem vertices are: (prod_vtx, mother)" << std::endl;
		  if ( prod_vtx ) prod_vtx->print(std::cerr);
		  hepevt_particle[mother]->end_vertex()->print(std::cerr);
		}
	    }
	    if ( ++mother > HEPEVT_Wrapper::last_parent(i) ) mother = 0;
	}
    }

    void IO_HERWIG::build_end_vertex
    ( int i, std::vector<GenParticle*>& hepevt_particle, GenEvent* evt ) 
    {
	// 
	// for particle in HEPEVT with index i, build an end vertex
	// if appropriate, and add that vertex to the event
	//    Identical steps as for build_production_vertex
	GenParticle* p = hepevt_particle[i];
	// a.
	int daughter = HEPEVT_Wrapper::first_child(i);
	GenVertex* end_vtx = p->end_vertex();
	while ( !end_vtx && daughter > 0 ) {
	    end_vtx = hepevt_particle[daughter]->production_vertex();
	    if ( end_vtx ) end_vtx->add_particle_in( p );
	    if ( ++daughter > HEPEVT_Wrapper::last_child(i) ) daughter = 0;
	}
	// b. (different from 3c. because HEPEVT particle can not know its
	//        decay position )
	if ( !end_vtx && HEPEVT_Wrapper::number_children(i)>0 ) {
	    end_vtx = new GenVertex();
	    end_vtx->add_particle_in( p );
	    evt->add_vertex( end_vtx );
	}
	// c+d. loop over daughters to make sure their production vertices 
	//    point back to the current vertex.
	//    We get the vertex position from the daughter as well.
	daughter = HEPEVT_Wrapper::first_child(i);
	while ( end_vtx && daughter > 0 ) {
	    if ( !hepevt_particle[daughter]->production_vertex() ) {
		// if end vertex of the mother isn't specified, do it now
		end_vtx->add_particle_out( hepevt_particle[daughter] );
		// 
		// 2001-03-29 M.Dobbs, fill vertex the position.
		if ( end_vtx->position()==FourVector(0,0,0,0) ) {
		    FourVector prod_pos( HEPEVT_Wrapper::x(daughter), 
					       HEPEVT_Wrapper::y(daughter), 
					       HEPEVT_Wrapper::z(daughter), 
					       HEPEVT_Wrapper::t(daughter) 
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
		// inconsistency in the HEPEVT event record. Print an
		// error
		if ( m_print_inconsistency_errors ) std::cerr
		    << "HepMC::IO_HERWIG: inconsistent mother/daugher "
		    << "information in HEPEVT event " 
		    << HEPEVT_Wrapper::event_number()
		    << ". \n I recommend you try "
		    << "inspecting the event first with "
		    << "\n\tHEPEVT_Wrapper::check_hepevt_consistency()"
		    << "\n This warning can be turned off with the "
		    << "IO_HERWIG::print_inconsistency_errors switch."
		    << std::endl;
	    }
	    if ( ++daughter > HEPEVT_Wrapper::last_child(i) ) daughter = 0;
	}
	if ( !p->end_vertex() && !p->production_vertex() ) {
	    // Added 2001-11-04, to try and handle Isajet problems.
	    build_production_vertex( i, hepevt_particle, evt );
	}
    }

    GenParticle* IO_HERWIG::build_particle( int index ) {
	// Builds a particle object corresponding to index in HEPEVT
	// 
	GenParticle* p 
	    = new GenParticle( FourVector( HEPEVT_Wrapper::px(index), 
						 HEPEVT_Wrapper::py(index), 
						 HEPEVT_Wrapper::pz(index), 
						 HEPEVT_Wrapper::e(index) ),
			       HEPEVT_Wrapper::id(index), 
			       HEPEVT_Wrapper::status(index) );
        p->setGeneratedMass( HEPEVT_Wrapper::m(index) );
	p->suggest_barcode( index );
	return p;
    }

    int IO_HERWIG::find_in_map( const std::map<GenParticle*,int>& m, 
				GenParticle* p) const {
        std::map<GenParticle*,int>::const_iterator iter = m.find(p);
        if ( iter == m.end() ) return 0;
        return iter->second;
    }

    void IO_HERWIG::repair_hepevt() const {
	//  This routine takes the HEPEVT common block as used in HERWIG,
	//  and converts it into the HEPEVT common block in the standard format
	//
	//  This means it:
	//    - removes the color structure, which herwig overloads 
	//      into the mother/daughter fields
	//    - zeros extra entries for hard subprocess, etc.
	//
	//
	// Special HERWIG status codes
	//   101,102   colliding beam particles
	//   103       beam-beam collision CMS vector
	//   120       hard subprocess CMS vector
	//   121,122   hard subprocess colliding partons
	//   123-129   hard subprocess outgoing particles
	//   141-149   (ID=94) mirror image of hard subrpocess particles
	//   100       (ID=0 cone)
	//
	// Special HERWIG particle id's
	//   91 clusters
	//   94 jets
	//   0  others with no pdg code

	// Make sure hepvt isn't empty.
	if ( HEPEVT_Wrapper::number_entries() <= 0 ) return;

	// Find the index of the beam-beam collision and of the hard subprocess
	// Later we will assume that 
	//              101 ---> 121 \. 
	//                             X  Hard subprocess
	//              102 ---> 122 /
	// 
	int index_collision = 0;
	int index_hard = 0;
	int index_101 = 0;
	int index_102 = 0;
	int index_121 = 0;
	int index_122 = 0;

	for ( int i = 1; i <=HEPEVT_Wrapper::number_entries(); i++ ) {
	    if ( HEPEVT_Wrapper::status(i)==101 ) index_101=i;
	    if ( HEPEVT_Wrapper::status(i)==102 ) index_102=i;
	    if ( HEPEVT_Wrapper::status(i)==103 ) index_collision=i;
	    if ( HEPEVT_Wrapper::status(i)==120 ) index_hard=i;
	    if ( HEPEVT_Wrapper::status(i)==121 ) index_121=i;
	    if ( HEPEVT_Wrapper::status(i)==122 ) index_122=i;
	    if ( index_collision!=0 && index_hard!=0 && index_101!=0 && 
		 index_102!=0 && index_121!=0 && index_122!=0 ) break;
	}

	// The mother daughter information for the hard subprocess entry (120)
	// IS correct, whereas the information for the particles participating
	// in the hard subprocess contains instead the color flow relationships
	// Transfer the hard subprocess info onto the other particles
	// in the hard subprocess.
	//
	// We cannot specify daughters of the incoming hard process particles
	// because they have some daughters (their showered versions) which 
	// are not adjacent in the particle record, so we cannot properly 
	// set the daughter indices in hepevt.
	//
	if (index_121) HEPEVT_Wrapper::set_parents(index_121, index_101, 0 );
	if (index_121) HEPEVT_Wrapper::set_children( index_121, 0, 0 );
	if (index_122) HEPEVT_Wrapper::set_parents(index_122, index_102, 0 );
	if (index_122) HEPEVT_Wrapper::set_children( index_122, 0, 0 );

	for ( int i = HEPEVT_Wrapper::first_child(index_hard);
	      i <= HEPEVT_Wrapper::last_child(index_hard); i++ ) {
	    HEPEVT_Wrapper::set_parents( 
		i, HEPEVT_Wrapper::first_parent(index_hard), 
		HEPEVT_Wrapper::last_parent(index_hard) );

	    // When the direct descendants of the hard process are hadrons,
	    // then the 2nd child contains color flow information, and so
	    // we zero it.
	    // However, if the direct descendant is status=195, then it is
	    // a non-hadron, and so the 2nd child does contain real mother
	    // daughter relationships. ( particularly relevant for H->WW,
	    //                           April 18, 2003 )
	    if ( HEPEVT_Wrapper::status(i) != 195 ) {	      
	      HEPEVT_Wrapper::set_children(i,HEPEVT_Wrapper::first_child(i),0);
	    }
	}

	// now zero the collision and hard entries.
	if (index_hard) zero_hepevt_entry(index_hard);
	if (index_hard) zero_hepevt_entry(index_collision);

	//     Loop over the particles individually and handle oddities
	for ( int i=1; i <=HEPEVT_Wrapper::number_entries(); i++ ) {

	    //       ----------- Fix ID codes ----------
	    //       particles with ID=94 are mirror images of their mothers:
	    if ( HEPEVT_Wrapper::id(i)==94 ) {
		HEPEVT_Wrapper::set_id( 
		    i, HEPEVT_Wrapper::id( HEPEVT_Wrapper::first_parent(i) ) );
	    }

	    //     ----------- fix STATUS codes ------
	    //     status=100 particles are "cones" which carry only color info
	    //     throw them away
	    if ( HEPEVT_Wrapper::status(i)==100 ) zero_hepevt_entry(i);


	    // NOTE: status 101,102 particles are the beam particles.
	    //       status 121,129 particles are the hard subprocess particles
	    // we choose to allow the herwig particles to have herwig
	    // specific codes, and so we don't bother to change these
	    // to status =3.




	    //  ----------- fix some MOTHER/DAUGHTER relationships
	    //  Whenever the mother points to the hard process, it is referring
	    //  to a color flow, so we zero it.
	    if ( HEPEVT_Wrapper::last_parent(i)==index_hard ) {
		HEPEVT_Wrapper::set_parents( 
		    i, HEPEVT_Wrapper::first_parent(i), 0 );
	    }

	    // It makes no sense to have a mother that is younger than you are!

	    if ( HEPEVT_Wrapper::first_parent(i) >= i ) {
		HEPEVT_Wrapper::set_parents( i, 0, 0 );
	    }
	    if ( HEPEVT_Wrapper::last_parent(i) >= i ) {
		HEPEVT_Wrapper::set_parents( 
		    i, HEPEVT_Wrapper::first_parent(i), 0 );
	    }

	    // Whenever the second mother/daughter has a lower index than the
	    // first, it means the second mother/daughter contains color
	    // info. Purge it.
	    if ( HEPEVT_Wrapper::last_parent(i) <= 
		 HEPEVT_Wrapper::first_parent(i) ) {
		HEPEVT_Wrapper::set_parents( 
		    i, HEPEVT_Wrapper::first_parent(i), 0 );
	    }

	    if ( HEPEVT_Wrapper::last_child(i) <= 
		 HEPEVT_Wrapper::first_child(i) ) {
		HEPEVT_Wrapper::set_children(
		    i, HEPEVT_Wrapper::first_child(i), 0 );
	    }

	    // The mothers & daughters of a soft centre of mass (stat=170) seem
	    // to be correct, but they are out of sequence. The information is
	    // elsewhere in the event record, so zero it.
	    //
	    if ( HEPEVT_Wrapper::status(i) == 170 ) {
		HEPEVT_Wrapper::set_parents( i, 0, 0 );
		HEPEVT_Wrapper::set_children( i, 0, 0 );
	    }

	    // Recognise clusters.
	    // Case 1: cluster has particle parents.  
	    // Clusters normally DO point to its two
	    // correct mothers, but those 2 mothers are rarely adjacent in the
	    // event record ... so the mother information might say something
	    // like 123,48 where index123 and index48 really are the correct
	    // mothers... however the hepevt standard states that the mother
	    // pointers should give the index range. So we would have to
	    // reorder the event record and add entries if we wanted to use
	    // it. Instead we just zero the mothers, since all of that
	    // information is contained in the daughter information of the
	    // mothers.
	    // Case 2: cluster has a soft process centre of mass (stat=170)
	    // as parent. This is ok, keep it.
	    //
	    // Note if we were going directly to HepMC, then we could 
	    //  use this information properly!

	    if ( HEPEVT_Wrapper::id(i)==91 ) {
		// if the cluster comes from a SOFT (id=0,stat=170)
		if ( HEPEVT_Wrapper::status(HEPEVT_Wrapper::first_parent(i)) 
		     == 170 ) {
		    ; // In this case the mothers are ok
		} else {
		    HEPEVT_Wrapper::set_parents( i, 0, 0 );
		}
	    }
	}
	
	//     ---------- Loop over the particles individually and look 
	//                for mother/daughter inconsistencies.
	// We consider a mother daughter relationship to be valid
	// ONLy when the mother points to the daughter AND the
	// daughter points back (true valid bidirectional
	// pointers) OR when a one thing points to the other, but
	// the other points to zero. If this isn't true, we zero
	// the offending relationship.

	for ( int i=1; i <=HEPEVT_Wrapper::number_entries(); i++ ) {
	    // loop over parents
	    int ifirst = HEPEVT_Wrapper::first_parent(i);
	    int ilast = HEPEVT_Wrapper::last_parent(i);
	    if ( ilast == 0 ) ilast = HEPEVT_Wrapper::first_parent(i);
	    bool first_is_acceptable = true;
	    bool last_is_acceptable = true;
	    // check for out of range.
	    if ( ifirst>=i || ifirst<0 ) first_is_acceptable = false;
	    if ( ilast>=i || ilast<ifirst || ilast<0 )last_is_acceptable=false;
	    if ( first_is_acceptable ) {
		for ( int j = ifirst; j<=ilast; j++ ) {
		    // these are the acceptable outcomes
		    if ( HEPEVT_Wrapper::first_child(j)==i ) {;} 
		    // watch out
		    else if ( HEPEVT_Wrapper::first_child(j) <=i && 
			      HEPEVT_Wrapper::last_child(j) >=i ) {;}
		    else if ( HEPEVT_Wrapper::first_child(j) ==0 && 
			      HEPEVT_Wrapper::last_child(j) ==0 ) {;}

		    // Error Condition:
		    // modified by MADobbs@lbl.gov April 21, 2003
		    // we distinguish between the first parent and all parents
		    //  being incorrect
		    else if (j==ifirst) { first_is_acceptable = false; break; }
		    else { last_is_acceptable = false; break; }
		}
	    }
	    // if any one of the mothers gave a bad outcome, zero all mothers
	    if ( !first_is_acceptable ) {
	      HEPEVT_Wrapper::set_parents( i, 0, 0 );
	    } else if ( !last_is_acceptable ) {
	      HEPEVT_Wrapper::set_parents(i,HEPEVT_Wrapper::first_parent(i),0);
	    }
	}
	// Note: it's important to finish the mother loop, before
	// starting the daughter loop ... since many mother relations
	// will be zero'd which will validate the daughters.... i.e.,
	// we want relationships like:
	//      IHEP    ID      IDPDG IST MO1 MO2 DA1 DA2
	//        27 TQRK           6   3  26  26  30  30
	//        30 TQRK           6 155  26  11  31  32
	// to come out right.

	for ( int i=1; i <=HEPEVT_Wrapper::number_entries(); i++ ) {
	    // loop over daughters
	    int ifirst = HEPEVT_Wrapper::first_child(i);
	    int ilast = HEPEVT_Wrapper::last_child(i);
	    if ( ilast==0 ) ilast = HEPEVT_Wrapper::first_child(i);
	    bool is_acceptable = true;
	    // check for out of range.
	    if ( ifirst<=i || ifirst<0 ) is_acceptable = false;
	    if ( ilast<=i || ilast<ifirst || ilast<0 ) is_acceptable = false;
	    if ( is_acceptable ) {
		for ( int j = ifirst; j<=ilast; j++ ) {
		    // these are the acceptable outcomes
		    if ( HEPEVT_Wrapper::first_parent(j)==i ) {;} 
		    else if ( HEPEVT_Wrapper::first_parent(j) <=i && 
			      HEPEVT_Wrapper::last_parent(j) >=i ) {;}
		    else if ( HEPEVT_Wrapper::first_parent(j) ==0 && 
			      HEPEVT_Wrapper::last_parent(j) ==0 ) {;}
		    else { is_acceptable = false; } // error condition 
		}
	    }
	    // if any one of the children gave a bad outcome, zero all children
	    if ( !is_acceptable ) HEPEVT_Wrapper::set_children( i, 0, 0 );
	}

	// fixme

	for ( int i=1; i <=HEPEVT_Wrapper::number_entries(); i++ ) {
	    HEPEVT_Wrapper::set_id(
		i, translate_herwig_to_pdg_id(HEPEVT_Wrapper::id(i)) );
	}


	if ( m_no_gaps_in_barcodes ) remove_gaps_in_hepevt();
    }

    void IO_HERWIG::remove_gaps_in_hepevt() const {
	// in this scenario, we do not allow there to be zero-ed
	// entries in the HEPEVT common block, and so be reshuffle
	// the common block, removing the zeero-ed entries as we
	// go and making sure we keep the mother/daughter
	// relationships appropriate
	std::vector<int> mymap(HEPEVT_Wrapper::number_entries()+1,0);
	int ilast = 0;
	for ( int i=1; i <=HEPEVT_Wrapper::number_entries(); i++ ) {
	    if (HEPEVT_Wrapper::status(i)==0 && HEPEVT_Wrapper::id(i)==0) {
		// we remove all entries for which stat=0, id=0
		mymap[i]=0;
	    } else {
		ilast += 1;
		if ( ilast != i ) {
		    HEPEVT_Wrapper::set_status(ilast, 
					       HEPEVT_Wrapper::status(i) );
		    HEPEVT_Wrapper::set_id(ilast, HEPEVT_Wrapper::id(i) );
		    HEPEVT_Wrapper::set_parents(
			ilast, 
			HEPEVT_Wrapper::first_parent(i),
			HEPEVT_Wrapper::last_parent(i) );
		    HEPEVT_Wrapper::set_children(
			ilast, 
			HEPEVT_Wrapper::first_child(i),
			HEPEVT_Wrapper::last_child(i) );
		    HEPEVT_Wrapper::set_momentum(
			ilast, 
			HEPEVT_Wrapper::px(i), HEPEVT_Wrapper::py(i),
			HEPEVT_Wrapper::pz(i), HEPEVT_Wrapper::e(i)  );
		    HEPEVT_Wrapper::set_mass(ilast, HEPEVT_Wrapper::m(i) );
		    HEPEVT_Wrapper::set_position(
			ilast, HEPEVT_Wrapper::x(i),HEPEVT_Wrapper::y(i),
			HEPEVT_Wrapper::z(i),HEPEVT_Wrapper::t(i) );
		}
		mymap[i]=ilast;
	    }
	}

	// M. Dobbs (from Borut) - April 26, to fix tauolo/herwig past
	// the end problem with daughter pointers: 
	// HEPEVT_Wrapper::set_number_entries( ilast );

	// Finally we need to re-map the mother/daughter pointers.	
	for ( int i=1; i <=ilast; i++ ) {

	    HEPEVT_Wrapper::set_parents(
		i, 
		mymap[HEPEVT_Wrapper::first_parent(i)],
		mymap[HEPEVT_Wrapper::last_parent(i)] );
	    HEPEVT_Wrapper::set_children(
		i, 
		mymap[HEPEVT_Wrapper::first_child(i)],
		mymap[HEPEVT_Wrapper::last_child(i)] );
	}
	// M. Dobbs (from Borut, part B) - April 26, to fix tauolo/herwig past
	// the end problem with daughter pointers: 
	HEPEVT_Wrapper::set_number_entries( ilast );
    }

    void IO_HERWIG::zero_hepevt_entry( int i ) const {
      if ( i <=0 || i > HepMC::HEPEVT_Wrapper::max_number_entries() ) return;
      HEPEVT_Wrapper::set_status( i, 0 );
      HEPEVT_Wrapper::set_id( i, 0 );
      HEPEVT_Wrapper::set_parents( i, 0, 0 );
      HEPEVT_Wrapper::set_children( i, 0, 0 );
      HEPEVT_Wrapper::set_momentum( i, 0, 0, 0, 0 );
      HEPEVT_Wrapper::set_mass( i, 0 );
      HEPEVT_Wrapper::set_position( i, 0, 0, 0, 0 );
    }

    int IO_HERWIG::translate_herwig_to_pdg_id( int id ) const {
	// This routine is copied from Lynn Garren's stdhep 5.01.
	//   see http://www-pat.fnal.gov/stdhep.html
 
	                                       // example -9922212
	int hwtran = id;                       //         -9922212
	int ida    = abs(id);                  //          9922212
	int j1     = ida%10;                   //                2
	int i1     = (ida/10)%10;              //               1
	int i2     = (ida/100)%10;             //              2
	int i3     = (ida/1000)%10;            //             2
	//int i4     =(ida/10000)%10;          //            2
	//int i5     =(ida/100000)%10;         //           9
	//int k99    = (ida/100000)%100;       //          9
	int ksusy  = (ida/1000000)%10;         //         0
	//int ku     = (ida/10000000)%10;      //        0
	int kqn    = (ida/1000000000)%10;      //       0

	if ( kqn==1 ) {
	    //  ions not recognized
	    hwtran=0;
	    if ( m_print_inconsistency_errors ) {
		std::cerr << "IO_HERWIG::translate_herwig_to_pdg_id " << id
			  << "nonallowed ion" << std::endl;
	    }
	} 
	else if (ida < 100) {
	    // Higgs, etc.
	    hwtran = m_herwig_to_pdg_id[ida];
	    if ( id < 0 ) hwtran *= -1;
	    // check for illegal antiparticles
	    if ( id < 0 ) {
		if ( hwtran>=-99 && hwtran<=-81) hwtran=0;
		if ( m_no_antiparticles.count(hwtran) ) hwtran=0;
	    }
	}
	else if ( ksusy==1 || ksusy==2 ) { ; }
	//  SUSY
	else if ( i1!=0 && i3!=0 && j1==2 ) {;}
	// spin 1/2 baryons
	else if ( i1!=0 && i3!=0 && j1==4 ) {;}
	// spin 3/2 baryons
	else if ( i1!=0 && i2!=0 && i3==0 ) {
	    // mesons 
	    // check for illegal antiparticles
	    if ( i1==i2 && id<0) hwtran=0;
	} 
	else if ( i2!=0 && i3!=0 && i1==0 ) {;}
	// diquarks
	else {
	    // undefined
	    hwtran=0;
	}

	// check for illegal anti KS, KL
	if ( id==-130 || id==-310 ) hwtran=0;

	if ( hwtran==0 && ida!=0 && m_print_inconsistency_errors ) {
	    std::cerr 
		<< "IO_HERWIG::translate_herwig_to_pdg_id HERWIG particle " 
		<< id << " translates to zero." << std::endl;
	}

	return hwtran;
    }

} // HepMC




