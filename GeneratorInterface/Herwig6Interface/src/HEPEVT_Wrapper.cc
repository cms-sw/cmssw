//////////////////////////////////////////////////////////////////////////
// Matt.Dobbs@Cern.CH       June 30, 2000
// Generic Wrapper for the fortran HEPEVT common block
// 
// The static data member's initializations must be separate from .h file.
// 
//////////////////////////////////////////////////////////////////////////

#include "HEPEVT_Wrapper.h"

namespace HepMC {

    ////////////////////////////////////////
    // static data member initializations //
    ////////////////////////////////////////

    unsigned int HEPEVT_Wrapper::s_sizeof_int = 4;

    unsigned int HEPEVT_Wrapper::s_sizeof_real = sizeof(double);

    unsigned int HEPEVT_Wrapper::s_max_number_entries = 4000;

    ///////////////////
    // Print Methods //
    ///////////////////

    void HEPEVT_Wrapper::print_hepevt( std::ostream& ostr ) 
    {
	// dumps the content of this HEPEVT event to ostr  (Width is 80)
	ostr << "________________________________________"
	     << "________________________________________" << std::endl;
	ostr << "***** HEPEVT Common Event#: " 
	     << event_number()
	     << ", " << number_entries() << " particles (max "
	     << max_number_entries() << ") *****";
	if ( is_double_precision() ) {
	    ostr << " Double Precision" << std::endl;
	} else {
	    ostr << " Single Precision" << std::endl;
	}
	ostr << sizeof_int() << "-byte integers, " 
	     << sizeof_real() << "-byte floating point numbers, "
	     << max_number_entries() << "-allocated entries." 
	     << std::endl;
	print_legend(ostr);
	ostr << "________________________________________"
	     << "________________________________________" << std::endl;
	for ( int i=1; i <= number_entries(); ++i ) {
	    print_hepevt_particle( i, ostr );
	}
	ostr << "________________________________________"
	     << "________________________________________" << std::endl;
    }

    void HEPEVT_Wrapper::print_legend( std::ostream& ostr )
    {
	char outline[81];
	sprintf( outline,"%4s %4s %4s %5s   %10s, %9s, %9s, %9s, %10s",
		 "Indx","Stat","Par-","chil-",
		 "(  P_x","P_y","P_z","Energy","M ) ");
	ostr << outline << std::endl;
	sprintf( outline,"%9s %4s %4s    %10s, %9s, %9s, %9s) %9s",
		 "ID ","ents","dren",
		 "Prod (   X","Y","Z","cT", "[mm]");
	ostr << outline << std::endl;
    }	

    void HEPEVT_Wrapper::print_hepevt_particle( int i, std::ostream& ostr ) 
    {
	// dumps the content HEPEVT particle entry i   (Width is 120)
	// here i is the C array index (i.e. it starts at 0 ... whereas the
	// fortran array index starts at 1) So if there's 100 particles, the
	// last valid index is 100-1=99
	char outline[81];
	sprintf( outline,
		 "%4d %+4d %4d %4d    (%9.3g, %9.3g, %9.3g, %9.3g, %9.3g)"
		 ,i, status(i), first_parent(i), first_child(i),
		 px(i), py(i), pz(i), e(i), m(i) );
	ostr << outline << "\n";
	sprintf( outline,"%+9d %4d %4d    (%9.3g, %9.3g, %9.3g, %9.3g)",
		 // old version was:" (%+9.2e, %+9.2e, %+9.2e, %+9.2e)"
		 id(i), last_parent(i), last_child(i), 
		 x(i), y(i), z(i), t(i) );
	ostr << outline << std::endl;
    }


    bool HEPEVT_Wrapper::check_hepevt_consistency( std::ostream& os )
    {
	// This method inspects the HEPEVT common block and looks for 
	// inconsistencies in the mother/daughter pointers
	bool isConsistent=true;
        char header[81];
        sprintf( header,
                 "\n\n\t**** WARNINGInconsistent HEPEVT input, Event %10d ****"
		 , HEPEVT_Wrapper::event_number() );

	for ( int i = 1; i <= HEPEVT_Wrapper::number_entries(); ++i ) {
	    // 1. check its mothers
	    int moth1 = HEPEVT_Wrapper::first_parent( i );
	    int moth2 = HEPEVT_Wrapper::last_parent( i );
	    if ( moth2<moth1 ) {
		if ( isConsistent ) {
		    os << header << std::endl;
		    isConsistent = false;
		    print_legend(os);
		}
		os << "Inconsistent entry " << i 
		   << " first parent > last parent " << std::endl;
		HEPEVT_Wrapper::print_hepevt_particle( i, os );
	    }
	    for ( int m = moth1; m<=moth2 && m!=0; ++m ) {
		if ( m>HEPEVT_Wrapper::number_entries() || m < 0 ) {
		    if ( isConsistent ) {
			os << header << std::endl;
			isConsistent = false;
			print_legend(os);
		    }
		    os << "Inconsistent entry " << i 
		       << " mother points out of range " << std::endl;
		    HEPEVT_Wrapper::print_hepevt_particle( i, os );
		}
		int mChild1 = HEPEVT_Wrapper::first_child(m);
		int mChild2 = HEPEVT_Wrapper::last_child(m);
		// we don't consider null pointers as inconsistent
		if ( mChild1==0 && mChild2==0 ) continue;
		if ( i<mChild1 || i>mChild2 ) {
		    if ( isConsistent ) {
			os << header << std::endl;
			isConsistent = false;
			print_legend(os);
		    }
		    os << "Inconsistent mother-daughter relationship between "
		       << i << " & " << m 
		       << " (try !trust_mother)" << std::endl;
		    HEPEVT_Wrapper::print_hepevt_particle( i, os );
		    HEPEVT_Wrapper::print_hepevt_particle( m, os );
		}
	    }
	    // 2. check its daughters
	    int dau1 = HEPEVT_Wrapper::first_child( i );
	    int dau2 = HEPEVT_Wrapper::last_child( i );
	    if ( dau2<dau1 ) {
		if ( isConsistent ) {
		    os << header << std::endl;
		    isConsistent = false;
		    print_legend(os);
		}
		os << "Inconsistent entry " << i 
		   << " first child > last child " << std::endl;
		HEPEVT_Wrapper::print_hepevt_particle( i, os );
	    }
	    for ( int d = dau1; d<=dau2 && d!=0; ++d ) {
		if ( d>HEPEVT_Wrapper::number_entries() || d < 0 ) {
		    if ( isConsistent ) {
			os << header << std::endl;
			isConsistent = false;
			print_legend(os);
		    }
		    os << "Inconsistent entry " << i 
		       << " child points out of range " << std::endl;
		    HEPEVT_Wrapper::print_hepevt_particle( i, os );
		}
		int d_moth1 = HEPEVT_Wrapper::first_parent(d);
		int d_moth2 = HEPEVT_Wrapper::last_parent(d);
		// we don't consider null pointers as inconsistent
		if ( d_moth1==0 && d_moth2==0 ) continue;
		if ( i<d_moth1 || i>d_moth2 ) {
		    if ( isConsistent ) {
			os << header << std::endl;
			isConsistent = false;
			print_legend(os);
		    }
		    os << "Inconsistent mother-daughter relationship between "
		       << i << " & " << d 
		       << " (try trust_mothers)"<< std::endl;
		    HEPEVT_Wrapper::print_hepevt_particle( i, os );
		    HEPEVT_Wrapper::print_hepevt_particle( d, os );
		}
	    }
	}
	if (!isConsistent) {
	    os << "Above lists all the inconsistencies in the HEPEVT common "
	       << "\n block which has been provided as input to HepMC. "
	       << "\n HepMC WILL have trouble interpreting the mother-daughter"
	       << "\n relationships ... but all other information "
	       << "\n (4-vectors etc) will be correctly transferred."
	       << "\n In order for HepMC to be able to interpret the mother/"
	       << "\n daughter hierachy, it MUST be given consistent input."
	       << "\n This is one of the design criteria of HepMC: "
	       << "\n consistency is enforced by the code.";
	    os << "\nThere is a switch in IO_HEPEVT, set-able using "
	       << "\n IO_HEPEVT::set_trust_mothers_before_daughters( bool )"
	       << "\n which you may want to try.";
	    os << "\nNote: if HEPEVT common block has been filled by pythia"
	       << "\n pyhepc, then the switch MSTP(128)=2 should be used in"
	       << "\n pythia, which instructs pythia not to put multiple "
	       << "\n copies of resonances in the event record.\n";
	    os << "To obtain a file summarizing the inconsistency, you should:"
	       << "\n\t ofstream myFile(\"myInconsistentEvent.txt\"); "
	       << "\n\t HEPEVT_Wrapper::check_hepevt_consistency(myFile); "
	       << "\n\t HEPEVT_Wrapper::print_hepevt(myFile); "
	       << "\n[now write the event to HepMC using something like"
	       << "\n\t\t myIO_HEPEVT->write_event(myEvent); ]"
	       << "\n\t myEvent->print( myFile ); "
	       << "      // print event as HepMC sees it"
	       << "\n ------------------------- Thank-you. \n\n" << std::endl;
	}
	return isConsistent;
    }

    void HEPEVT_Wrapper::zero_everything()
    {
	set_event_number( 0 );
        set_number_entries( 0 );
	for ( int i = 1; i<=max_number_entries(); ++i ) {
	    set_status( i, 0 );
	    set_id( i, 0 );
	    set_parents( i, 0, 0 );
	    set_children( i, 0, 0 );
	    set_momentum( i, 0, 0, 0, 0 );
	    set_mass( i, 0 );
	    set_position( i, 0, 0, 0, 0 );
	}
    }

} // HepMC

