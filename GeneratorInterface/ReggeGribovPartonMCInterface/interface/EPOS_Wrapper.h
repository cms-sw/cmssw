//--------------------------------------------------------------------------
//THIS IS A BRUTAL COPY OF HEPEVT_Wrapper from HEPMC
//We need it because the EPOS generator needs a largeer version of HEPEVT to store the event
#ifndef EPOS_EntriesAllocation
#define EPOS_EntriesAllocation 99900
#endif  // EPOS_EntriesAllocation

//--------------------------------------------------------------------------
#ifndef HEPMC_EPOS_COMMON_H
#define HEPMC_EPOS_COMMON_H
//////////////////////////////////////////////////////////////////////////
//
//      PARAMETER (NMXHEP=2000) 
//      COMMON/HEPCOM/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP), 
//     &        JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* NEVHEP          - event number (or some special meaning*/
/*                    (see documentation for details)     */
/* NHEP            - actual number of entries in current  */
/*                    event.                              */
/* ISTHEP[IHEP]    - status code for IHEP'th entry - see  */
/*                    documentation for details           */
/* IDHEP [IHEP]    - IHEP'th particle identifier according*/
/*                    to PDG.                             */
/* JMOHEP[IHEP][0] - pointer to position of 1st mother    */
/* JMOHEP[IHEP][1] - pointer to position of 2nd mother    */
/* JDAHEP[IHEP][0] - pointer to position of 1st daughter  */
/* JDAHEP[IHEP][1] - pointer to position of 2nd daughter  */
/* PHEP  [IHEP][0] - X momentum                           */
/* PHEP  [IHEP][1] - Y momentum                           */
/* PHEP  [IHEP][2] - Z momentum                           */
/* PHEP  [IHEP][3] - Energy                               */
/* PHEP  [IHEP][4] - Mass                                 */
/* VHEP  [IHEP][0] - X vertex                             */
/* VHEP  [IHEP][1] - Y vertex                             */
/* VHEP  [IHEP][2] - Z vertex                             */
/* VHEP  [IHEP][3] - production time                      */
/*========================================================*/
// Remember, array(1) is the first entry in a fortran array, array[0] is the
//           first entry in a C array.
//
// This interface to EPOS common block treats the block as
// an array of bytes --- the precision and number of entries 
// is determined "on the fly" by the wrapper and used to decode
// each entry.
//
// EPOS_EntriesAllocation is the maximum size of the EPOS common block 
//   that can be interfaced.
//   It is NOT the actual size of the EPOS common used in each
//   individual application. The actual size can be changed on
//   the fly using EPOS_Wrapper::set_max_number_entries().
// Thus EPOS_EntriesAllocation should typically be set
// to the maximum possible number of entries --- 10000 is a good choice
// (and is the number used by ATLAS versions of Pythia).
//
// Note: a statement like    *( (int*)&hepcom.data[0] )
//      takes the memory address of the first byte in EPOS,
//      interprets it as an integer pointer, 
//      and dereferences the pointer.
//      i.e. it returns an integer corresponding to nevhep
//

#include <ctype.h>

    const unsigned int epos_bytes_allocation = 
                sizeof(long int) * ( 2 + 6 * EPOS_EntriesAllocation )
                + sizeof(double) * ( 9 * EPOS_EntriesAllocation );


#ifdef _WIN32 // Platform: Windows MS Visual C++
struct HEPCOM_DEF{
        char data[epos_bytes_allocation];
    };
extern "C" HEPCOM_DEF HEPCOM;
#define hepcom HEPCOM

#else
extern "C" {
    extern struct {
	char data[epos_bytes_allocation];
    } hepcom_;
}
#define hepcom hepcom_

#endif // Platform

#endif  // HEPMC_EPOS_COMMON_H

//--------------------------------------------------------------------------
#ifndef HEPMC_EPOS_WRAPPER_H
#define HEPMC_EPOS_WRAPPER_H

//////////////////////////////////////////////////////////////////////////
// Matt.Dobbs@Cern.CH, April 24, 2000, refer to:
// M. Dobbs and J.B. Hansen, "The HepMC C++ Monte Carlo Event Record for
// High Energy Physics", Computer Physics Communications (to be published).
//
// Generic Wrapper for the fortran EPOS common block
// This class is intended for static use only - it makes no sense to 
// instantiate it.
// Updated: June 30, 2000 (static initialization moved to separate .cxx file)
//////////////////////////////////////////////////////////////////////////
//
// The index refers to the fortran style index: 
// i.e. index=1 refers to the first entry in the EPOS common block.
// all indices must be >0
// number_entries --> integer between 0 and max_number_entries() giving total
//                    number of sequential particle indices
// first_parent/child --> index of first mother/child if there is one, 
//                        zero otherwise
// last_parent/child --> if number children is >1, address of last parent/child
//                       if number of children is 1, same as first_parent/child
//                       if there are no children, returns zero.
// is_double_precision --> T or F depending if floating point variables 
//                         are 8 or 4 bytes
//

#include <iostream>
#include <cstdio>       // needed for formatted output using sprintf 

namespace EPOS {

    //! Generic Wrapper for the fortran EPOS common block
    
    /// \class EPOS_Wrapper
    /// This class is intended for static use only - it makes no sense to 
    /// instantiate it.
    ///
    class EPOS_Wrapper {
    public:

        /// write information from EPOS common block
	static void print_hepcom( std::ostream& ostr = std::cout );
	/// write particle information to ostr
	static void print_hepcom_particle( int index, 
					   std::ostream& ostr = std::cout );
        /// set all entries in EPOS to zero
	static void zero_everything();

	////////////////////
	// Access Methods //
	////////////////////
        static int    event_number();             //!< event number
        static int    number_entries();           //!< num entries in current evt
        static int    status( int index );        //!< status code
        static int    id( int index );            //!< PDG particle id
        static int    first_parent( int index );  //!< index of 1st mother
        static int    last_parent( int index );   //!< index of last mother
	static int    number_parents( int index ); //!< number of parents 
        static int    first_child( int index );   //!< index of 1st daughter
        static int    last_child( int index );    //!< index of last daughter
	static int    number_children( int index ); //!< number of children
        static double px( int index );            //!< X momentum       
        static double py( int index );            //!< Y momentum   
        static double pz( int index );            //!< Z momentum   
        static double e( int index );             //!< Energy
        static double m( int index );             //!< generated mass
        static double x( int index );             //!< X Production vertex
        static double y( int index );             //!< Y Production vertex
        static double z( int index );             //!< Z Production vertex
        static double t( int index );             //!< production time

	////////////////////
	// Set Methods    //
	////////////////////

	/// set event number
        static void set_event_number( int evtno );
	/// set number of entries in EPOS
        static void set_number_entries( int noentries );
	/// set particle status
        static void set_status( int index, int status );
	/// set particle ID
        static void set_id( int index, int id );
	/// define parents of a particle
        static void set_parents( int index, int firstparent, int lastparent );
	/// define children of a particle
        static void set_children( int index, int firstchild, int lastchild );
	/// set particle momentum
        static void set_momentum( int index, double px, double py,
				  double pz, double e );
	/// set particle mass
        static void set_mass( int index, double mass );
	/// set particle production vertex
        static void set_position( int index, double x, double y, double z, 
				  double t );
	//////////////////////
	// EPOS Floorplan //
	//////////////////////
	static unsigned int sizeof_int();  //!< size of integer in bytes
	static unsigned int sizeof_real(); //!< size of real in bytes
        static int  max_number_entries();  //!< size of common block
	static void set_sizeof_int(unsigned int);  //!< define size of integer
	static void set_sizeof_real(unsigned int); //!< define size of real
	static void set_max_number_entries(unsigned int); //!< define size of common block

    protected:
        /// navigate a byte array
	static double byte_num_to_double( unsigned int );
        /// navigate a byte array
	static int    byte_num_to_int( unsigned int );
        /// pretend common block is an array of bytes
	static void   write_byte_num( double, unsigned int );
        /// pretend common block is an array of bytes
	static void   write_byte_num( int, unsigned int );
	/// print output legend
	static void   print_legend( std::ostream& ostr = std::cout );

    private:
	static unsigned int s_sizeof_int;
	static unsigned int s_sizeof_real;
	static unsigned int s_max_number_entries;

    }; 

    //////////////////////////////
    // EPOS Floorplan Inlines //
    //////////////////////////////
    inline unsigned int EPOS_Wrapper::sizeof_int(){ return s_sizeof_int; }

    inline unsigned int EPOS_Wrapper::sizeof_real(){ return s_sizeof_real; }

    inline int EPOS_Wrapper::max_number_entries() 
    { return (int)s_max_number_entries; }

    inline void EPOS_Wrapper::set_sizeof_int( unsigned int size ) 
    {
	if ( size != sizeof(short int) && size != sizeof(long int) && size != sizeof(int) ) {
	    std::cerr << "HepMC is not able to handle integers "
		      << " of size other than 2 or 4."
		      << " You requested: " << size << std::endl;
	}
	s_sizeof_int = size;
    }

    inline void EPOS_Wrapper::set_sizeof_real( unsigned int size ) {
	if ( size != sizeof(float) && size != sizeof(double) ) {
	    std::cerr << "HepMC is not able to handle floating point numbers"
		      << " of size other than 4 or 8."
		      << " You requested: " << size << std::endl;
	}
	s_sizeof_real = size;
    }

    inline void EPOS_Wrapper::set_max_number_entries( unsigned int size ) {
	s_max_number_entries = size;
    }

    inline double EPOS_Wrapper::byte_num_to_double( unsigned int b ) {
	if ( b >= epos_bytes_allocation ) std::cerr 
		  << "EPOS_Wrapper: requested hepcom data exceeds allocation"
		  << std::endl;
	if ( s_sizeof_real == sizeof(float) ) {
	    float* myfloat = (float*)&hepcom.data[b];
	    return (double)(*myfloat);
	} else if ( s_sizeof_real == sizeof(double) ) {
	    double* mydouble = (double*)&hepcom.data[b];
	    return (*mydouble);
	} else {
	    std::cerr 
		<< "EPOS_Wrapper: illegal floating point number length." 
		<< s_sizeof_real << std::endl;
	}
	return 0;
    }

    inline int EPOS_Wrapper::byte_num_to_int( unsigned int b ) {
	if ( b >= epos_bytes_allocation ) std::cerr 
		  << "EPOS_Wrapper: requested hepcom data exceeds allocation"
		  << std::endl;
	if ( s_sizeof_int == sizeof(short int) ) {
	    short int* myshortint = (short int*)&hepcom.data[b];
	    return (int)(*myshortint);
	} else if ( s_sizeof_int == sizeof(long int) ) {
	    long int* mylongint = (long int*)&hepcom.data[b];
	    return (*mylongint);
       // on some 64 bit machines, int, short, and long are all different
	} else if ( s_sizeof_int == sizeof(int) ) {
	    int* myint = (int*)&hepcom.data[b];
	    return (*myint);
	} else {
	    std::cerr 
		<< "EPOS_Wrapper: illegal integer number length." 
		<< s_sizeof_int << std::endl;
	}
	return 0;
    }

    inline void EPOS_Wrapper::write_byte_num( double in, unsigned int b ) {
	if ( b >= epos_bytes_allocation ) std::cerr 
		  << "EPOS_Wrapper: requested hepcom data exceeds allocation"
		  << std::endl;
	if ( s_sizeof_real == sizeof(float) ) {
	    float* myfloat = (float*)&hepcom.data[b];
	    (*myfloat) = (float)in;
	} else if ( s_sizeof_real == sizeof(double) ) {
	    double* mydouble = (double*)&hepcom.data[b];
	    (*mydouble) = (double)in;
	} else {
	    std::cerr 
		<< "EPOS_Wrapper: illegal floating point number length." 
		<< s_sizeof_real << std::endl;
	}
    }

    inline void EPOS_Wrapper::write_byte_num( int in, unsigned int b ) {
	if ( b >= epos_bytes_allocation ) std::cerr 
		  << "EPOS_Wrapper: requested hepcom data exceeds allocation"
		  << std::endl;
	if ( s_sizeof_int == sizeof(short int) ) {
	    short int* myshortint = (short int*)&hepcom.data[b];
	    (*myshortint) = (short int)in;
	} else if ( s_sizeof_int == sizeof(long int) ) {
	    long int* mylongint = (long int*)&hepcom.data[b];
	    (*mylongint) = (int)in;
       // on some 64 bit machines, int, short, and long are all different
	} else if ( s_sizeof_int == sizeof(int) ) {
	    int* myint = (int*)&hepcom.data[b];
	    (*myint) = (int)in;
	} else {
	    std::cerr 
		<< "EPOS_Wrapper: illegal integer number length." 
		<< s_sizeof_int << std::endl;
	}
    }

    //////////////
    // INLINES  //
    //////////////

    inline int EPOS_Wrapper::event_number()
    { return byte_num_to_int(0); }

    inline int EPOS_Wrapper::number_entries() 
    { 
	int nhep = byte_num_to_int( 1*sizeof_int() );
	return ( nhep <= max_number_entries() ?
		 nhep : max_number_entries() );
    }

    inline int EPOS_Wrapper::status( int index )   
    { return byte_num_to_int( (2+index-1) * sizeof_int() ); }

    inline int EPOS_Wrapper::id( int index )
    { 
	return byte_num_to_int( (2+max_number_entries()+index-1) 
				* sizeof_int() ); 
    }

    inline int EPOS_Wrapper::first_parent( int index )
    { 
	int parent = byte_num_to_int( (2+2*max_number_entries()+2*(index-1)) 
				      * sizeof_int() ); 
	return ( parent > 0 && parent <= number_entries() ) ?
					 parent : 0; 
    }

    inline int EPOS_Wrapper::last_parent( int index )
    { 
	// Returns the Index of the LAST parent in the EPOS record
	// for particle with Index index.
	// If there is only one parent, the last parent is forced to 
	// be the same as the first parent.
	// If there are no parents for this particle, both the first_parent
	// and the last_parent with return 0.
	// Error checking is done to ensure the parent is always
	// within range ( 0 <= parent <= nhep )
	//
	int firstparent = first_parent(index);
	int parent = byte_num_to_int( (2+2*max_number_entries()+2*(index-1)+1) 
				      * sizeof_int() ); 
	return ( parent > firstparent && parent <= number_entries() ) 
						   ? parent : firstparent; 
    }

    inline int EPOS_Wrapper::number_parents( int index ) {
	int firstparent = first_parent(index);
	return ( firstparent>0 ) ? 
	    ( 1+last_parent(index)-firstparent ) : 0;
    }

    inline int EPOS_Wrapper::first_child( int index )
    { 
	int child = byte_num_to_int( (2+4*max_number_entries()+2*(index-1)) 
				     * sizeof_int() ); 
	return ( child > 0 && child <= number_entries() ) ?
				       child : 0; 
    }

    inline int EPOS_Wrapper::last_child( int index )
    { 
	// Returns the Index of the LAST child in the EPOS record
	// for particle with Index index.
	// If there is only one child, the last child is forced to 
	// be the same as the first child.
	// If there are no children for this particle, both the first_child
	// and the last_child with return 0.
	// Error checking is done to ensure the child is always
	// within range ( 0 <= parent <= nhep )
	//
	int firstchild = first_child(index);
	int child = byte_num_to_int( (2+4*max_number_entries()+2*(index-1)+1) 
				     * sizeof_int() ); 
	return ( child > firstchild && child <= number_entries() ) 
						? child : firstchild;
    }

    inline int EPOS_Wrapper::number_children( int index ) 
    {
	int firstchild = first_child(index);
	return ( firstchild>0 ) ? 
	    ( 1+last_child(index)-firstchild ) : 0;
    }

    inline double EPOS_Wrapper::px( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				 + (5*(index-1)+0) *sizeof_real() );
    }

    inline double EPOS_Wrapper::py( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				 + (5*(index-1)+1) *sizeof_real() );
    }


    inline double EPOS_Wrapper::pz( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				 + (5*(index-1)+2) *sizeof_real() );
    }

    inline double EPOS_Wrapper::e( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				 + (5*(index-1)+3) *sizeof_real() );
    }

    inline double EPOS_Wrapper::m( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				 + (5*(index-1)+4) *sizeof_real() );
    }

    inline double EPOS_Wrapper::x( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				   + ( 5*max_number_entries()
				       + (4*(index-1)+0) ) *sizeof_real() );
    }

    inline double EPOS_Wrapper::y( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				   + ( 5*max_number_entries()
				       + (4*(index-1)+1) ) *sizeof_real() );
    }

    inline double EPOS_Wrapper::z( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				   + ( 5*max_number_entries()
				       + (4*(index-1)+2) ) *sizeof_real() );
    }

    inline double EPOS_Wrapper::t( int index )
    { 
	return byte_num_to_double( (2+6*max_number_entries())*sizeof_int()
				   + ( 5*max_number_entries()
				       + (4*(index-1)+3) ) *sizeof_real() );
    }

    inline void EPOS_Wrapper::set_event_number( int evtno ) 
    { write_byte_num( evtno, 0 ); }

    inline void EPOS_Wrapper::set_number_entries( int noentries ) 
    { write_byte_num( noentries, 1*sizeof_int() ); }

    inline void EPOS_Wrapper::set_status( int index, int status ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	write_byte_num( status, (2+index-1) * sizeof_int() );
    }

    inline void EPOS_Wrapper::set_id( int index, int id ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	write_byte_num( id, (2+max_number_entries()+index-1) *sizeof_int() );
    }

    inline void EPOS_Wrapper::set_parents( int index, int firstparent, 
					     int lastparent ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	write_byte_num( firstparent, (2+2*max_number_entries()+2*(index-1)) 
			             *sizeof_int() );
	write_byte_num( lastparent, (2+2*max_number_entries()+2*(index-1)+1) 
				    * sizeof_int() );
    }
    
    inline void EPOS_Wrapper::set_children( int index, int firstchild, 
					      int lastchild ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	write_byte_num( firstchild, (2+4*max_number_entries()+2*(index-1)) 
				     *sizeof_int() );
	write_byte_num( lastchild, (2+4*max_number_entries()+2*(index-1)+1) 
				    *sizeof_int() );
    }

    inline void EPOS_Wrapper::set_momentum( int index, double px, 
					      double py, double pz, double e ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	write_byte_num( px, (2+6*max_number_entries()) *sizeof_int()
			    + (5*(index-1)+0) *sizeof_real() );
	write_byte_num( py, (2+6*max_number_entries())*sizeof_int()
			    + (5*(index-1)+1) *sizeof_real() );
	write_byte_num( pz, (2+6*max_number_entries())*sizeof_int()
			    + (5*(index-1)+2) *sizeof_real() );
	write_byte_num( e,  (2+6*max_number_entries())*sizeof_int()
			    + (5*(index-1)+3) *sizeof_real() );
    }

    inline void EPOS_Wrapper::set_mass( int index, double mass ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	write_byte_num( mass, (2+6*max_number_entries())*sizeof_int()
			      + (5*(index-1)+4) *sizeof_real() );
    }

    inline void EPOS_Wrapper::set_position( int index, double x, double y,
					      double z, double t ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	write_byte_num( x, (2+6*max_number_entries())*sizeof_int()
			   + ( 5*max_number_entries()
			       + (4*(index-1)+0) ) *sizeof_real() );
	write_byte_num( y, (2+6*max_number_entries())*sizeof_int()
			   + ( 5*max_number_entries()
			       + (4*(index-1)+1) ) *sizeof_real() );
	write_byte_num( z, (2+6*max_number_entries())*sizeof_int()
			   + ( 5*max_number_entries()
			       + (4*(index-1)+2) ) *sizeof_real() );
	write_byte_num( t, (2+6*max_number_entries())*sizeof_int()
			   + ( 5*max_number_entries()
			       + (4*(index-1)+3) ) *sizeof_real() );
    }

    inline void EPOS_Wrapper::zero_everything()
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

    inline void EPOS_Wrapper::print_hepcom( std::ostream& ostr ) 
    {
        ostr << "________________________________________"
             << "________________________________________" << std::endl;
        ostr << "***** HEPEVT Common Event#: " 
             << event_number()
             << ", " << number_entries() << " particles (max "
             << max_number_entries() << ") *****";
        ostr << sizeof_int() << "-byte integers, " 
             << sizeof_real() << "-byte floating point numbers, "
             << max_number_entries() << "-allocated entries." 
             << std::endl;
        print_legend(ostr);
        ostr << "________________________________________"
             << "________________________________________" << std::endl;
        for ( int i=1; i <= number_entries(); ++i ) {
             print_hepcom_particle( i, ostr );
        }
        ostr << "________________________________________"
             << "________________________________________" << std::endl;
    } 

    inline void EPOS_Wrapper::print_hepcom_particle( int i, std::ostream& ostr ) 
    {
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
    
    inline void EPOS_Wrapper::print_legend( std::ostream& ostr )
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

} // HepMC

#endif  // HEPMC_EPOS_WRAPPER_H
//--------------------------------------------------------------------------

