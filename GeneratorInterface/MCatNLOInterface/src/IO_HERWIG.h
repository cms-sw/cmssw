//--------------------------------------------------------------------------
#ifndef HEPMC_IO_HERWIG_H
#define HEPMC_IO_HERWIG_H

//////////////////////////////////////////////////////////////////////////
// Matt.Dobbs@Cern.CH, October 2002, refer to:
// M. Dobbs and J.B. Hansen, "The HepMC C++ Monte Carlo Event Record for
// High Energy Physics", Computer Physics Communications (to be published).
//
// IO class for reading the (non-standard) HEPEVT common block from 
//  the Herwig monte carlo program.
// Notes:
//   - The HERWIG HEPEVT common block is non-standard, primarily because it 
//     contains some color flow information. When you call IO_HERWIG, the 
//     HEPEVT common block is transformed to the standard. THIS CHANGES THE
//     CONTENT of HEPEVT!.
//   - The HERWIG HEPEVT common block has some EXTRA non-physical ENTRIES 
//     (such as CMS frame, HARD subprocess, and CONE).
//     These are removed by IO_HERWIG. Thus the HepMC event will APPEAR
//     to have fewer particles in it that herwig did.
//     There is a switch m_no_gaps_in_barcodes. For
//       true  - then the extra particles are removed from HEPEVT, with 
//               the result that the HepMC barcodes will be sequential, with 
//               no gaps.
//       false - the barcodes will correspond directly to the HEPEVT index, but
//               there will be gaps ... ie some barcodes will be unassigned.
//       this switch requested by I Hinchliffe, October 31, 2002
//   - some of the Herwig GLUON SPLITTING products are not properly documented
//     in hepevt. I was unable to repair this in a simple and robust way. 
//     Therefore some of the gluon splitting products will be orphans 
//     in the HepMC output. 
//   - Herwig uses      HEPEVT_Wrapper::set_max_number_entries(4000);
//  	                HEPEVT_Wrapper::set_sizeof_real(8);
//     which are the defaults for HEPEVT_Wrapper.
//////////////////////////////////////////////////////////////////////////
//

#include <set>
#include <vector>
#include "IO_BaseClass.h"
#include "HEPEVT_Wrapper.h"

namespace HepMC {

    class GenEvent;
    class GenVertex;
    class GenParticle;
    class ParticleDataTable;

    class IO_HERWIG : public IO_BaseClass {
    public:
	IO_HERWIG();
	virtual           ~IO_HERWIG();
	bool              fill_next_event( GenEvent* );
	void              print( std::ostream& ostr = std::cout ) const;
	double            interfaces_to_version_number() const {return 6.400;}
		
	// see comments below for these switches.
	bool              print_inconsistency_errors() const;
	void              set_print_inconsistency_errors( bool b = 1 );

	bool              no_gaps_in_barcodes() const 
	                     { return m_no_gaps_in_barcodes; }
	void              set_no_gaps_in_barcodes( bool a ) 
	                     { m_no_gaps_in_barcodes=a; }

    protected: // for internal use only
	bool              trust_both_mothers_and_daughters() const;
	bool              trust_mothers_before_daughters() const;
	void              set_trust_mothers_before_daughters( bool b = 1 );
	void              set_trust_both_mothers_and_daughters( bool b = 0 );

	GenParticle* build_particle( int index );
	void         build_production_vertex( 
	    int i,std::vector<GenParticle*>& hepevt_particle, GenEvent* evt );
	void         build_end_vertex( 
	    int i, std::vector<GenParticle*>& hepevt_particle, GenEvent* evt );
	int          find_in_map( 
	    const std::map<GenParticle*,int>& m, GenParticle* p) const;

	void repair_hepevt() const;
	void remove_gaps_in_hepevt() const;
	void zero_hepevt_entry( int i ) const;
	int  translate_herwig_to_pdg_id( int i ) const;

    private: // following are not implemented for Herwig
	virtual void write_event( const GenEvent* ){}
        virtual void write_particle_data_table( const ParticleDataTable* ){}
        virtual bool fill_particle_data_table( ParticleDataTable* )
	    { return 0; }

    private: // use of copy constructor is not allowed
	IO_HERWIG( const IO_HERWIG& ) : IO_BaseClass() {}

    private: // data members
	bool              m_trust_mothers_before_daughters;
	bool              m_trust_both_mothers_and_daughters;
	bool              m_print_inconsistency_errors; 
	bool              m_no_gaps_in_barcodes;
	std::vector<int>  m_herwig_to_pdg_id;
	std::set<int>     m_no_antiparticles;
    };

    ////////////////////////////
    // INLINES access methods //
    ////////////////////////////
    inline bool IO_HERWIG::trust_both_mothers_and_daughters() const 
    { return m_trust_both_mothers_and_daughters; }
	
    inline bool IO_HERWIG::trust_mothers_before_daughters() const 
    { return m_trust_mothers_before_daughters; }

    inline bool IO_HERWIG::print_inconsistency_errors() const
    { return m_print_inconsistency_errors; }

    inline void IO_HERWIG::set_trust_both_mothers_and_daughters( bool b )
    { m_trust_both_mothers_and_daughters = b; }

    inline void IO_HERWIG::set_trust_mothers_before_daughters( bool b )
    { m_trust_mothers_before_daughters = b; }

    inline void IO_HERWIG::set_print_inconsistency_errors( bool b  )
    { m_print_inconsistency_errors = b; }

} // HepMC

#endif  // HEPMC_IO_HERWIG_H
//--------------------------------------------------------------------------
