//--------------------------------------------------------------------------
#ifndef HEPMC_IO_EPOS_H
#define HEPMC_IO_EPOS_H
//THIS IS A BRUTAL COPY OF EPOS g_Wrapper from HEPMC
//We need it because the EPOS generator needs a largeer version of EPOS g to store the event
#include <map>
#include <vector>
#include "HepMC/IO_BaseClass.h"
#include "GeneratorInterface/ReggeGribovPartonMCInterface/interface/EPOS_Wrapper.h"

namespace EPOS {
/*
    namespace HepMC{
      class GenEvent;
      class GenVertex;
      class GenParticle;
    }  
*/

    class IO_EPOS : public HepMC::IO_BaseClass {
    public:
        IO_EPOS();
        virtual           ~IO_EPOS();
        bool              fill_next_event( HepMC::GenEvent* );
        void              write_event( const HepMC::GenEvent* );
        void              print( std::ostream& ostr = std::cout ) const;
        
        // see comments below for these switches.
        bool              trust_both_mothers_and_daughters() const;
        bool              trust_mothers_before_daughters() const;
        bool              print_inconsistency_errors() const;
        bool              trust_beam_particles() const;
        void              set_trust_mothers_before_daughters( bool b = true );
        void              set_trust_both_mothers_and_daughters( bool b = false );
        void              set_print_inconsistency_errors( bool b = true );
        void              set_trust_beam_particles( bool b = true );

    protected: // for internal use only
        HepMC::GenParticle* build_particle( int index );
        void build_production_vertex( 
            int i,std::vector<HepMC::GenParticle*>& hepevt_particle, HepMC::GenEvent* evt );
        void build_end_vertex( 
            int i, std::vector<HepMC::GenParticle*>& hepevt_particle, HepMC::GenEvent* evt );
        int  find_in_map( 
            const std::map<HepMC::GenParticle*,int>& m, HepMC::GenParticle* p) const;

    private: // use of copy constructor is not allowed
        IO_EPOS( const IO_EPOS& ) : HepMC::IO_BaseClass() {}

    private: // data members

        bool m_trust_mothers_before_daughters;
        bool m_trust_both_mothers_and_daughters;
        bool m_print_inconsistency_errors; 
        bool m_trust_beam_particles;
    };

    // INLINES access methods //
    inline bool IO_EPOS::trust_both_mothers_and_daughters() const 
    { return m_trust_both_mothers_and_daughters; }
        
    inline bool IO_EPOS::trust_mothers_before_daughters() const 
    { return m_trust_mothers_before_daughters; }

    inline bool IO_EPOS::print_inconsistency_errors() const
    { return m_print_inconsistency_errors; }

    inline void IO_EPOS::set_trust_both_mothers_and_daughters( bool b )
    { m_trust_both_mothers_and_daughters = b; }

    inline void IO_EPOS::set_trust_mothers_before_daughters( bool b )
    { m_trust_mothers_before_daughters = b; }

    inline void IO_EPOS::set_print_inconsistency_errors( bool b  )
    { m_print_inconsistency_errors = b; }

    inline bool IO_EPOS::trust_beam_particles() const
    { return m_trust_beam_particles; }

    inline void IO_EPOS::set_trust_beam_particles( bool b )
    { m_trust_beam_particles = b; }

} // HepMC

#endif  // HEPMC_IO_EPOS_H
//--------------------------------------------------------------------------
