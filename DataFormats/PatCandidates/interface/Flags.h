#ifndef DataFormats_PatCandidates_interface_Flags_h
#define DataFormats_PatCandidates_interface_Flags_h
//
// $Id: Flags.h,v 1.4 2008/12/05 20:55:50 hegner Exp $
//

/**
  \class    pat::Flags Flag.h "DataFormats/PatCandidates/interface/Flags.h"
  \brief    Flags used in PAT, and static translator from flags to strings 

  \author   Steven Lowette
  \version  $Id: Flags.h,v 1.4 2008/12/05 20:55:50 hegner Exp $
*/

#include "DataFormats/Candidate/interface/Candidate.h"
#include <string>
#include <vector>
#include <boost/cstdint.hpp> 

namespace pat {
    struct Flags {
        enum CleanerFlags {
            AllBits       = 0xFFFFFFFF,
            CoreBits      = 0x0000000F,
            SelectionBits = 0x0000FFF0,
            OverlapBits   = 0x00FF0000,
            IsolationBits = 0xFF000000

        };
        inline static bool test(uint32_t val, uint32_t mask) { return (val & mask) == 0; }
        inline static bool test(const reco::Candidate &c, uint32_t mask) { return test(c.status(), mask); }

        static const std::string & bitToString( uint32_t bit );
        static std::string maskToString( uint32_t bit );
        static uint32_t  get ( const std::string & str );
        static uint32_t  get ( const std::vector<std::string> & str );

        struct Core {
            enum { Shift =  0 };
            enum Bits {
                All           = 0x0000000F,
                Duplicate     = 0x00000001, // internal duplication
                Preselection  = 0x00000002, // base preselection 1 (e.g. pt, eta cuts)
                Vertexing     = 0x00000004, // vertex association cuts
                Overflow      = 0x00000008, // if one requests to save "at most X items", 
                                            // the overflowing ones will have this bit set
                Undefined     = 0x00000000
            };
            static const std::string & bitToString( Bits bit );
            static Bits      get ( const std::string & str );
            static uint32_t  get ( const std::vector<std::string> & str );
        };

        struct Overlap {
            enum { Shift =  16 };
            enum Bits {
                All       = 0x00FF0000,
                Jets      = 0x00010000,
                Electrons = 0x00020000,
                Muons     = 0x00040000,
                Taus      = 0x00080000,
                Photons   = 0x00100000,
                User      = 0X00E00000,
                User1     = 0x00200000,
                User2     = 0x00400000,
                User3     = 0x00800000,
                Undefined = 0x00000000
            };
            static const std::string & bitToString( Bits bit );
            static Bits      get ( const std::string & str );
            static uint32_t  get ( const std::vector<std::string> & str );
        };

        struct Selection {
            enum { Shift =  4 };
            enum Bits {
                All       = 0x0000FFF0,
                Bit0      = 0x00000010, 
                Bit1      = 0x00000020, 
                Bit2      = 0x00000040, 
                Bit3      = 0x00000080, 
                Bit4      = 0x00000100,
                Bit5      = 0x00000200,
                Bit6      = 0x00000400,
                Bit7      = 0x00000800,
                Bit8      = 0x00001000,
                Bit9      = 0x00002000,
                Bit10     = 0x00004000,
                Bit11     = 0x00008000,
                Undefined = 0x00000000
            };
            static const std::string & bitToString( Bits bit );
            static Bits     get ( int8_t bit );
            static Bits     get ( const std::string & str );
            static uint32_t get ( const std::vector<std::string> & str );
        };
        struct Isolation {
            enum { Shift =  24 };
            enum Bits {
                All       = 0xFF000000,
                Tracker   = 0x01000000,
                ECal      = 0x02000000,
                HCal      = 0x04000000,
                Calo      = 0x06000000,
                User      = 0xF8000000,
                User1     = 0x08000000,
                User2     = 0x10000000,
                User3     = 0x20000000,
                User4     = 0x40000000,
                User5     = 0x80000000,
                Undefined = 0x00000000
            };
            static const std::string & bitToString( Bits bit );
            static Bits      get ( const std::string & str );
            static uint32_t  get ( const std::vector<std::string> & str );
        };
    };
}

#endif
