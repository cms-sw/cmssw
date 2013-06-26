#ifndef CondFormats_JetMETObjects_FFTJetCorrTypes_h
#define CondFormats_JetMETObjects_FFTJetCorrTypes_h

// FFTJet jet correction types. These types are essentially just
// labels for ES records -- all ES record types must be unique.
//
// Why do we need this many correction types, there are, after all,
// only several jet types in the system? The problem is really with
// the event setup (ES) record dependency tracking. The dependency
// tracking works for types only, there appears to be no way to add
// record labels to the ES dependency tracking mechanism. It would
// be very nice if we could say that record of type R1 with label L1
// depends on record of type R2 with label L2. Then we could use only
// the types we really need for R1 and R2 and use labels to distinguish
// different records of the same type. Alas, things are not designed
// this way -- we can only say that type R1 depends on type R2.
// This means that we need to create as many R1 and R2 types as the
// number of distinct R1 records one expects to use, even though all
// records of type R2 could be associated with the same underlying
// data structure type.
//
// Of course, to actually create these records one would use
// a templated ESProducer, and the corresponding "produce" method
// (or the whole producer class) would have to be instantiated for
// every record type if we want to avoid changing the code and
// recompiling things every time we want to add a new jet correction.
// Such a nuisance.
//
// Igor Volobouev
// Aug 3, 2012

namespace fftcorrtypes
{
    //
    // Fundamental records for jet correction factors
    //
    struct BasicJet
    {
        enum {value = 0};
        inline static const char* classname() {return "BasicJet";}
    };

    struct GenJet
    {
        enum {value = 1};
        inline static const char* classname() {return "GenJet";}
    };

    struct CaloJet
    {
        enum {value = 2};
        inline static const char* classname() {return "CaloJet";}
    };

    struct PFJet
    {
        enum {value = 3};
        inline static const char* classname() {return "PFJet";}
    };

    struct TrackJet
    {
        enum {value = 4};
        inline static const char* classname() {return "TrackJet";}
    };

    struct JPTJet
    {
        enum {value = 5};
        inline static const char* classname() {return "JPTJet";}
    };

    struct PFCHS0
    {
        enum {value = 6};
        inline static const char* classname() {return "PFCHS0";}
    };

    struct PFCHS1
    {
        enum {value = 7};
        inline static const char* classname() {return "PFCHS1";}
    };

    struct PFCHS2
    {
        enum {value = 8};
        inline static const char* classname() {return "PFCHS2";}
    };

    //
    // Fundamental records for jet correction systematic errors
    //
    struct BasicJetSys
    {
        enum {value = 9};
        inline static const char* classname() {return "BasicJetSys";}
    };

    struct GenJetSys
    {
        enum {value = 10};
        inline static const char* classname() {return "GenJetSys";}
    };

    struct CaloJetSys
    {
        enum {value = 11};
        inline static const char* classname() {return "CaloJetSys";}
    };

    struct PFJetSys
    {
        enum {value = 12};
        inline static const char* classname() {return "PFJetSys";}
    };

    struct TrackJetSys
    {
        enum {value = 13};
        inline static const char* classname() {return "TrackJetSys";}
    };

    struct JPTJetSys
    {
        enum {value = 14};
        inline static const char* classname() {return "JPTJetSys";}
    };

    struct PFCHS0Sys
    {
        enum {value = 15};
        inline static const char* classname() {return "PFCHS0Sys";}
    };

    struct PFCHS1Sys
    {
        enum {value = 16};
        inline static const char* classname() {return "PFCHS1Sys";}
    };

    struct PFCHS2Sys
    {
        enum {value = 17};
        inline static const char* classname() {return "PFCHS2Sys";}
    };

    //
    // General pool of records -- it is nice not to have to
    // recompile everything when one just wants to include
    // an additional correction table on top of those already
    // made.
    //
    struct Gen0
    {
        enum {value = 18};
        inline static const char* classname() {return "Gen0";}
    };

    struct Gen1
    {
        enum {value = 19};
        inline static const char* classname() {return "Gen1";}
    };

    struct Gen2
    {
        enum {value = 20};
        inline static const char* classname() {return "Gen2";}
    };

    struct PF0
    {
        enum {value = 21};
        inline static const char* classname() {return "PF0";}
    };

    struct PF1
    {
        enum {value = 22};
        inline static const char* classname() {return "PF1";}
    };

    struct PF2
    {
        enum {value = 23};
        inline static const char* classname() {return "PF2";}
    };

    struct PF3
    {
        enum {value = 24};
        inline static const char* classname() {return "PF3";}
    };

    struct PF4
    {
        enum {value = 25};
        inline static const char* classname() {return "PF4";}
    };

    struct Calo0
    {
        enum {value = 26};
        inline static const char* classname() {return "Calo0";}
    };

    struct Calo1
    {
        enum {value = 27};
        inline static const char* classname() {return "Calo1";}
    };

    struct Calo2
    {
        enum {value = 28};
        inline static const char* classname() {return "Calo2";}
    };

    struct Calo3
    {
        enum {value = 29};
        inline static const char* classname() {return "Calo3";}
    };

    struct Calo4
    {
        enum {value = 30};
        inline static const char* classname() {return "Calo4";}
    };

    //
    // Pool of records for calculating systematic errors
    //
    struct Gen0Sys
    {
        enum {value = 31};
        inline static const char* classname() {return "Gen0Sys";}
    };

    struct Gen1Sys
    {
        enum {value = 32};
        inline static const char* classname() {return "Gen1Sys";}
    };

    struct Gen2Sys
    {
        enum {value = 33};
        inline static const char* classname() {return "Gen2Sys";}
    };

    struct PF0Sys
    {
        enum {value = 34};
        inline static const char* classname() {return "PF0Sys";}
    };

    struct PF1Sys
    {
        enum {value = 35};
        inline static const char* classname() {return "PF1Sys";}
    };

    struct PF2Sys
    {
        enum {value = 36};
        inline static const char* classname() {return "PF2Sys";}
    };

    struct PF3Sys
    {
        enum {value = 37};
        inline static const char* classname() {return "PF3Sys";}
    };

    struct PF4Sys
    {
        enum {value = 38};
        inline static const char* classname() {return "PF4Sys";}
    };

    struct PF5Sys
    {
        enum {value = 39};
        inline static const char* classname() {return "PF5Sys";}
    };

    struct PF6Sys
    {
        enum {value = 40};
        inline static const char* classname() {return "PF6Sys";}
    };

    struct PF7Sys
    {
        enum {value = 41};
        inline static const char* classname() {return "PF7Sys";}
    };

    struct PF8Sys
    {
        enum {value = 42};
        inline static const char* classname() {return "PF8Sys";}
    };

    struct PF9Sys
    {
        enum {value = 43};
        inline static const char* classname() {return "PF9Sys";}
    };

    struct Calo0Sys
    {
        enum {value = 44};
        inline static const char* classname() {return "Calo0Sys";}
    };

    struct Calo1Sys
    {
        enum {value = 45};
        inline static const char* classname() {return "Calo1Sys";}
    };

    struct Calo2Sys
    {
        enum {value = 46};
        inline static const char* classname() {return "Calo2Sys";}
    };

    struct Calo3Sys
    {
        enum {value = 47};
        inline static const char* classname() {return "Calo3Sys";}
    };

    struct Calo4Sys
    {
        enum {value = 48};
        inline static const char* classname() {return "Calo4Sys";}
    };

    struct Calo5Sys
    {
        enum {value = 49};
        inline static const char* classname() {return "Calo5Sys";}
    };

    struct Calo6Sys
    {
        enum {value = 50};
        inline static const char* classname() {return "Calo6Sys";}
    };

    struct Calo7Sys
    {
        enum {value = 51};
        inline static const char* classname() {return "Calo7Sys";}
    };

    struct Calo8Sys
    {
        enum {value = 52};
        inline static const char* classname() {return "Calo8Sys";}
    };

    struct Calo9Sys
    {
        enum {value = 53};
        inline static const char* classname() {return "Calo9Sys";}
    };

    struct CHS0Sys
    {
        enum {value = 54};
        inline static const char* classname() {return "CHS0Sys";}
    };

    struct CHS1Sys
    {
        enum {value = 55};
        inline static const char* classname() {return "CHS1Sys";}
    };

    struct CHS2Sys
    {
        enum {value = 56};
        inline static const char* classname() {return "CHS2Sys";}
    };

    struct CHS3Sys
    {
        enum {value = 57};
        inline static const char* classname() {return "CHS3Sys";}
    };

    struct CHS4Sys
    {
        enum {value = 58};
        inline static const char* classname() {return "CHS4Sys";}
    };

    struct CHS5Sys
    {
        enum {value = 59};
        inline static const char* classname() {return "CHS5Sys";}
    };

    struct CHS6Sys
    {
        enum {value = 60};
        inline static const char* classname() {return "CHS6Sys";}
    };

    struct CHS7Sys
    {
        enum {value = 61};
        inline static const char* classname() {return "CHS7Sys";}
    };

    struct CHS8Sys
    {
        enum {value = 62};
        inline static const char* classname() {return "CHS8Sys";}
    };

    struct CHS9Sys
    {
        enum {value = 63};
        inline static const char* classname() {return "CHS9Sys";}
    };
}

#endif // CondFormats_JetMETObjects_FFTJetCorrTypes_h
