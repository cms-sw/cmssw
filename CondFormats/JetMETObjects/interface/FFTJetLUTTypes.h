#ifndef CondFormats_JetMETObjects_FFTJetLUTTypes_h
#define CondFormats_JetMETObjects_FFTJetLUTTypes_h

// FFTJet lookup table types. These types are essentially just
// labels for ES records -- all ES record types must be unique.

namespace fftluttypes
{
    struct EtaFlatteningFactors
    {
        enum {value = 1000};
        inline static const char* classname() {return "EtaFlatteningFactors";}
    };

    struct PileupRhoCalibration
    {
        enum {value = 1001};
        inline static const char* classname() {return "PileupRhoCalibration";}
    };

    struct PileupRhoEtaDependence
    {
        enum {value = 1002};
        inline static const char* classname() {return "PileupRhoEtaDependence";}
    };

    struct LUT0
    {
        enum {value = 1003};
        inline static const char* classname() {return "LUT0";}
    };

    struct LUT1
    {
        enum {value = 1004};
        inline static const char* classname() {return "LUT1";}
    };

    struct LUT2
    {
        enum {value = 1005};
        inline static const char* classname() {return "LUT2";}
    };

    struct LUT3
    {
        enum {value = 1006};
        inline static const char* classname() {return "LUT3";}
    };

    struct LUT4
    {
        enum {value = 1007};
        inline static const char* classname() {return "LUT4";}
    };

    struct LUT5
    {
        enum {value = 1008};
        inline static const char* classname() {return "LUT5";}
    };

    struct LUT6
    {
        enum {value = 1009};
        inline static const char* classname() {return "LUT6";}
    };

    struct LUT7
    {
        enum {value = 1010};
        inline static const char* classname() {return "LUT7";}
    };

    struct LUT8
    {
        enum {value = 1011};
        inline static const char* classname() {return "LUT8";}
    };

    struct LUT9
    {
        enum {value = 1012};
        inline static const char* classname() {return "LUT9";}
    };

    struct LUT10
    {
        enum {value = 1013};
        inline static const char* classname() {return "LUT10";}
    };

    struct LUT11
    {
        enum {value = 1014};
        inline static const char* classname() {return "LUT11";}
    };

    struct LUT12
    {
        enum {value = 1015};
        inline static const char* classname() {return "LUT12";}
    };

    struct LUT13
    {
        enum {value = 1016};
        inline static const char* classname() {return "LUT13";}
    };

    struct LUT14
    {
        enum {value = 1017};
        inline static const char* classname() {return "LUT14";}
    };

    struct LUT15
    {
        enum {value = 1018};
        inline static const char* classname() {return "LUT15";}
    };
}

#endif // CondFormats_JetMETObjects_FFTJetLUTTypes_h
