#ifndef RecoLocalCalo_HcalRecAlgos_OOTPileupCorrectionReader_h
#define RecoLocalCalo_HcalRecAlgos_OOTPileupCorrectionReader_h

#include "Alignment/Geners/interface/AbsReader.hh"
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsOOTPileupCorrection.h"

// Note that the folowing class does not have any public constructors.
// All application usage is through the gs::StaticReader wrapper.
//
class OOTPileupCorrectionReader : public gs::DefaultReader<AbsOOTPileupCorrection>
{
    typedef gs::DefaultReader<AbsOOTPileupCorrection> Base;
    friend class gs::StaticReader<OOTPileupCorrectionReader>;
    OOTPileupCorrectionReader();
};

typedef gs::StaticReader<OOTPileupCorrectionReader> StaticOOTPileupCorrectionReader;

#endif // RecoLocalCalo_HcalRecAlgos_OOTPileupCorrectionReader_h
