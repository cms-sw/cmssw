#include "RecoLocalCalo/HcalRecAlgos/interface/OOTPileupCorrectionReader.h"

AbsOOTPileupCorrection* AbsOOTPileupCorrection::read(const gs::ClassId& id,
                                                     std::istream& in)
{
    return StaticOOTPileupCorrectionReader::instance().read(id, in);
}
