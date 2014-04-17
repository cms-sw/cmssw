#include "RecoLocalCalo/HcalRecAlgos/interface/OOTPileupCorrectionReader.h"

// Include headers for all classes derived from AbsOOTPileupCorrection
// which are known at this point in code development
#include "RecoLocalCalo/HcalRecAlgos/interface/DummyOOTPileupCorrection.h"

// Simple macro for adding a reader for a class derived from AbsOOTPileupCorrection
#define add_reader(Derived) do {                                                   \
    const gs::ClassId& id(gs::ClassId::makeId<Derived >());                        \
    (*this)[id.name()] = new gs::ConcreteReader<AbsOOTPileupCorrection,Derived >();\
} while(0);

OOTPileupCorrectionReader::OOTPileupCorrectionReader()
{
    add_reader(DummyOOTPileupCorrection);
}
