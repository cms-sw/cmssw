#ifndef JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequence_h
#define JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequence_h

#include <string>

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetDict.h"
#include "JetMETCorrections/InterpolationTables/interface/StorableMultivariateFunctor.h"

typedef FFTJetDict<std::string, FFTJetDict<std::string, std::shared_ptr<npstat::StorableMultivariateFunctor> > >
    FFTJetLookupTableSequence;

#endif  // JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequence_h
