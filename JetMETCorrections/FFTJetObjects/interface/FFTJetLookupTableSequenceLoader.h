#ifndef JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequenceLoader_h
#define JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequenceLoader_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetRcdMapper.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableSequence.h"

//
// Note that the class below does not have any public constructors.
// All application usage is via the StaticFFTJetRcdMapper wrapper.
//
class FFTJetLookupTableSequenceLoader :
    public DefaultFFTJetRcdMapper<FFTJetLookupTableSequence>
{
    typedef DefaultFFTJetRcdMapper<FFTJetLookupTableSequence> Base;
    friend class StaticFFTJetRcdMapper<FFTJetLookupTableSequenceLoader>;
    FFTJetLookupTableSequenceLoader();
};
        
typedef StaticFFTJetRcdMapper<FFTJetLookupTableSequenceLoader>
StaticFFTJetLookupTableSequenceLoader;

#endif // JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequenceLoader_h
