#ifndef JetMETCorrections_FFTJetObjects_loadFFTJetInterpolationTable_h
#define JetMETCorrections_FFTJetObjects_loadFFTJetInterpolationTable_h

#include "Alignment/Geners/interface/StringArchive.hh"
#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "JetMETCorrections/InterpolationTables/interface/StorableMultivariateFunctor.h"

CPP11_auto_ptr<npstat::StorableMultivariateFunctor>
loadFFTJetInterpolationTable(const edm::ParameterSet& ps,
                             gs::StringArchive& ar, bool verbose);

#endif // JetMETCorrections_FFTJetObjects_loadFFTJetInterpolationTable_h
