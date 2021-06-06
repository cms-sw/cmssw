#ifndef JetMETCorrections_FFTJetLookupTableRcd_h
#define JetMETCorrections_FFTJetLookupTableRcd_h
// -*- C++ -*-
//
// Package:     JetMETCorrections/FFTJetObjects
// Class  :     FFTJetLookupTableRcd
//
/**\class FFTJetLookupTableRcd FFTJetLookupTableRcd.h JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableRcd.h

 Description: record for FFTJet calibration lookup tables

 Usage:
    <usage>

*/
//
// Author:      I. Volobouev
// Created:     Tue Jul 31 19:49:12 CDT 2012
//

#include <FWCore/Utilities/interface/mplVector.h>

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcd.h"

template <typename CT>
struct FFTJetLookupTableRcd
    : public edm::eventsetup::DependentRecordImplementation<FFTJetLookupTableRcd<CT>,
                                                            edm::mpl::Vector<FFTJetCorrectorParametersRcd<CT> > > {
  typedef CT correction_type;
};

#endif  // JetMETCorrections_FFTJetLookupTableRcd_h
