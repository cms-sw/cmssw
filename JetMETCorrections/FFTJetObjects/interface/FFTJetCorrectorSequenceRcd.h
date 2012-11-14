#ifndef JetMETCorrections_FFTJetCorrectorSequenceRcd_h
#define JetMETCorrections_FFTJetCorrectorSequenceRcd_h
// -*- C++ -*-
//
// Package:     JetMETCorrections/FFTJetObjects
// Class  :     FFTJetCorrectorSequenceRcd
// 
/**\class FFTJetCorrectorSequenceRcd FFTJetCorrectorSequenceRcd.h JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceRcd.h

 Description: record for FFTJet jet corrector sequences

 Usage:
    <usage>

*/
//
// Author:      I. Volobouev
// Created:     Tue Jul 31 19:49:12 CDT 2012
// $Id$
//

#include <boost/mpl/vector.hpp>

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcd.h"

template <typename CT>
struct FFTJetCorrectorSequenceRcd : public edm::eventsetup::DependentRecordImplementation<
    FFTJetCorrectorSequenceRcd<CT>,
    boost::mpl::vector<FFTJetCorrectorParametersRcd<CT> >
> {typedef CT correction_type;};

#endif // JetMETCorrections_FFTJetCorrectorSequenceRcd_h
