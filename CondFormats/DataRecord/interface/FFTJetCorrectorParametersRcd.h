#ifndef DataRecord_FFTJetCorrectorParametersRcd_h
#define DataRecord_FFTJetCorrectorParametersRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     FFTJetCorrectorParametersRcd
// 
/**\class FFTJetCorrectorParametersRcd FFTJetCorrectorParametersRcd.h CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcd.h

 Description: record for FFTJet correction data

 Usage:
    <usage>

*/
//
// Author:      I. Volobouev
// Created:     Tue Jul 31 19:49:12 CDT 2012
// $Id$
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

template <typename CT>
struct FFTJetCorrectorParametersRcd : public edm::eventsetup::EventSetupRecordImplementation<FFTJetCorrectorParametersRcd<CT> > {typedef CT correction_type;};

#endif
