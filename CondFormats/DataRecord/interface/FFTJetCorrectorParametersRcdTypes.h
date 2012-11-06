#ifndef DataRecord_FFTJetCorrectorParametersRcdTypes_h
#define DataRecord_FFTJetCorrectorParametersRcdTypes_h

// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     FFTJetCorrectorParametersRcdTypes
// 
/**\class FFTJetCorrectorParametersRcdTypes FFTJetCorrectorParametersRcdTypes.h CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcdTypes.h

 Description: typedefs for FFTJetCorrectorParametersRcd

 Usage:
    <usage>

*/
//
// Author:      I. Volobouev
// Created:     Tue Jul 31 19:49:12 CDT 2012
// $Id$
//

#include "CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcd.h"
#include "CondFormats/JetMETObjects/interface/FFTJetCorrTypes.h"
#include "CondFormats/JetMETObjects/interface/FFTJetLUTTypes.h"

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::BasicJet> FFTBasicJetCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::GenJet>   FFTGenJetCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CaloJet>  FFTCaloJetCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFJet>    FFTPFJetCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::TrackJet> FFTTrackJetCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::JPTJet>   FFTJPTJetCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFCHS0>   FFTPFCHS0CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFCHS1>   FFTPFCHS1CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFCHS2>   FFTPFCHS2CorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::BasicJetSys> FFTBasicJetSysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::GenJetSys>   FFTGenJetSysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CaloJetSys>  FFTCaloJetSysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFJetSys>    FFTPFJetSysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::TrackJetSys> FFTTrackJetSysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::JPTJetSys>   FFTJPTJetSysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFCHS0Sys>   FFTPFCHS0SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFCHS1Sys>   FFTPFCHS1SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PFCHS2Sys>   FFTPFCHS2SysCorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Gen0> FFTGen0CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Gen1> FFTGen1CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Gen2> FFTGen2CorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF0> FFTPF0CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF1> FFTPF1CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF2> FFTPF2CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF3> FFTPF3CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF4> FFTPF4CorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo0> FFTCalo0CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo1> FFTCalo1CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo2> FFTCalo2CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo3> FFTCalo3CorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo4> FFTCalo4CorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Gen0Sys> FFTGen0SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Gen1Sys> FFTGen1SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Gen2Sys> FFTGen2SysCorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF0Sys> FFTPF0SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF1Sys> FFTPF1SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF2Sys> FFTPF2SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF3Sys> FFTPF3SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF4Sys> FFTPF4SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF5Sys> FFTPF5SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF6Sys> FFTPF6SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF7Sys> FFTPF7SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF8Sys> FFTPF8SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::PF9Sys> FFTPF9SysCorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo0Sys> FFTCalo0SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo1Sys> FFTCalo1SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo2Sys> FFTCalo2SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo3Sys> FFTCalo3SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo4Sys> FFTCalo4SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo5Sys> FFTCalo5SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo6Sys> FFTCalo6SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo7Sys> FFTCalo7SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo8Sys> FFTCalo8SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::Calo9Sys> FFTCalo9SysCorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS0Sys> FFTCHS0SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS1Sys> FFTCHS1SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS2Sys> FFTCHS2SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS3Sys> FFTCHS3SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS4Sys> FFTCHS4SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS5Sys> FFTCHS5SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS6Sys> FFTCHS6SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS7Sys> FFTCHS7SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS8Sys> FFTCHS8SysCorrectorParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftcorrtypes::CHS9Sys> FFTCHS9SysCorrectorParametersRcd;

typedef FFTJetCorrectorParametersRcd<fftluttypes::EtaFlatteningFactors> FFTEtaFlatteningFactorsParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::PileupRhoCalibration> FFTPileupRhoCalibrationParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::PileupRhoEtaDependence> FFTPileupRhoEtaDependenceParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT0>  FFTLUT0ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT1>  FFTLUT1ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT2>  FFTLUT2ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT3>  FFTLUT3ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT4>  FFTLUT4ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT5>  FFTLUT5ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT6>  FFTLUT6ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT7>  FFTLUT7ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT8>  FFTLUT8ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT9>  FFTLUT9ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT10> FFTLUT10ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT11> FFTLUT11ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT12> FFTLUT12ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT13> FFTLUT13ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT14> FFTLUT14ParametersRcd;
typedef FFTJetCorrectorParametersRcd<fftluttypes::LUT15> FFTLUT15ParametersRcd;

#endif
