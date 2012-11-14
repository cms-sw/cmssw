#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceLoader.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceRcdTypes.h"

#define add_loader_1(Record) (*this)[ #Record ] = \
   new ConcreteFFTJetRcdMapper<FFTBasicJetCorrectorSequence,Record >()

FFTBasicJetCorrectorSequenceLoader::FFTBasicJetCorrectorSequenceLoader()
{
    add_loader_1(FFTBasicJetCorrectorSequenceRcd);
    add_loader_1(FFTBasicJetSysCorrectorSequenceRcd);
}

#define add_loader_3(Record) (*this)[ #Record ] = \
   new ConcreteFFTJetRcdMapper<FFTGenJetCorrectorSequence,Record >()

FFTGenJetCorrectorSequenceLoader::FFTGenJetCorrectorSequenceLoader()
{
    add_loader_3(FFTGenJetCorrectorSequenceRcd);
    add_loader_3(FFTGenJetSysCorrectorSequenceRcd);
    add_loader_3(FFTGen0CorrectorSequenceRcd);
    add_loader_3(FFTGen1CorrectorSequenceRcd);
    add_loader_3(FFTGen2CorrectorSequenceRcd);
    add_loader_3(FFTGen0SysCorrectorSequenceRcd);
    add_loader_3(FFTGen1SysCorrectorSequenceRcd);
    add_loader_3(FFTGen2SysCorrectorSequenceRcd);
}

#define add_loader_2(Record) (*this)[ #Record ] = \
   new ConcreteFFTJetRcdMapper<FFTCaloJetCorrectorSequence,Record >()

FFTCaloJetCorrectorSequenceLoader::FFTCaloJetCorrectorSequenceLoader()
{
    add_loader_2(FFTCaloJetCorrectorSequenceRcd);
    add_loader_2(FFTCaloJetSysCorrectorSequenceRcd);
    add_loader_2(FFTCalo0CorrectorSequenceRcd);
    add_loader_2(FFTCalo1CorrectorSequenceRcd);
    add_loader_2(FFTCalo2CorrectorSequenceRcd);
    add_loader_2(FFTCalo3CorrectorSequenceRcd);
    add_loader_2(FFTCalo4CorrectorSequenceRcd);
    add_loader_2(FFTCalo0SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo1SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo2SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo3SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo4SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo5SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo6SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo7SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo8SysCorrectorSequenceRcd);
    add_loader_2(FFTCalo9SysCorrectorSequenceRcd);
}

#define add_loader_4(Record) (*this)[ #Record ] = \
   new ConcreteFFTJetRcdMapper<FFTPFJetCorrectorSequence,Record >()

FFTPFJetCorrectorSequenceLoader::FFTPFJetCorrectorSequenceLoader()
{
    add_loader_4(FFTPFJetCorrectorSequenceRcd);
    add_loader_4(FFTPFCHS0CorrectorSequenceRcd);
    add_loader_4(FFTPFCHS1CorrectorSequenceRcd);
    add_loader_4(FFTPFCHS2CorrectorSequenceRcd);
    add_loader_4(FFTPFJetSysCorrectorSequenceRcd);
    add_loader_4(FFTPFCHS0SysCorrectorSequenceRcd);
    add_loader_4(FFTPFCHS1SysCorrectorSequenceRcd);
    add_loader_4(FFTPFCHS2SysCorrectorSequenceRcd);
    add_loader_4(FFTPF0CorrectorSequenceRcd);
    add_loader_4(FFTPF1CorrectorSequenceRcd);
    add_loader_4(FFTPF2CorrectorSequenceRcd);
    add_loader_4(FFTPF3CorrectorSequenceRcd);
    add_loader_4(FFTPF4CorrectorSequenceRcd);
    add_loader_4(FFTPF0SysCorrectorSequenceRcd);
    add_loader_4(FFTPF1SysCorrectorSequenceRcd);
    add_loader_4(FFTPF2SysCorrectorSequenceRcd);
    add_loader_4(FFTPF3SysCorrectorSequenceRcd);
    add_loader_4(FFTPF4SysCorrectorSequenceRcd);
    add_loader_4(FFTPF5SysCorrectorSequenceRcd);
    add_loader_4(FFTPF6SysCorrectorSequenceRcd);
    add_loader_4(FFTPF7SysCorrectorSequenceRcd);
    add_loader_4(FFTPF8SysCorrectorSequenceRcd);
    add_loader_4(FFTPF9SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS0SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS1SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS2SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS3SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS4SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS5SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS6SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS7SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS8SysCorrectorSequenceRcd);
    add_loader_4(FFTCHS9SysCorrectorSequenceRcd);
}

#define add_loader_5(Record) (*this)[ #Record ] = \
   new ConcreteFFTJetRcdMapper<FFTTrackJetCorrectorSequence,Record >()

FFTTrackJetCorrectorSequenceLoader::FFTTrackJetCorrectorSequenceLoader()
{
    add_loader_5(FFTTrackJetCorrectorSequenceRcd);
    add_loader_5(FFTTrackJetSysCorrectorSequenceRcd);
}

#define add_loader_6(Record) (*this)[ #Record ] = \
   new ConcreteFFTJetRcdMapper<FFTJPTJetCorrectorSequence,Record >()

FFTJPTJetCorrectorSequenceLoader::FFTJPTJetCorrectorSequenceLoader()
{
    add_loader_6(FFTJPTJetCorrectorSequenceRcd);
    add_loader_6(FFTJPTJetSysCorrectorSequenceRcd);
}
