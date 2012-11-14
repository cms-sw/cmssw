#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableSequenceLoader.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableRcdTypes.h"

#define add_loader(Record) (*this)[ #Record ] = \
   new ConcreteFFTJetRcdMapper<FFTJetLookupTableSequence,Record >()

FFTJetLookupTableSequenceLoader::FFTJetLookupTableSequenceLoader()
{
    add_loader(FFTEtaFlatteningFactorsTableRcd);
    add_loader(FFTPileupRhoCalibrationTableRcd);
    add_loader(FFTPileupRhoEtaDependenceTableRcd);
    add_loader(FFTLUT0TableRcd);
    add_loader(FFTLUT1TableRcd);
    add_loader(FFTLUT2TableRcd);
    add_loader(FFTLUT3TableRcd);
    add_loader(FFTLUT4TableRcd);
    add_loader(FFTLUT5TableRcd);
    add_loader(FFTLUT6TableRcd);
    add_loader(FFTLUT7TableRcd);
    add_loader(FFTLUT8TableRcd);
    add_loader(FFTLUT9TableRcd);
    add_loader(FFTLUT10TableRcd);
    add_loader(FFTLUT11TableRcd);
    add_loader(FFTLUT12TableRcd);
    add_loader(FFTLUT13TableRcd);
    add_loader(FFTLUT14TableRcd);
    add_loader(FFTLUT15TableRcd);
}
