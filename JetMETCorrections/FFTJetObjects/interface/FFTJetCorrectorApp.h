#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorApp_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorApp_h

#include <string>

enum FFTJetCorrectorApp
{
    MC_ONLY = 0,
    DATA_ONLY,
    MC_OR_DATA
};

FFTJetCorrectorApp parseFFTJetCorrectorApp(const std::string& config);

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrectorApp_h
