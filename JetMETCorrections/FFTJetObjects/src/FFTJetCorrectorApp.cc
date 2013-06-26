#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorApp.h"
#include "FWCore/Utilities/interface/Exception.h"

FFTJetCorrectorApp parseFFTJetCorrectorApp(const std::string& config)
{
    if (!config.compare("MCOnly"))
        return FFTJetCorrectorApp::MC_ONLY;
    else if (!config.compare("DataOnly"))
        return FFTJetCorrectorApp::DATA_ONLY;
    else if (!config.compare("MCOrData"))
        return FFTJetCorrectorApp::MC_OR_DATA;
    else if (!config.compare("DataOrMC"))
        return FFTJetCorrectorApp::MC_OR_DATA;
    else
        throw cms::Exception("FFTJetBadConfig")
            << "Error in parseFFTJetCorrectorApp: invalid string parameter \""
            << config << "\", must be one of \"MCOnly\", "
            << "\"DataOnly\", or \"DataOrMC\".\n";
}
