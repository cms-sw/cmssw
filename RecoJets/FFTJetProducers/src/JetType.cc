#include "RecoJets/FFTJetProducers/interface/JetType.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace fftjetcms {
    JetType parseJetType(const std::string& name)
    {
        if (!name.compare("BasicJet"))
            return BASICJET;
        else if (!name.compare("GenJet"))
            return GENJET;
        else if (!name.compare("CaloJet"))
            return CALOJET;
        else if (!name.compare("PFJet"))
            return PFJET;
        else if (!name.compare("TrackJet"))
            return TRACKJET;
        else if (!name.compare("JPTJet"))
            return JPTJET;
        else
            throw cms::Exception("FFTJetBadConfig")
                << "In parseJetType: unsupported jet type specification \""
                << name << "\"\n";
    }
}
