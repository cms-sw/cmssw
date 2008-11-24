#ifndef CASTOR_EGAMMA_H
#define CASTOR_EGAMMA_H

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/CastorReco/interface/CastorEgamma.h"

using namespace reco;

class Egamma {
    public:
    CastorEgammaCollection runEgamma (const CastorJetCollection inputjets, const double minratio, const double maxwidth, const double maxdepth);
};

#endif /* EGAMMA_H */
