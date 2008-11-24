#ifndef CASTOR_TOWER_H
#define CASTOR_TOWER_H

#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/CastorReco/interface/CastorEgamma.h"

using namespace reco;

class Tower {
    public:
    CastorTowerCollection runTowerProduction (const CastorCellCollection inputcells, const double eta);
};

#endif /* TOWER_H */
