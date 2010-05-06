//
// constructors and destructor
//

#include "Fireworks/Calo/plugins/FWCaloTowerDetailView.h"

FWCaloTowerDetailView::FWCaloTowerDetailView()
{ 
}

FWCaloTowerDetailView::~FWCaloTowerDetailView()
{
}

//
// member functions
//
void FWCaloTowerDetailView::build(const FWModelId &id, const CaloTower*)
{
}

void
FWCaloTowerDetailView::setTextInfo(const FWModelId& id, const CaloTower*)
{
}

REGISTER_FWDETAILVIEW(FWCaloTowerDetailView, Tower);
