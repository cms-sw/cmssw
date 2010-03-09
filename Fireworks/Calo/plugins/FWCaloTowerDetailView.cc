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
   printf("build tower detail view\n");
}

void
FWCaloTowerDetailView::setTextInfo(const FWModelId& id, const CaloTower*)
{
}

REGISTER_FWDETAILVIEW(FWCaloTowerDetailView, Tower);
