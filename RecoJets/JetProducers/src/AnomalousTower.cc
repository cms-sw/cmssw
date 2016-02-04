#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoJets/JetProducers/interface/AnomalousTower.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

AnomalousTower::AnomalousTower(const edm::ParameterSet& ps)
    : init_param(unsigned, maxBadEcalCells),
      init_param(unsigned, maxRecoveredEcalCells),
      init_param(unsigned, maxProblematicEcalCells),
      init_param(unsigned, maxBadHcalCells),
      init_param(unsigned, maxRecoveredHcalCells),
      init_param(unsigned, maxProblematicHcalCells)
{
}

bool AnomalousTower::operator()(const reco::Candidate& input) const
{
    const CaloTower* tower = dynamic_cast<const CaloTower*>(&input);
    if (tower)
        return tower->numBadEcalCells()         > maxBadEcalCells         ||
               tower->numRecoveredEcalCells()   > maxRecoveredEcalCells   ||
               tower->numProblematicEcalCells() > maxProblematicEcalCells ||
               tower->numBadHcalCells()         > maxBadHcalCells         ||
               tower->numRecoveredHcalCells()   > maxRecoveredHcalCells   ||
               tower->numProblematicHcalCells() > maxProblematicHcalCells;
    else
        return false;
}
