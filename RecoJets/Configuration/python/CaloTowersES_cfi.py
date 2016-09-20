import FWCore.ParameterSet.Config as cms

import Geometry.CaloEventSetup.caloTowerConstituents_cfi

CaloTowerConstituentsMapBuilder = Geometry.CaloEventSetup.caloTowerConstituents_cfi.caloTowerConstituents.clone()
CaloTowerConstituentsMapBuilder.MapFile = "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz"

from Configuration.StandardSequences.Eras import eras
eras.phase2_hgcal.toModify( CaloTowerConstituentsMapBuilder, MapFile = "", SkipHE = True )
