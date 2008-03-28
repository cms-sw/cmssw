import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to make the CaloTowersConstituentsMap 
#
EcalTrigTowerConstituentsMapBuilder = cms.ESProducer("EcalTrigTowerConstituentsMapBuilder",
    # untracked string MapFile = "Geometry/CaloTopology/data/ee_tt_map.dat"     # file with EE, other mapping
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EndCap_TTMap.txt')
)


