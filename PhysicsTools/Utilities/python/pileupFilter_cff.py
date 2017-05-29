import FWCore.ParameterSet.Config as cms

pu20to25 = pileupFilter.clone()
pu20to25.minPU = cms.double(20)
pu20to25.maxPU = cms.double(25)

pu25to30 = pileupFilter.clone()
pu25to30.minPU = cms.double(25)
pu25to30.maxPU = cms.double(30)

pu30to35 = pileupFilter.clone()
pu30to35.minPU = cms.double(30)
pu30to35.maxPU = cms.double(35)

pu35to40 = pileupFilter.clone()
pu35to40.minPU = cms.double(35)
pu35to40.maxPU = cms.double(40)

pu40to45 = pileupFilter.clone()
pu40to45.minPU = cms.double(40)
pu40to45.maxPU = cms.double(45)

pu45to50 = pileupFilter.clone()
pu45to50.minPU = cms.double(45)
pu45to50.maxPU = cms.double(50)

pu50to55 = pileupFilter.clone()
pu50to55.minPU = cms.double(50)
pu50to55.maxPU = cms.double(55)

pu55to60 = pileupFilter.clone()
pu55to60.minPU = cms.double(55)
pu55to60.maxPU = cms.double(60)

pu60to65 = pileupFilter.clone()
pu60to65.minPU = cms.double(60)
pu60to65.maxPU = cms.double(65)
