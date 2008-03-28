import FWCore.ParameterSet.Config as cms

pythiaHepMCVerbosity = cms.untracked.bool(False)
pythiaPylistVerbosity = cms.untracked.int32(0)
maxEventsToPrint = cms.untracked.int32(0) ## events to print if pythiaPylistVerbosit

aBeamTarget = cms.double(207.0) ## beam/target atomic number

bFixed = cms.double(0.0) ## fixed impact param (fm); valid only if cflag_=0

cFlag = cms.int32(0) ## centrality flag

# =  0 fixed impact param
# <> 0 --> minbias with standard glauber geometry
comEnergy = cms.double(5500.0)
angularSpectrumSelector = cms.int32(0) ## angular emitted gluon spectrum : 

# 0-small angle, 1-broad angle, 2-collinear                           
doQuench = cms.bool(True)
doRadiativeEnLoss = cms.bool(True) ## if true, perform partonic radiative en loss

doCollisionalEnLoss = cms.bool(True) ## if true, perform partonic collisional en loss

numQuarkFlavor = cms.int32(0) ## number of active quark flavors in qgp; allowed values: 0,1,2,3

qgpInitialTemperature = cms.double(1.0) ## initial temperature of QGP; allowed range [0.2,2.0]GeV;

qgpProperTimeFormation = cms.double(0.1) ## proper time of QGP formation; allowed range [0.01,10.0]fm/c; 


