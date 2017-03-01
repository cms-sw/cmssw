import FWCore.ParameterSet.Config as cms

# REMINDER : det.simulation need a startup seed;
# in your cfg, do NOT forget to give seeds via RandomNumberGeneratorService !!!
# Include Configuration/StandardSequences/data/SimulationRandomNumberGeneratorSeeds.cff
# 
# Vertex smearing is exectuted in the pgen sequence
# Geant4-based detector simulation
# (necessary geometry and mag.field records included)
#
# It retuns label "g4SimHits" that one places in the path
#
# Advise : OscarProducer has a config. parameter to control
# which HepMCProduct (gen.info) to pickup as event input, 
# the original or the one with the vertex smearing applied; 
# the parameter's name is HepMCProductLabel, it belongs to
# the PSet Generator, and the default = "generatorSmeared"
#
from Configuration.StandardSequences.Sim_cff import *
#
# if you want to skip vertex smearing, you need to reset it:
# replace g4SimHits.Generator.HepMCProductLabel = "source"
#
# several other useful parameters are listed in the WorkBook:
# https://twiki.cern.ch/twiki/bin/view/CMS/WorkBookSimDigi
#
# include TrackingParticle Producer
# NOTA BENE: it MUST be run here at the moment, since it depends 
# of the availability of the CrossingFrame in the Event
#
# Digitization (electronics response modeling)
# (all necessary geometry and other records included in the cff's)
#
# returns sequence "doAllDigi"
#
from Configuration.StandardSequences.Digi_cff import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
simulation = cms.Sequence(psim*pdigi)


