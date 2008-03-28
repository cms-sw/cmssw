# The following comments couldn't be translated into the new config version:

# This cfg file can be used to visualize the simulation geometry
# and the magnetic field.
# Use it with 
# iguana cmssw-geom.cfg

#
# This is a service to configure IGUANA
#

#
# Number of various windows to be popped up and tiled. 
# Default value is 'RPhi Window'
#
# untracked vstring Views = {'3D Window', 'Lego Window', 'RPhi Window', 'RZ Window'}

# #########
# Services (this is a default value):
# untracked vstring Services = {'Services/Framework/GUI/'}
# #########
# Whether or not load the Text browser (this is a default value):
# untracked bool TextBrowser = true
# #########
# Whether or not load the Twig browser (this is a default value):
# untracked bool TwigBrowser = true
# #########
# Which context data proxies to load (the default value is defined
# in VisApplicationMain and usually loads all available data proxies):

import FWCore.ParameterSet.Config as cms

process = cms.Process("IGUANA")
#Geometry
process.load("Geometry.ForwardCommonData.iguanaZdcTestConfiguration_cfi")

#Magnetic Field
process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.VisConfigurationService = cms.Service("VisConfigurationService",
    Views = cms.untracked.vstring('3D Window'),
    ContentProxies = cms.untracked.vstring('Simulation/Core', 'Simulation/Geometry', 'Simulation/MagField')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.geom = cms.EDProducer("GeometryProducer",
    MagneticField = cms.PSet(
        delta = cms.double(1.0)
    ),
    UseMagneticField = cms.bool(False),
    UseSensitiveDetectors = cms.bool(False)
)

process.p1 = cms.Path(process.geom)

