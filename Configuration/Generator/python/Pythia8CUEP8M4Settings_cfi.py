import FWCore.ParameterSet.Config as cms

pythia8CUEP8M4SettingsBlock = cms.PSet(
    pythia8CUEP8M4Settings = cms.vstring(
              'Tune:pp 14',
	      'Tune:ee 7',
              'Diffraction:PomFlux=4',
	      'Diffraction:PomFluxEpsilon=0.1195',
	      'Diffraction:PomFluxAlphaPrime=0.1417',
	      'MultipartonInteractions:ecmRef=13000.0',
	      'MultipartonInteractions:ecmPow=0.25208',
	      'SpaceShower:alphaSvalue=0.1108',
	      'SigmaTotal:zeroAXB=off',
	      'PDF:pSet=LHAPDF6:NNPDF30_lo_as_0130',
	      'MultipartonInteractions:bProfile=2',
	      'MultipartonInteractions:pT0Ref=2.481',
	      'MultipartonInteractions:coreFraction=0.5457',
	      'MultipartonInteractions:coreRadius=0.7195',
	      'ColourReconnection:range=2.921',
    )
)
