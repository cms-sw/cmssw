import FWCore.ParameterSet.Config as cms

pythia8CUEP8M3SettingsBlock = cms.PSet(
    pythia8CUEP8M3Settings = cms.vstring(
              'Tune:pp 14',
	      'Tune:ee 7',
	      'Diffraction:PomFlux=5',
	      'Diffraction:MBRepsilon=0.0903',
	      'Diffraction:MBRalpha=0.1',
	      'Diffraction:MBRdyminDD = 0.',
	      'Diffraction:MBRdyminSigDD = 0.001',
	      'Diffraction:MBRdyminDDflux = 1.35',
	      'MultipartonInteractions:ecmRef=13000.0',
	      'MultipartonInteractions:ecmPow=0.25208',
	      'SpaceShower:alphaSvalue=0.1108',
	      'PDF:pSet=LHAPDF6:NNPDF30_lo_as_0130',
	      'MultipartonInteractions:bProfile=2',
	      'MultipartonInteractions:pT0Ref=2.675',
	      'MultipartonInteractions:coreRadius=0.2924',
	      'MultipartonInteractions:coreFraction=0.3356',
	      'ColourReconnection:range=1.956',
    )
)
