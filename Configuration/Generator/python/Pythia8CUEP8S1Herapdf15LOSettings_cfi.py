import FWCore.ParameterSet.Config as cms

pythia8CUEP8S1herapdfSettingsBlock = cms.PSet(
    pythia8CUEP8S1herapdfSettings = cms.vstring(
        'Tune:pp 5',
        'Tune:ee 3',
        'PDF:useLHAPDF=on',
	'PDF:LHAPDFset=HERAPDF1.5LO_EIG.LHgrid',
        'MultipleInteractions:pT0Ref=2.000072e+00',
	'MultipleInteractions:ecmPow=2.498802e-01',
	'MultipleInteractions:expPow=1.690506e+00',
	'BeamRemnants:reconnectRange=6.096364e+00',        
   )
)

