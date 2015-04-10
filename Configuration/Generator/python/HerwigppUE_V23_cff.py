import FWCore.ParameterSet.Config as cms

# UE Tune from Herwig++ 2.3 (MRST2001)

herwigppUESettingsBlock = cms.PSet(

     ue_2_3 =  cms.vstring(
	'cd /Herwig/UnderlyingEvent',
	'set KtCut:MinKT 4.0',
	'set UECuts:MHatMin 8.0',
	'set MPIHandler:InvRadius 1.5',
	'cd /',
        ),
)
