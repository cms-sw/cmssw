import FWCore.ParameterSet.Config as cms

# UE Tune from Herwig++ 2.3 (MRST2001)

herwigppUESettingsBlock = cms.PSet(

     hwpp_ue_V23 =  cms.vstring(
	'set /Herwig/UnderlyingEvent/KtCut:MinKT 4.0',
	'set /Herwig/UnderlyingEvent/UECuts:MHatMin 8.0',
	'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.5',
        ),
)
