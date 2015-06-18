import FWCore.ParameterSet.Config as cms

# UE Tune from Herwig++ 2.4 (MRST2008LO**)

herwigppUESettingsBlock = cms.PSet(

     hwpp_ue_V24 =  cms.vstring(
    		'set /Herwig/UnderlyingEvent/KtCut:MinKT 4.3',
    		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 8.6',
    		'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.2',
        ),
)
