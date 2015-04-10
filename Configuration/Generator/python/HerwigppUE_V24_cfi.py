import FWCore.ParameterSet.Config as cms

# UE Tune from Herwig++ 2.4 (MRST2008LO**)

herwigppUESettingsBlock = cms.PSet(

     ue_2_4 =  cms.vstring(
    		'cd /Herwig/UnderlyingEvent',
    		'set KtCut:MinKT 4.3',
    		'set UECuts:MHatMin 8.6',
    		'set MPIHandler:InvRadius 1.2',
    		'cd /',
        ),
)
