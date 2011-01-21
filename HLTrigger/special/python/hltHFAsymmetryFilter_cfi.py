import FWCore.ParameterSet.Config as cms

# HLTHFAsymmetryFilter
#
# More details:
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.MIB
#


hltHFAsymmetryFilter = cms.EDFilter( "HLTHFAsymmetryFilter",
                                    ECut_HF         = cms.double( 3.0 ),  # minimum energy for a cluster to be selected
                                    OS_Asym_max     = cms.double( 0.2 ),  # Opposite side asymmetry maximum value
                                    SS_Asym_min     = cms.double( 0.8 ),  # Same side asymmetry minimum value
                                    HFHitCollection = cms.InputTag( "hltHfreco" )
                                    ) 
