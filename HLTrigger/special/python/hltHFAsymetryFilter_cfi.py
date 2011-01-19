import FWCore.ParameterSet.Config as cms

# HLTHFAsymetryFilter
#
# More details:
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.MIB
#


hltHFAsymetryFilter = cms.EDFilter( "HLTHFAsymetryFilter",
                                    ECut_HF         = cms.double( 3.0 ),  # minimum energy for a cluster to be selected
                                    OS_Asym_max     = cms.double( 0.2 ),  # Opposite side asymetry maximum value
                                    SS_Asym_min     = cms.double( 0.8 ),  # Same side asymetry minimum value
                                    HFHitCollection = cms.InputTag( "hltHfreco" )
                                    ) 
