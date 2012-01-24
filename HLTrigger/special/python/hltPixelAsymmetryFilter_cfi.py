import FWCore.ParameterSet.Config as cms

# HLTPixelAsymmetryFilter
#
# More details:
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.MIB
#


hltPixelAsymmetryFilter = cms.EDFilter( "HLTPixelAsymmetryFilter",
                                       inputTag  = cms.InputTag( "hltSiPixelClusters" ),
                                       saveTags = cms.bool( False ),
                                       MinAsym   = cms.double( 0. ),     # minimum asymmetry 
                                       MaxAsym   = cms.double( 1. ),     # maximum asymmetry
                                       MinCharge = cms.double( 4000. ),  # minimum charge for a cluster to be selected (in e-)
                                       MinBarrel = cms.double( 10000. ), # minimum average charge in the barrel (bpix, in e-)
                                       ) 
