import FWCore.ParameterSet.Config as cms

# HLTPixelAsymetryFilter
#
# More details:
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.MIB
#


hltPixelAsymetryFilter = cms.EDFilter( "HLTPixelAsymetryFilter",
                                       inputTag  = cms.InputTag( "hltSiPixelClusters" ),
                                       MinAsym   = cms.double( 0. ),     # minimum asymetry 
                                       MaxAsym   = cms.double( 1. ),     # maximum asymetry
                                       MinCharge = cms.double( 4000. ),  # minimum charge for a cluster to be selected (in e-)
                                       MinBarrel = cms.double( 10000. ), # minimum average charge in the barrel (bpix, in e-)
                                       ) 
