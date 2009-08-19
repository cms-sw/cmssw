import FWCore.ParameterSet.Config as cms

# include standard file
from Configuration.EventContent.EventContent_cff import *

# include from RecoHI packages
from RecoHI.HiTracking.RecoHiTracker_EventContent_cff import *
from RecoHI.HiJetAlgos.RecoHiJets_EventContent_cff import *
#from RecoHI.HiEgammaAlgos.RecoHiEgamma_EventContent_cff import *
#from RecoHI.HiCentralityAlgos.RecoHiCentrality_EventContent_cff import *
#from RecoHI.HiEvtPlaneAlgos.RecoHiEvtPlane_EventContent_cff import *
#from RecoHI.HiMuonAlgos.RecoHiMuon_EventContent_cff import *

########################################################################
#
#  RAW , RECO, AOD: 
#    include reconstruction content
#
#  RAWSIM, RECOSIM, AODSIM: 
#    include reconstruction and simulation
#
#  RAWDEBUG(RAWSIM+ALL_SIM_INFO), RAWDEBUGHLT(RAWDEBUG+HLTDEBUG)
#
#  FEVT (RAW+RECO), FEVTSIM (RAWSIM+RECOSIM), 
#  FEVTDEBUG (FEVTSIM+ALL_SIM_INFO), FEVTDEBUGHLT (FEVTDEBUG+HLTDEBUG)
#
########################################################################

# extend existing data tiers with HI-specific content
