import FWCore.ParameterSet.Config as cms

simpleDRfilter = cms.EDFilter('simpleDRfilter',

# In debug mode, there are print-out if the MET is due to dead cell or cracks
  debug = cms.untracked.bool( False ),
# No usage now
  printSkimInfo = cms.untracked.bool( False ),

  taggingMode   = cms.bool(False),

# It's written in general that one can put pf, calo and tracking jets
  jetInputTag = cms.InputTag('ak5PFJets'),
# The pt and eta cuts applied, for instance, pt>30 && |eta|<9999
  jetSelCuts = cms.vdouble(30, 9999), # pt, eta

# This is also in general that one can put pf, tc and calo met
  metInputTag = cms.InputTag('pfMet'),

# If enabled, a root file will be produced with name give in profileRootName.
# One can produce histograms. Currently, no histograms are produced.
  makeProfileRoot = cms.untracked.bool( False ),
  profileRootName = cms.untracked.string( "simpleDRfilter.root" ),

# The status of masked cells we want to pick from global tag, for instance here, >=1
# Don't need to change ususally.
  maskedEcalChannelStatusThreshold = cms.int32( 1 ),
# The channels status we want to evaluate
# positive numbers, e.g., 12, means only channels with status 12 are considered
# negative numbers, e.g., -12, means channels with status >=12 are all considered
  chnStatusToBeEvaluated = cms.int32(-12),

# No usage now
  isProd = cms.untracked.bool( False ),

# If enabled, also check if MET is due to cracks or not. If found, events are filtered
# (if doFilter is enabled)
  doCracks = cms.untracked.bool( False ),

# No usage now
  verbose = cms.int32( 0 ),

# Simple DR filter 0 : dphi cut of jet to MET  1 : dR cut of jets to masked channles
  simpleDRfilterInput = cms.vdouble(0.5, 0.3), # 0.5, 0.3 are what RA1 use

# Definition of cracks for HB/HE and HE/HF
  cracksHBHEdef = cms.vdouble(1.3, 1.7), # crab between 1.3 and 1.7
  cracksHEHFdef = cms.vdouble(2.8, 3.2), # crab between 2.8 and 3.2

)
