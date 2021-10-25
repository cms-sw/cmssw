import FWCore.ParameterSet.Config as cms

## The iso-based HBHE noise filter ___________________________________________||
from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *
from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import *

## The CSC beam halo tight filter ____________________________________________||
from RecoMET.METFilters.CSCTightHaloFilter_cfi import *

## The CSC beam halo tight filter ____________________________________________||
from RecoMET.METFilters.CSCTightHaloTrkMuUnvetoFilter_cfi import *

## The CSC beam halo tight filter ____________________________________________||
from RecoMET.METFilters.CSCTightHalo2015Filter_cfi import *

## The hcal problematic strip halo filter ____________________________________________||
from RecoMET.METFilters.HcalStripHaloFilter_cfi import *

## The Global TightHaloFilter2016
from RecoMET.METFilters.globalTightHalo2016Filter_cfi import *

## The Global SuperTightHaloFilter2016
from RecoMET.METFilters.globalSuperTightHalo2016Filter_cfi import *

## The HCAL laser filter _____________________________________________________||
from RecoMET.METFilters.hcalLaserEventFilter_cfi import *

## The ECAL dead cell trigger primitive filter _______________________________||
from RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi import *

## The ECAL dead cell trigger primitive filter _______________________________||
from RecoMET.METFilters.EcalDeadCellBoundaryEnergyFilter_cfi import *

## The EE bad SuperCrystal filter ____________________________________________||
from RecoMET.METFilters.eeBadScFilter_cfi import *

## The ECAL laser correction filter
from RecoMET.METFilters.ecalLaserCorrFilter_cfi import *

## The ECAL bad calibration filter ____________________________________________||
from RecoMET.METFilters.ecalBadCalibFilter_cfi import *

## The Good vertices collection needed by the tracking failure filter ________||
goodVertices = cms.EDFilter(
  "VertexSelector",
  filter = cms.bool(True),
  src = cms.InputTag("offlinePrimaryVertices"),
  cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.rho < 2")
)

## The tracking failure filter _______________________________________________||
from RecoMET.METFilters.trackingFailureFilter_cfi import *

##noscraping (outdated)_______________________________________________________||
from RecoMET.METFilters.scrapingFilter_cfi import *

## The primary vertex filter__ _______________________________________________||
from RecoMET.METFilters.primaryVertexFilter_cfi import *

## The tracking POG filters __________________________________________________||
from RecoMET.METFilters.trackingPOGFilters_cff import *
## NOTE: to make tagging mode of the tracking POG filters (three of them), please do:
##    manystripclus53X.taggedMode = cms.untracked.bool(True)
##    manystripclus53X.forcedValue = cms.untracked.bool(False)
##    toomanystripclus53X.taggedMode = cms.untracked.bool(True)
##    toomanystripclus53X.forcedValue = cms.untracked.bool(False)
##    logErrorTooManyClusters.taggedMode = cms.untracked.bool(True)
##    logErrorTooManyClusters.forcedValue = cms.untracked.bool(False)
## Also the stored boolean for the three filters is opposite to what we usually
## have for other filters, i.e., true means rejected bad events while false means 
## good events.

## The charged hadron track resolution filter _______________________________||
from RecoMET.METFilters.chargedHadronTrackResolutionFilter_cfi import *

## The muon bad track filter ________________________________________________||
from RecoMET.METFilters.muonBadTrackFilter_cfi import *

## The charged hadron track track filter (2016) ____________________________________||
from RecoMET.METFilters.BadChargedCandidateSummer16Filter_cfi import *

## The muon bad track filter (2016) ________________________________________________||
from RecoMET.METFilters.BadPFMuonSummer16Filter_cfi import *

## The charged hadron track track filter (2016) ____________________________________||
from RecoMET.METFilters.BadChargedCandidateFilter_cfi import *

## The muon bad track filter (2016) ________________________________________________||
from RecoMET.METFilters.BadPFMuonFilter_cfi import *

## The muon bad track filter with Dz cut (2020) _____________________________________||
from RecoMET.METFilters.BadPFMuonDzFilter_cfi import *

#HF noise filter 
from RecoMET.METFilters.hfNoisyHitsFilter_cfi import *

metFilters = cms.Sequence(
    goodVertices *
    globalSuperTightHalo2016Filter *
    HBHENoiseFilterResultProducer  *
    HBHENoiseFilter *
    HBHENoiseIsoFilter *
    EcalDeadCellTriggerPrimitiveFilter *  
    BadPFMuonFilter *
    BadPFMuonDzFilter *
    hfNoisyHitsFilter *
    eeBadScFilter *
    ecalBadCalibFilter 
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(metFilters, metFilters.copyAndExclude([
    HBHENoiseFilterResultProducer, HBHENoiseFilter, HBHENoiseIsoFilter, # No hcalnoise for hgcal
    eeBadScFilter                                   # No EE
]))


from Configuration.Eras.Modifier_run2_jme_2016_cff import run2_jme_2016
run2_jme_2016.toReplaceWith(metFilters, metFilters.copyAndExclude([
    ecalBadCalibFilter, hfNoisyHitsFilter
]))

