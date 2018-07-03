from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools \
    import ( EleHLTSelection_V1,
             IsolationCutInputs_V2,
             configureVIDCutBasedEleHLTPreselection_V1 )             

#
# Set of requirements tighter than HLT, meant to keep HLT efficiency high once this
# preselection is applied.
#
# The preselection cuts below are derived for HLT WPLoose beginning summer 2016.
# See documentation of cut values on the twiki:
#        https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
# See presentation with details of tuning here:
#  https://indico.cern.ch/event/491507/contributions/2192817/attachments/1285452/1911768/EGM_HLTsafeCuts_31May16.pdf
#
# First, define cut values
#

# Veto working point Barrel and Endcap
idName = "cutBasedElectronHLTPreselection-Summer16-V1"
WP_HLTSafe_EB = EleHLTSelection_V1(
    idName   , # idName
    0.011   , # full5x5_sigmaIEtaIEtaCut
    0.004   , # dEtaInSeedCut
    0.020   , # dPhiInCut
    0.060   , # hOverECut
    0.013   , # absEInverseMinusPInverseCut
    # Calorimeter isolations: 
    0.160   , # ecalPFClusterIsoLowPtCut
    0.160   , # ecalPFClusterIsoHighPtCut
    0.120   , # hcalPFClusterIsoLowPtCut
    0.120   , # hcalPFClusterIsoHighPtCut
    # Tracker isolation:
    # Three constants for the GsfEleTrkPtIsoCut: 
    #     cut = constTerm if Et < slopeStart
    #     cut = slopeTerm * (Et - slopeStart) + constTerm if Et >= slopeStart
    0.080 ,   # trkIsoSlopeTerm        
    0.000,    # trkIsoSlopeStart
    0.000,    # trkIsoConstTerm
    #
    1e30      # normalizedGsfChi2Cut - no cut
    )

WP_HLTSafe_EE = EleHLTSelection_V1(
    idName   , # idName
    0.031   , # full5x5_sigmaIEtaIEtaCut
    1e30    , # dEtaInSeedCut - no cut
    1e30    , # dPhiInCut - no cut
    0.065   , # hOverECut
    0.013   , # absEInverseMinusPInverseCut
    # Calorimeter isolations: 
    0.120   , # ecalPFClusterIsoLowPtCut
    0.120   , # ecalPFClusterIsoHighPtCut
    0.120   , # hcalPFClusterIsoLowPtCut
    0.120   , # hcalPFClusterIsoHighPtCut
    # Tracker isolation:
    # Three constants for the GsfEleTrkPtIsoCut: 
    #     cut = constTerm if Et < slopeStart
    #     cut = slopeTerm * (Et - slopeStart) + constTerm if Et >= slopeStart
    0.080 ,   # trkIsoSlopeTerm        
    0.000,    # trkIsoSlopeStart
    0.000,    # trkIsoConstTerm
    #
    3.000     # normalizedGsfChi2Cut
    )


# Second, define what effective areas to use for pile-up correction
isoInputsEcal = IsolationCutInputs_V2(
    # isoEffAreas
    "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_HLT_ecalPFClusterIso.txt"
)
isoInputsHcal = IsolationCutInputs_V2(
    # isoEffAreas
    "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_HLT_hcalPFClusterIso.txt"
)


#
# Set up VID configuration for all cuts and working points
#
cutBasedElectronHLTPreselection_Summer16_V1 = configureVIDCutBasedEleHLTPreselection_V1(
    WP_HLTSafe_EB, WP_HLTSafe_EE, 
    isoInputsEcal, isoInputsHcal)


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedElectronHLTPreselection_Summer16_V1.idName,
                             'aef5f00cc25a63b44192c6fc449f7dc0')


### for now until we have a database...
cutBasedElectronHLTPreselection_Summer16_V1.isPOGApproved = cms.untracked.bool(True)
