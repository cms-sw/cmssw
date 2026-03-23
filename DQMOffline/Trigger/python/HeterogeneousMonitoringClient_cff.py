import FWCore.ParameterSet.Config as cms

# Tracker
from DQM.SiPixelHeterogeneous.siPixelTrackComparisonHarvester_cfi import *
hltSiPixelTrackComparisonHarvester = siPixelTrackComparisonHarvester.clone(topFolderName = 'HLT/HeterogeneousComparisons/PixelTracks')

from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toModify(hltSiPixelTrackComparisonHarvester,
                             topFolderName = 'HLT/HeterogeneousComparisons/pixelTracksSoA'
)

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
hltTrackToTrackEfficiencies = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring(
        "HLT/HeterogeneousComparisons/pixelTracks",
        "HLT/HeterogeneousComparisons/initialStepSeedTracks",
    ),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "Eff_pt              'Relative Efficiency vs Pt;#P_T;relative efficiency'               ref_matched_pt          ref_pt          eff",
        "Eff_eta             'Relative Efficiency vs Eta;#eta;relative efficiency'              ref_matched_eta         ref_eta         eff",
        "Eff_phi             'Relative Efficiency vs Phi;#phi;relative efficiency'              ref_matched_phi         ref_phi         eff",
        "Eff_dxy             'Relative Efficiency vs dxy;d_{xy};relative efficiency'            ref_matched_dxy         ref_dxy         eff",
        "Eff_dz              'Relative Efficiency vs dz;d_{z};relative efficiency'              ref_matched_dz          ref_dz          eff",
        "Eff_dxyWRTpv        'Relative Efficiency vs dxyWRTpv;d_{xy};relative efficiency'       ref_matched_dxyWRTpv    ref_dxyWRTpv    eff",
        "Eff_dzWRTpv         'Relative Efficiency vs dzWRTpv;d_{z};relative efficiency'         ref_matched_dzWRTpv     ref_dzWRTpv     eff",
        "Eff_charge          'Relative Efficiency vs charge;charge;relative efficiency'         ref_matched_charge      ref_charge      eff",
        "Eff_hits            'Relative Efficiency vs hits;number of hits;relative efficiency'   ref_matched_hits        ref_hits        eff",

        "FakeRate_pt         'Relative Fake Rate vs Pt;#P_T;relative fake rate'                 mon_unMatched_pt        mon_pt          eff",
        "FakeRate_eta        'Relative Fake Rate vs Eta;#eta;relative fake rate'                mon_unMatched_eta       mon_eta         eff",
        "FakeRate_phi        'Relative Fake Rate vs Phi;#phi;relative fake rate'                mon_unMatched_phi       mon_phi         eff",
        "FakeRate_dxy        'Relative Fake Rate vs dxy;d_{xy};relative fake rate'              mon_unMatched_dxy       mon_dxy         eff",
        "FakeRate_dz         'Relative Fake Rate vs dz;d_{z};relative fake rate'                mon_unMatched_dz        mon_dz          eff",
        "FakeRate_dxyWRTpv   'Relative Fake Rate vs dxyWRTpv;d_{xy};relative fake rate'         mon_unMatched_dxyWRTpv  mon_dxyWRTpv    eff",
        "FakeRate_dzWRTpv    'Relative Fake Rate vs dzWRTpv;d_{z};relative fake rate'           mon_unMatched_dzWRTpv   mon_dzWRTpv     eff",
        "FakeRate_charge     'Relative Fake Rate vs charge;charge;relative fake rate'           mon_unMatched_charge    mon_charge      eff",
        "FakeRate_hits       'Relative Fake Rate vs hits;number of hits;relative fake rate'     mon_unMatched_hits      mon_hits        eff",
    ),
)

HLTHeterogeneousMonitoringHarvesting =  cms.Sequence(
    hltSiPixelTrackComparisonHarvester
)

_HLTHeterogeneousMonitoringHarvesting = HLTHeterogeneousMonitoringHarvesting.copy()
_HLTHeterogeneousMonitoringHarvesting += hltTrackToTrackEfficiencies
alpakaValidationHLT.toReplaceWith(HLTHeterogeneousMonitoringHarvesting, _HLTHeterogeneousMonitoringHarvesting)
