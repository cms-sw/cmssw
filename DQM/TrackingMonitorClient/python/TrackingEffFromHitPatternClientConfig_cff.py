import FWCore.ParameterSet.Config as cms

def _layers(suffix, quant, histoPostfix):
    return [
        "effic_vs_%s_PXB1 'PXB Layer1 Efficiency vs %s' Hits%s_valid_PXB_Subdet1 Hits%s_total_PXB_Subdet1" % (suffix, quant, histoPostfix),
        "effic_vs_%s_PXB2 'PXB Layer2 Efficiency vs %s' Hits%s_valid_PXB_Subdet2 Hits%s_total_PXB_Subdet2" % (suffix, quant, histoPostfix),
        "effic_vs_%s_PXB3 'PXB Layer3 Efficiency vs %s' Hits%s_valid_PXB_Subdet3 Hits%s_total_PXB_Subdet3" % (suffix, quant, histoPostfix),
        "effic_vs_%s_PXF1 'PXF Layer1 Efficiency vs %s' Hits%s_valid_PXF_Subdet1 Hits%s_total_PXF_Subdet1" % (suffix, quant, histoPostfix),
        "effic_vs_%s_PXF2 'PXF Layer2 Efficiency vs %s' Hits%s_valid_PXF_Subdet2 Hits%s_total_PXF_Subdet2" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TIB1 'TIB Layer1 Efficiency vs %s' Hits%s_valid_TIB_Subdet1 Hits%s_total_TIB_Subdet1" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TIB2 'TIB Layer2 Efficiency vs %s' Hits%s_valid_TIB_Subdet2 Hits%s_total_TIB_Subdet2" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TIB3 'TIB Layer3 Efficiency vs %s' Hits%s_valid_TIB_Subdet3 Hits%s_total_TIB_Subdet3" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TIB4 'TIB Layer4 Efficiency vs %s' Hits%s_valid_TIB_Subdet4 Hits%s_total_TIB_Subdet4" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TOB1 'TOB Layer1 Efficiency vs %s' Hits%s_valid_TOB_Subdet1 Hits%s_total_TOB_Subdet1" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TOB2 'TOB Layer2 Efficiency vs %s' Hits%s_valid_TOB_Subdet2 Hits%s_total_TOB_Subdet2" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TOB3 'TOB Layer3 Efficiency vs %s' Hits%s_valid_TOB_Subdet3 Hits%s_total_TOB_Subdet3" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TOB4 'TOB Layer4 Efficiency vs %s' Hits%s_valid_TOB_Subdet4 Hits%s_total_TOB_Subdet4" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TOB5 'TOB Layer5 Efficiency vs %s' Hits%s_valid_TOB_Subdet5 Hits%s_total_TOB_Subdet5" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TOB6 'TOB Layer6 Efficiency vs %s' Hits%s_valid_TOB_Subdet6 Hits%s_total_TOB_Subdet6" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TID1 'TID Layer1 Efficiency vs %s' Hits%s_valid_TID_Subdet1 Hits%s_total_TID_Subdet1" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TID2 'TID Layer2 Efficiency vs %s' Hits%s_valid_TID_Subdet2 Hits%s_total_TID_Subdet2" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TID3 'TID Layer3 Efficiency vs %s' Hits%s_valid_TID_Subdet3 Hits%s_total_TID_Subdet3" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC1 'TEC Layer1 Efficiency vs %s' Hits%s_valid_TEC_Subdet1 Hits%s_total_TEC_Subdet1" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC2 'TEC Layer2 Efficiency vs %s' Hits%s_valid_TEC_Subdet2 Hits%s_total_TEC_Subdet2" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC3 'TEC Layer3 Efficiency vs %s' Hits%s_valid_TEC_Subdet3 Hits%s_total_TEC_Subdet3" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC4 'TEC Layer4 Efficiency vs %s' Hits%s_valid_TEC_Subdet4 Hits%s_total_TEC_Subdet4" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC5 'TEC Layer5 Efficiency vs %s' Hits%s_valid_TEC_Subdet5 Hits%s_total_TEC_Subdet5" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC6 'TEC Layer6 Efficiency vs %s' Hits%s_valid_TEC_Subdet6 Hits%s_total_TEC_Subdet6" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC7 'TEC Layer7 Efficiency vs %s' Hits%s_valid_TEC_Subdet7 Hits%s_total_TEC_Subdet7" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC8 'TEC Layer8 Efficiency vs %s' Hits%s_valid_TEC_Subdet8 Hits%s_total_TEC_Subdet8" % (suffix, quant, histoPostfix),
        "effic_vs_%s_TEC9 'TEC Layer9 Efficiency vs %s' Hits%s_valid_TEC_Subdet9 Hits%s_total_TEC_Subdet9" % (suffix, quant, histoPostfix),
    ]

trackingEffFromHitPattern = cms.EDAnalyzer("DQMGenericClient",
                                           subDirs = cms.untracked.vstring(
        "Tracking/TrackParameters/generalTracks/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/dzPV0p1/HitEffFromHitPattern*",
                                           ),
                                           efficiency = cms.vstring(
        _layers("PU", "GoodNumVertices", "") +
        _layers("BX", "BX", "VsBX") +
        _layers("LUMI", "LUMI", "VsLumi")
        ),
                                           resolution = cms.vstring(),
                                           verbose = cms.untracked.uint32(5),
                                           outputFileName = cms.untracked.string(""),
                                           )
def __extendEfficiencyForPixels(dets):
    """Inject the efficiency computation for the additional layers in the
    PhaseI detectors wrt Run2. The input list is cloned and modified
    rather than updated in place. The substitution add another layer
    by replacing flat '3' -> '4' for the barrel case and '2' -> '3'
    for the forward case.
    """
    from re import match
    ret = []
    for d in dets:
        ret.append(d)
        if match('.*PXB3.*', d):
            ret.append(d.replace('3', '4'))
        elif match('.*PXF2.*', d):
            ret.append(d.replace('2', '3'))
    return ret


# Use additional pixel layers in PhaseI geometry.
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(trackingEffFromHitPattern, efficiency = __extendEfficiencyForPixels(trackingEffFromHitPattern.efficiency))
