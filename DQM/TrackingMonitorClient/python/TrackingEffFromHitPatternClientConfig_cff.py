import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

def _layers(suffix, quant, histoPostfix):
    return [
        "effic_vs_{0}_PXB1  'PXB Layer1 Efficiency vs {1}'  Hits{2}_valid_PXB_Subdet1  Hits{2}_total_PXB_Subdet1" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXB2  'PXB Layer2 Efficiency vs {1}'  Hits{2}_valid_PXB_Subdet2  Hits{2}_total_PXB_Subdet2" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXB3  'PXB Layer3 Efficiency vs {1}'  Hits{2}_valid_PXB_Subdet3  Hits{2}_total_PXB_Subdet3" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXB4  'PXB Layer4 Efficiency vs {1}'  Hits{2}_valid_PXB_Subdet4  Hits{2}_total_PXB_Subdet4" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF1  'PXF Layer1 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet1  Hits{2}_total_PXF_Subdet1" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF2  'PXF Layer2 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet2  Hits{2}_total_PXF_Subdet2" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF3  'PXF Layer3 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet3  Hits{2}_total_PXF_Subdet3" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF4  'PXF Layer4 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet4  Hits{2}_total_PXF_Subdet4" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF5  'PXF Layer5 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet5  Hits{2}_total_PXF_Subdet5" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF6  'PXF Layer6 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet6  Hits{2}_total_PXF_Subdet6" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF7  'PXF Layer7 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet7  Hits{2}_total_PXF_Subdet7" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF8  'PXF Layer8 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet8  Hits{2}_total_PXF_Subdet8" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF9  'PXF Layer9 Efficiency vs {1}'  Hits{2}_valid_PXF_Subdet9  Hits{2}_total_PXF_Subdet9" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF10 'PXF Layer10 Efficiency vs {1}' Hits{2}_valid_PXF_Subdet10 Hits{2}_total_PXF_Subdet10".format(suffix, quant, histoPostfix),
        "effic_vs_{0}_PXF11 'PXF Layer11 Efficiency vs {1}' Hits{2}_valid_PXF_Subdet11 Hits{2}_total_PXF_Subdet11".format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TIB1  'TIB Layer1 Efficiency vs {1}'  Hits{2}_valid_TIB_Subdet1  Hits{2}_total_TIB_Subdet1" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TIB2  'TIB Layer2 Efficiency vs {1}'  Hits{2}_valid_TIB_Subdet2  Hits{2}_total_TIB_Subdet2" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TIB3  'TIB Layer3 Efficiency vs {1}'  Hits{2}_valid_TIB_Subdet3  Hits{2}_total_TIB_Subdet3" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TIB4  'TIB Layer4 Efficiency vs {1}'  Hits{2}_valid_TIB_Subdet4  Hits{2}_total_TIB_Subdet4" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TOB1  'TOB Layer1 Efficiency vs {1}'  Hits{2}_valid_TOB_Subdet1  Hits{2}_total_TOB_Subdet1" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TOB2  'TOB Layer2 Efficiency vs {1}'  Hits{2}_valid_TOB_Subdet2  Hits{2}_total_TOB_Subdet2" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TOB3  'TOB Layer3 Efficiency vs {1}'  Hits{2}_valid_TOB_Subdet3  Hits{2}_total_TOB_Subdet3" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TOB4  'TOB Layer4 Efficiency vs {1}'  Hits{2}_valid_TOB_Subdet4  Hits{2}_total_TOB_Subdet4" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TOB5  'TOB Layer5 Efficiency vs {1}'  Hits{2}_valid_TOB_Subdet5  Hits{2}_total_TOB_Subdet5" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TOB6  'TOB Layer6 Efficiency vs {1}'  Hits{2}_valid_TOB_Subdet6  Hits{2}_total_TOB_Subdet6" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TID1  'TID Layer1 Efficiency vs {1}'  Hits{2}_valid_TID_Subdet1  Hits{2}_total_TID_Subdet1" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TID2  'TID Layer2 Efficiency vs {1}'  Hits{2}_valid_TID_Subdet2  Hits{2}_total_TID_Subdet2" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TID3  'TID Layer3 Efficiency vs {1}'  Hits{2}_valid_TID_Subdet3  Hits{2}_total_TID_Subdet3" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TID4  'TID Layer3 Efficiency vs {1}'  Hits{2}_valid_TID_Subdet4  Hits{2}_total_TID_Subdet4" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TID5  'TID Layer3 Efficiency vs {1}'  Hits{2}_valid_TID_Subdet5  Hits{2}_total_TID_Subdet5" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC1  'TEC Layer1 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet1  Hits{2}_total_TEC_Subdet1" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC2  'TEC Layer2 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet2  Hits{2}_total_TEC_Subdet2" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC3  'TEC Layer3 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet3  Hits{2}_total_TEC_Subdet3" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC4  'TEC Layer4 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet4  Hits{2}_total_TEC_Subdet4" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC5  'TEC Layer5 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet5  Hits{2}_total_TEC_Subdet5" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC6  'TEC Layer6 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet6  Hits{2}_total_TEC_Subdet6" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC7  'TEC Layer7 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet7  Hits{2}_total_TEC_Subdet7" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC8  'TEC Layer8 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet8  Hits{2}_total_TEC_Subdet8" .format(suffix, quant, histoPostfix),
        "effic_vs_{0}_TEC9  'TEC Layer9 Efficiency vs {1}'  Hits{2}_valid_TEC_Subdet9  Hits{2}_total_TEC_Subdet9" .format(suffix, quant, histoPostfix),
    ]

trackingEffFromHitPattern = DQMEDHarvester("DQMGenericClient",
                                           subDirs = cms.untracked.vstring(
        "Tracking/TrackParameters/generalTracks/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/dzPV0p1/HitEffFromHitPattern*",
        "Muons/Tracking/innerTrack/HitEffFromHitPattern*",
        "Muons/globalMuons/HitEffFromHitPattern*",
                                           ),
                                           efficiency = cms.vstring(
        _layers("PU", "GoodNumVertices", "") +
        _layers("BX", "BX", "VsBX") +
        _layers("LUMI", "LUMI", "VsLUMI")
        ),
                                           resolution = cms.vstring(),
                                           verbose = cms.untracked.uint32(5),
                                           outputFileName = cms.untracked.string(""),
                                           )
