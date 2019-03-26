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
    ]

pixelTrackingEffFromHitPattern = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("Tracking/PixelTrackParameters/pixelTracks/HitEffFromHitPattern*",
                                    "Tracking/PixelTrackParameters/dzPV0p1/HitEffFromHitPattern*",
                                    "Tracking/PixelTrackParameters/pt_0to1/HitEffFromHitPattern*",
                                    "Tracking/PixelTrackParameters/pt_1/HitEffFromHitPattern*"),
    efficiency = cms.vstring(
        _layers("PU", "GoodNumVertices", "") +
        _layers("BX", "BX", "VsBX") +
        _layers("LUMI", "LUMI", "VsLUMI")
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(5),
    outputFileName = cms.untracked.string(""),
)
