from ..layouts.layout_manager import register_layout

register_layout(source='DT/02-Segments/SegmentGlbSummary', destination='00 Shift/DT/SegmentGlbSummary', name='00-SegmentOccupancySummary', overlay='')
register_layout(source='DT/02-Segments/00-MeanRes/MeanDistr', destination='00 Shift/DT/MeanDistr', name='01-SegmentReso-Mean', overlay='')
register_layout(source='DT/02-Segments/01-SigmaRes/SigmaDistr', destination='00 Shift/DT/SigmaDistr', name='02-SegmentReso-Sigma', overlay='')
register_layout(source='DT/05-ChamberEff/EfficiencyGlbSummary', destination='00 Shift/DT/EfficiencyGlbSummary', name='03-EfficiencySummary', overlay='')
