from ..layouts.layout_manager import register_layout

register_layout(source='Muons/EventInfo/reportSummaryMap', destination='00 Shift/Muons/reportSummaryMap', name='00-reportSummary', overlay='')
register_layout(source='Muons/TestSummary/kinematicsSummaryMap', destination='00 Shift/Muons/kinematicsSummaryMap', name='01-kinematicsSummary', overlay='')
register_layout(source='Muons/TestSummary/residualsSummaryMap', destination='00 Shift/Muons/residualsSummaryMap', name='02-residualsSummary', overlay='')
register_layout(source='Muons/TestSummary/energySummaryMap', destination='00 Shift/Muons/energySummaryMap', name='03-energySummary', overlay='')
register_layout(source='Muons/TestSummary/muonIdSummaryMap', destination='00 Shift/Muons/muonIdSummaryMap', name='04-muonIdSummary', overlay='')
register_layout(source='Muons/TestSummary/molteplicitySummaryMap', destination='00 Shift/Muons/molteplicitySummaryMap', name='05-molteplicitySummary', overlay='')
