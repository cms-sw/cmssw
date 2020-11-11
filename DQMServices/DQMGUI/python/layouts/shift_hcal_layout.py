from ..layouts.layout_manager import register_layout

register_layout(source='Hcal/TPTask/SummaryvsLS/SummaryvsLS', destination='00 Shift/Hcal/SummaryvsLS', name='00 Run Summary', overlay='')
register_layout(source='Hcal/DigiTask/SummaryvsLS/SummaryvsLS', destination='00 Shift/Hcal/SummaryvsLS', name='00 Run Summary', overlay='')
register_layout(source='Hcal/RawTask/SummaryvsLS/SummaryvsLS', destination='00 Shift/Hcal/SummaryvsLS', name='00 Run Summary', overlay='')
register_layout(source='Hcal2/RecHitTask/SummaryvsLS/SummaryvsLS', destination='00 Shift/Hcal/SummaryvsLS', name='00 Run Summary', overlay='')
register_layout(source='Hcal/EventInfo/runSummary', destination='00 Shift/Hcal/runSummary', name='01 Run Summary', overlay='')
