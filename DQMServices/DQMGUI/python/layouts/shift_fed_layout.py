from ..layouts.layout_manager import register_layout

register_layout(source='FED/EventInfo/reportSummaryMap', destination='00 Shift/FED/reportSummaryMap', name='00 Report Summary Map', description='FED summary for each subdetector', overlay='')
register_layout(source='FED/FEDIntegrity_EvF/FedEntries', destination='00 Shift/FED/FedEntries', name='01 FED Integrity Check', description='Number of entries vs FED ID', overlay='')
register_layout(source='FED/FEDIntegrity_EvF/FedFatal', destination='00 Shift/FED/FedFatal', name='01 FED Integrity Check', description='Number of fatal errors vs FED ID', overlay='')
register_layout(source='FED/FEDIntegrity_EvF/FedNonFatal', destination='00 Shift/FED/FedNonFatal', name='01 FED Integrity Check', description='Number of non-fatal errors vs FED ID', overlay='')
