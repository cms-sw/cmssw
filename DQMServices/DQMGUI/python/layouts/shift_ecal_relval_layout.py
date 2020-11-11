from ..layouts.layout_manager import register_layout

register_layout(source='EcalBarrel/EBOccupancyTask/EBOT rec hit spectrum', destination='00 Shift/Ecal/EBOT rec hit spectrumEBOT rec hit spectrum', name='00 RecHit Spectra', overlay='')
register_layout(source='EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE -', destination='00 Shift/Ecal/EEOT rec hit spectrum EE -EEOT rec hit spectrum EE -', name='00 RecHit Spectra', overlay='')
register_layout(source='EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE +', destination='00 Shift/Ecal/EEOT rec hit spectrum EE +EEOT rec hit spectrum EE +', name='00 RecHit Spectra', overlay='')
register_layout(source='EcalBarrel/EBOccupancyTask/EBOT number of filtered rec hits in event', destination='00 Shift/Ecal/EBOT number of filtered rec hits in eventEBOT number of filtered rec hits in event', name='01 Number of RecHits', overlay='')
register_layout(source='EcalEndcap/EEOccupancyTask/EEOT number of filtered rec hits in event', destination='00 Shift/Ecal/EEOT number of filtered rec hits in eventEEOT number of filtered rec hits in event', name='01 Number of RecHits', overlay='')
register_layout(source='EcalBarrel/EBSummaryClient/EBTMT timing mean 1D summary', destination='00 Shift/Ecal/EBTMT timing mean 1D summaryEBTMT timing mean 1D summary', name='02 Mean Timing', overlay='')
register_layout(source='EcalEndcap/EESummaryClient/EETMT EE - timing mean 1D summary', destination='00 Shift/Ecal/EETMT EE - timing mean 1D summaryEETMT EE - timing mean 1D summary', name='02 Mean Timing', overlay='')
register_layout(source='EcalEndcap/EESummaryClient/EETMT EE + timing mean 1D summary', destination='00 Shift/Ecal/EETMT EE + timing mean 1D summaryEETMT EE + timing mean 1D summary', name='02 Mean Timing', overlay='')
