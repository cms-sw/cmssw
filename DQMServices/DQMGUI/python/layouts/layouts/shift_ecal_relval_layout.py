from .adapt_to_new_backend import *
dqmitems={}

def ecallayout(i, p, *rows): i[p] = rows

ecallayout(dqmitems, '00 Shift/Ecal/00 RecHit Spectra',[{'path': 'EcalBarrel/EBOccupancyTask/EBOT rec hit spectrum', 'description': 'Rec hit energy distribution.'}],[{'path': 'EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE -', 'description': 'Rec hit energy distribution.'}, {'path': 'EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE +', 'description': 'Rec hit energy distribution.'}])
ecallayout(dqmitems, '00 Shift/Ecal/01 Number of RecHits',[{'path': 'EcalBarrel/EBOccupancyTask/EBOT number of filtered rec hits in event', 'description': 'Occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'}],[{'path': 'EcalEndcap/EEOccupancyTask/EEOT number of filtered rec hits in event', 'description': 'Occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'}])
ecallayout(dqmitems, '00 Shift/Ecal/02 Mean Timing',[{'path': 'EcalBarrel/EBSummaryClient/EBTMT timing mean 1D summary', 'description': 'Distribution of per-channel timing mean. Channels with entries less than 1 (or 8 in forward region) are not considered.'}],[{'path': 'EcalEndcap/EESummaryClient/EETMT EE - timing mean 1D summary', 'description': 'Distribution of per-channel timing mean. Channels with entries less than 1 (or 8 in forward region) are not considered.'}, {'path': 'EcalEndcap/EESummaryClient/EETMT EE + timing mean 1D summary', 'description': 'Distribution of per-channel timing mean. Channels with entries less than 1 (or 8 in forward region) are not considered.'}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
