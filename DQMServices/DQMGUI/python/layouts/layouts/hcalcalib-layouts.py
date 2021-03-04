from .adapt_to_new_backend import *
dqmitems={}

def hcalcaliblayout(i, p, *rows): i['HcalCalib/Layouts/' + p] = rows

hcalcaliblayout(dqmitems, 'BadQualityvsLS/RAW', [{'path':'HcalCalib/RawTask/BadQualityvsLS/BadQualityvsLS', 'description':"""Distribution of Bad Channels vs Lumi Section <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadMeanvsLS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HB', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadMeanvsLS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HE', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadMeanvsLS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HF', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadMeanvsLS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HO', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BcnMsm/RAW/Electronics/VME', [{'path':'HcalCalib/RawTask/BcnMsm/Electronics/VME', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BcnMsm/RAW/Electronics/uTCA', [{'path':'HcalCalib/RawTask/BcnMsm/Electronics/uTCA', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/depth/depth1', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/depth/depth2', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/depth/depth3', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/depth/depth4', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/RMS/depth/depth1', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/RMS/depth/depth2', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/RMS/depth/depth3', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/RMS/depth/depth4', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/Mean1LS/depth/depth1', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/Mean1LS/depth/depth2', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/Mean1LS/depth/depth3', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/Mean1LS/depth/depth4', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyEAvsLS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HB', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyEAvsLS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HE', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyEAvsLS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HF', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyEAvsLS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HO', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyvsLS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyvsLS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyvsLS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'OccupancyvsLS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQualityvsBX/RAW', [{'path':'HcalCalib/RawTask/BadQualityvsBX/BadQualityvsBX', 'description':"""Distribution of Bad Channels vs Bunch Crossing <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth1', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth2', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth3', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth4', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadRMSvsLS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HB', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadRMSvsLS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HE', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadRMSvsLS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HF', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'NBadRMSvsLS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HO', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1100', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1102', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1104', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1106', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1108', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1110', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1112', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1114', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1116', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1118', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1120', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1122', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED1132', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED724', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED725', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED726', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED727', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED728', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED729', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED730', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/FED/FED731', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth1', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth2', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth3', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth4', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/Missing1LS/depth/depth1', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/Missing1LS/depth/depth2', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/Missing1LS/depth/depth3', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/Missing1LS/depth/depth4', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/RMS1LS/depth/depth1', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/RMS1LS/depth/depth2', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/RMS1LS/depth/depth3', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/RMS1LS/depth/depth4', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/Subdet/HB', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/Subdet/HE', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/Subdet/HF', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/Subdet/HO', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1100', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1100', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1102', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1102', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1104', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1104', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1106', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1106', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1108', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1108', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1110', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1110', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1112', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1112', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1114', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1114', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1116', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1116', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1118', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1118', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1120', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1120', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1122', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1122', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1132', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED1132', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED724', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED724', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED725', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED725', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED726', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED726', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED727', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED727', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED728', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED728', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED729', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED729', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED730', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED730', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW/FED/FED731', [{'path':'HcalCalib/RawTask/SummaryvsLS/FED/FED731', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/Mean/depth/depth1', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/Mean/depth/depth2', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/Mean/depth/depth3', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/Mean/depth/depth4', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1100', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1102', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1104', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1106', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1108', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1110', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1112', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1114', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1116', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1118', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1120', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1122', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED1132', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED724', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED725', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED726', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED727', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED728', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED729', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED730', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing1LS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/Missing1LS/FED/FED731', 'description':"""Missing channels w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/Subdet/HB', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/Subdet/HE', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/Subdet/HF', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/Subdet/HO', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/RMS1LS/Subdet/HB', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/RMS1LS/Subdet/HE', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/RMS1LS/Subdet/HF', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/RMS1LS/Subdet/HO', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1100', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1100', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1102', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1102', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1104', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1104', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1106', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1106', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1108', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1108', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1110', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1110', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1112', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1112', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1114', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1114', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1116', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1116', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1118', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1118', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1120', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1120', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1122', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1122', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED1132', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED1132', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED724', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED724', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED725', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED725', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED726', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED726', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED727', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED727', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED728', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED728', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED729', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED729', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED730', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED730', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/FED/FED731', [{'path':'HcalCalib/RawTask/BadQuality/FED/FED731', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/depth/depth1', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/depth/depth2', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/depth/depth3', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/depth/depth4', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/Mean/Subdet/HB', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/Mean/Subdet/HE', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/Mean/Subdet/HF', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/Mean/Subdet/HO', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/depth/depth1', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/depth/depth2', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/depth/depth3', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad1LS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/RMSBad1LS/depth/depth4', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/depth/depth1', [{'path':'HcalCalib/RawTask/BadQuality/depth/depth1', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/depth/depth2', [{'path':'HcalCalib/RawTask/BadQuality/depth/depth2', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/depth/depth3', [{'path':'HcalCalib/RawTask/BadQuality/depth/depth3', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'BadQuality/RAW/depth/depth4', [{'path':'HcalCalib/RawTask/BadQuality/depth/depth4', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1100', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1102', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1104', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1106', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1108', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1110', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1112', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1114', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1116', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1118', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1120', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1122', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED1132', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED724', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED725', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED726', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED727', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED728', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED729', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED730', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/RMS/FED/FED731', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/Missing/depth/depth1', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/Missing/depth/depth2', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/Missing/depth/depth3', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/Missing/depth/depth4', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1100', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1102', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1104', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1106', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1108', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1110', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1112', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1114', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1116', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1118', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1120', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1122', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED1132', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED724', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED725', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED726', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED727', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED728', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED729', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED730', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/FED/FED731', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1100', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1102', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1104', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1106', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1108', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1110', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1112', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1114', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1116', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1118', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1120', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1122', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED1132', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED724', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED725', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED726', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED727', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED728', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED729', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED730', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad1LS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/MeanBad1LS/FED/FED731', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1100', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1102', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1104', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1106', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1108', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1110', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1112', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1114', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1116', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1118', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1120', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1122', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED1132', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED724', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED725', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED726', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED727', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED728', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED729', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED730', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef1LS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/MeanDBRef1LS/FED/FED731', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1100', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1102', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1104', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1106', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1108', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1110', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1112', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1114', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1116', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1118', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1120', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1122', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED1132', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED724', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED725', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED726', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED727', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED728', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED729', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED730', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/MeanBad/FED/FED731', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1100', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1102', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1104', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1106', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1108', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1110', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1112', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1114', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1116', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1118', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1120', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1122', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED1132', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED724', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED725', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED726', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED727', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED728', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED729', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED730', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS1LS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/RMS1LS/FED/FED731', 'description':"""Pedestal RMS Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/depth/depth1', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/depth/depth2', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/depth/depth3', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/depth/depth4', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1100', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1102', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1104', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1106', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1108', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1110', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1112', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1114', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1116', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1118', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1120', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1122', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED1132', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED724', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED725', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED726', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED727', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED728', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED729', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED730', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Missing/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/Missing/FED/FED731', 'description':"""Missing channels w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1100', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1102', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1104', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1106', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1108', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1110', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1112', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1114', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1116', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1118', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1120', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1122', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED1132', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED724', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED725', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED726', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED727', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED728', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED729', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED730', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/RMSBad/FED/FED731', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1100', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1102', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1104', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1106', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1108', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1110', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1112', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1114', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1116', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1118', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1120', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1122', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED1132', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED724', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED725', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED726', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED727', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED728', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED729', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED730', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef1LS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/RMSDBRef1LS/FED/FED731', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/PEDESTAL', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/SummaryvsLS', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MissingvsLS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HB', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MissingvsLS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HE', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MissingvsLS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HF', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MissingvsLS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HO', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1100', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1102', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1104', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1106', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1108', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1110', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1112', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1114', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1116', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1118', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1120', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1122', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED1132', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED724', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED725', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED726', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED727', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED728', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED729', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED730', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/Mean1LS/FED/FED731', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1100', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1102', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1104', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1106', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1108', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1110', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1112', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1114', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1116', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1118', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1120', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1122', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED1132', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED724', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED725', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED726', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED727', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED728', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED729', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED730', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/Mean/FED/FED731', 'description':"""Pedestal Mean Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/RMS/Subdet/HB', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/RMS/Subdet/HE', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/RMS/Subdet/HF', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/RMS/Subdet/HO', 'description':"""Pedestal RMS Distributions Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'EvnMsm/RAW/Electronics/VME', [{'path':'HcalCalib/RawTask/EvnMsm/Electronics/VME', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'EvnMsm/RAW/Electronics/uTCA', [{'path':'HcalCalib/RawTask/EvnMsm/Electronics/uTCA', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HB', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HE', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HF', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HO', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/MeanBad/depth/depth1', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/MeanBad/depth/depth2', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/MeanBad/depth/depth3', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanBad/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/MeanBad/depth/depth4', 'description':"""Bad Pedestal Means w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'SummaryvsLS/RAW', [{'path':'HcalCalib/RawTask/SummaryvsLS/SummaryvsLS', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1100', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1102', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1104', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1106', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1108', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1110', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1112', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1114', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1116', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1118', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1120', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1122', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED1132', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED724', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED725', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED726', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED727', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED728', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED729', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED730', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/RMSDBRef/FED/FED731', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1100', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1100', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1102', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1102', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1104', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1104', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1106', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1106', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1108', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1108', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1110', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1110', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1112', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1112', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1114', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1114', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1116', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1116', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1118', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1118', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1120', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1120', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1122', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1122', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED1132', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED1132', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED724', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED724', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED725', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED725', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED726', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED726', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED727', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED727', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED728', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED728', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED729', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED729', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED730', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED730', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'MeanDBRef/PEDESTAL/FED/FED731', [{'path':'HcalCalib/PedestalTask/MeanDBRef/FED/FED731', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/depth/depth1', [{'path':'HcalCalib/PedestalTask/RMSBad/depth/depth1', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/depth/depth2', [{'path':'HcalCalib/PedestalTask/RMSBad/depth/depth2', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/depth/depth3', [{'path':'HcalCalib/PedestalTask/RMSBad/depth/depth3', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSBad/PEDESTAL/depth/depth4', [{'path':'HcalCalib/PedestalTask/RMSBad/depth/depth4', 'description':"""Bad Pedestal RMS w.r.t. CondDB. Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HB', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HE', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HF', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'RMSDBRef/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HO', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/Subdet/HB', [{'path':'HcalCalib/PedestalTask/Mean1LS/Subdet/HB', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/Subdet/HE', [{'path':'HcalCalib/PedestalTask/Mean1LS/Subdet/HE', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/Subdet/HF', [{'path':'HcalCalib/PedestalTask/Mean1LS/Subdet/HF', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, 'Mean1LS/PEDESTAL/Subdet/HO', [{'path':'HcalCalib/PedestalTask/Mean1LS/Subdet/HO', 'description':"""Pedestal Mean Distributions Statistics over 1LS only is combined <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '00 Run Summary', [{'path':'HcalCalib/PedestalTask/SummaryvsLS/SummaryvsLS', 'description':"""Calibration Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/RawTask/SummaryvsLS/SummaryvsLS', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '01 Pedestal Mean vs CondDB', [{'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HB', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HE', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HF', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/MeanDBRef/Subdet/HO', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '02 Pedestal Mean vs CondDB', [{'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth1', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth2', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth3', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/MeanDBRef/depth/depth4', 'description':"""Comparison of Pedestal Mean with CondBDStatistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '03 Pedestal RMS vs CondDB', [{'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HB', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HE', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HF', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/RMSDBRef/Subdet/HO', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '04 Pedestal RMS vs CondDB', [{'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth1', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth2', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth3', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/RMSDBRef/depth/depth4', 'description':"""Comparison of Pedestal RMS with the CondDB Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '05 Pedestal Missing vs LS', [{'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HB', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HE', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HF', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/MissingvsLS/Subdet/HO', 'description':"""Missing channels vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '06 Pedestal Occupancy vs LS', [{'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS (Number of unique channels read out per LS) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '07 Pedestal #Bad Mean Chs vs LS', [{'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HB', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HE', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HF', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/NBadMeanvsLS/Subdet/HO', 'description':"""Number of channels with Bad Mean w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '08 Pedestal #Bad RMS Chs vs LS', [{'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HB', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HE', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HF', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/NBadRMSvsLS/Subdet/HO', 'description':"""Number of channels with bad RMSs w.r.t. CondDB vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '09 Pedestal Occupancy EA vs LS ', [{'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HB', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HE', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HF', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/PedestalTask/OccupancyEAvsLS/Subdet/HO', 'description':"""Occupancy vs LS. Averaged over all events per LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Pedestal_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '10 RAW BadQuality vs BX (LS)', [{'path':'HcalCalib/RawTask/BadQualityvsBX/BadQualityvsBX', 'description':"""Distribution of Bad Channels vs Bunch Crossing <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/RawTask/BadQualityvsLS/BadQualityvsLS', 'description':"""Distribution of Bad Channels vs Lumi Section <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '11 RAW Bcn(Evn) Mismatches', [{'path':'HcalCalib/RawTask/BcnMsm/Electronics/VME', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/RawTask/BcnMsm/Electronics/uTCA', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}], [{'path':'HcalCalib/RawTask/EvnMsm/Electronics/VME', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'HcalCalib/RawTask/EvnMsm/Electronics/uTCA', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '12 LED pulse shape', 
	[
		{
			'path':'HcalCalib/LEDTask/ADCvsTS/SubdetPM/HEM',
			'description':"""Pulse shape (ADC vs TS) from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, 
		{
			'path':'HcalCalib/LEDTask/ADCvsTS/SubdetPM/HEP',
			'description':"""Pulse shape (ADC vs TS) from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'HcalCalib/LEDTask/ADCvsTS/SubdetPM/HFM',
			'description':"""Pulse shape (ADC vs TS) from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, 
		{
			'path':'HcalCalib/LEDTask/ADCvsTS/SubdetPM/HFP',
			'description':"""Pulse shape (ADC vs TS) from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	]
)

hcalcaliblayout(dqmitems, '13 LED pin diode amplitude', [{'path':'HcalCalib/LEDTask/LED_ADCvsBX', 'description':"""LED pin diode amplitude <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""}])

hcalcaliblayout(dqmitems, '14 LED SignalMean', 
	[
		{
			'path':'HcalCalib/LEDTask/SignalMean/depth/depth1', 
			'description':"""Mean signal per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/SignalMean/depth/depth2', 
			'description':"""Mean signal per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/SignalMean/depth/depth3', 
			'description':"""Mean signal per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'HcalCalib/LEDTask/SignalMean/depth/depth4', 
			'description':"""Mean signal per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/SignalMean/depth/depth5', 
			'description':"""Mean signal per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/SignalMean/depth/depth6', 
			'description':"""Mean signal per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'HcalCalib/LEDTask/SignalMean/depth/depth7', 
			'description':"""Mean signal per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	]
)

hcalcaliblayout(dqmitems, '15 LED TDCTime', 
	[
		{
			'path':'HcalCalib/LEDTask/TDCTime/depth/depth1', 
			'description':"""Average timing (TDC based) per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/TDCTime/depth/depth2', 
			'description':"""Average timing (TDC based) per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/TDCTime/depth/depth3', 
			'description':"""Average timing (TDC based) per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'HcalCalib/LEDTask/TDCTime/depth/depth4', 
			'description':"""Average timing (TDC based) per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/TDCTime/depth/depth5', 
			'description':"""Average timing (TDC based) per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/LEDTask/TDCTime/depth/depth6', 
			'description':"""Average timing (TDC based) per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'HcalCalib/LEDTask/TDCTime/depth/depth7', 
			'description':"""Average timing (TDC based) per channel from LED events<a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#LED_Task_Description'>Details...</a>"""
		}
	]
)

hcalcaliblayout(dqmitems, "16 Laser summary flags", 
	[
		{
			'path':'HcalCalib/HBHEHPDTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/HBPMegaTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/HBMMegaTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/HEPMegaTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'HcalCalib/HEMMegaTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/HFTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/HOTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}, {
			'path':'HcalCalib/HFRaddamTask/SummaryvsLS/SummaryvsLS', 
			'description':"""Laser summary flags <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
		}
	], 
)

# Create laser plots with a loop, due to multiple LaserTasks. Require a starting value for the plot index.
plot_index = 16
subdetpms = {
	"HBHEHPD":["HBP", "HBM", "HEP", "HEM"], "HBPMega":["HBP"], "HBMMega":["HBM"], "HEPMega":["HEP"], "HEMMega":["HEM"], "HFRaddam":["HF"], "HF":["HF"], "HO":["HO"]
}

for laser_position in ["HBHEHPD", "HBPMega", "HBMMega", "HEPMega", "HEMMega", "HFRaddam", "HF", "HO"]:

	row1 = []
	plots1 = ["HcalCalib/{}Task/LaserMonSumQ_LS/LaserMonSumQ_LS".format(laser_position), "HcalCalib/{}Task/LaserMonTiming/LaserMonTiming".format(laser_position)]
	for plot in plots1:
		row1.append({"path":plot, "description":"""LaserMon amplitude and timing <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""})
	plots2 = []
	row2 = []
	for subdetpm in subdetpms[laser_position]:
		plots2.append("HcalCalib/{}Task/TimingDiff_DigiMinusLaserMon/SubdetPM/{}".format(laser_position, subdetpm))
	for plot in plots2:
		row2.append({"path":plot, "description":"""Timing vs RBX vs LS, relative to LaserMon <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""})
	plot_index += 1
	hcalcaliblayout(dqmitems, "{} LaserMon, {}".format(plot_index, laser_position), row1, row2)

	plot_index += 1
	hcalcaliblayout(dqmitems, "{} Laser SignalMean, {}".format(plot_index, laser_position), 
		[
			{
				'path':'HcalCalib/{}Task/SignalMean/depth/depth1'.format(laser_position), 
				'description':"""Laser SignalMean <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
			}, {
				'path':'HcalCalib/{}Task/SignalMean/depth/depth2'.format(laser_position), 
				'description':"""Laser SignalMean <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
			}, {
				'path':'HcalCalib/{}Task/SignalMean/depth/depth3'.format(laser_position), 
				'description':"""Laser SignalMean <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
			}
		], 
		[
			{
				'path':'HcalCalib/{}Task/SignalMean/depth/depth4'.format(laser_position), 
				'description':"""Laser SignalMean <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
			}, {
				'path':'HcalCalib/{}Task/SignalMean/depth/depth5'.format(laser_position), 
				'description':"""Laser SignalMean <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
			}, {
				'path':'HcalCalib/{}Task/SignalMean/depth/depth6'.format(laser_position), 
				'description':"""Laser SignalMean <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
			}
		], 
		[
			{
				'path':'HcalCalib/{}Task/SignalMean/depth/depth7'.format(laser_position), 
				'description':"""Laser SignalMean <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Laser_Task_Description'>Details...</a>"""
			}
		]
	)

apply_dqm_items_to_new_back_end(dqmitems, __file__)
