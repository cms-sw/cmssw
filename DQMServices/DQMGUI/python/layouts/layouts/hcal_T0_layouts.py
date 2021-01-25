from .adapt_to_new_backend import *
dqmitems={}

def hcallayout(i, p, *rows): i['Hcal/Layouts/' + p] = rows


hcallayout(dqmitems, 'EtEmul/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtEmul/TTSubdet/HBHE', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtEmul/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtEmul/TTSubdet/HF', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmul/TP/Electronics/VME', [{'path':'Hcal/TPTask/OccupancyEmul/Electronics/VME', 'description':"""Occupancy Distributions for Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmul/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/OccupancyEmul/Electronics/uTCA', 'description':"""Occupancy Distributions for Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQualityvsBX/RAW', [{'path':'Hcal/RawTask/BadQualityvsBX/BadQualityvsBX', 'description':"""Distribution of Bad Channels vs Bunch Crossing <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth1', [{'path':'Hcal/RecHitTask/TimingCut/depth/depth1', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth2', [{'path':'Hcal/RecHitTask/TimingCut/depth/depth2', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth3', [{'path':'Hcal/RecHitTask/TimingCut/depth/depth3', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth4', [{'path':'Hcal/RecHitTask/TimingCut/depth/depth4', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Summary/RECO', [{'path':'Hcal/RecoRunHarvesting/Summary/Summary', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br>WHITE color stands for INAPPLICABLE flag<br>Flag(Y) vs FED(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HBM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HBP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HEM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HEP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HFM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HFP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HOM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HOP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyData/TP/Electronics/VME', [{'path':'Hcal/TPTask/OccupancyData/Electronics/VME', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyData/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/OccupancyData/Electronics/uTCA', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtData/TTSubdet/HBHE', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtData/TTSubdet/HF', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HB', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HE', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HF', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HO', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP/Electronics/VME', [{'path':'Hcal/TPTask/FGMsm/Electronics/VME', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/FGMsm/Electronics/uTCA', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatio/TP', [{'path':'Hcal/TPTask/EtCorrRatio/EtCorrRatio', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1100', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1100', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1102', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1102', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1104', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1104', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1106', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1106', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1108', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1108', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1110', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1110', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1112', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1112', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1114', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1114', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1116', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1116', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1118', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1118', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1120', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1120', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1122', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1122', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1132', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED1132', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED724', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED724', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED725', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED725', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED726', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED726', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED727', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED727', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED728', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED728', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED729', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED729', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED730', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED730', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED731', [{'path':'Hcal/RecHitTask/Occupancy/FED/FED731', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP/Electronics/VME', [{'path':'Hcal/TPTask/EtMsm/Electronics/VME', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/EtMsm/Electronics/uTCA', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutData/TP', [{'path':'Hcal/TPTask/OccupancyCutData/OccupancyCutData', 'description':"""Occupancy Distributions for Data with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP', [{'path':'Hcal/TPTask/FGMsm/FGMsm', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/Electronics/VME', [{'path':'Hcal/RecHitTask/OccupancyCut/Electronics/VME', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/Electronics/uTCA', [{'path':'Hcal/RecHitTask/OccupancyCut/Electronics/uTCA', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/SummaryvsLS', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth1', [{'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth1', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth2', [{'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth2', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth3', [{'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth3', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth4', [{'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth4', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HB', [{'path':'Hcal/RecHitTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HE', [{'path':'Hcal/RecHitTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HF', [{'path':'Hcal/RecHitTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HO', [{'path':'Hcal/RecHitTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HB', [{'path':'Hcal/RecHitTask/Energy/Subdet/HB', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HE', [{'path':'Hcal/RecHitTask/Energy/Subdet/HE', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HF', [{'path':'Hcal/RecHitTask/Energy/Subdet/HF', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HO', [{'path':'Hcal/RecHitTask/Energy/Subdet/HO', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1100', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1102', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1104', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1106', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1108', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1110', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1112', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1114', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1116', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1118', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1120', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1122', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/Occupancy/FED/FED1132', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/Occupancy/FED/FED724', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/Occupancy/FED/FED725', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/Occupancy/FED/FED726', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/Occupancy/FED/FED727', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/Occupancy/FED/FED728', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/Occupancy/FED/FED729', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/Occupancy/FED/FED730', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/Occupancy/FED/FED731', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1100', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1100', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1102', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1102', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1104', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1104', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1106', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1106', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1108', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1108', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1110', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1110', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1112', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1112', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1114', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1114', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1116', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1116', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1118', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1118', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1120', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1120', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1122', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1122', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1132', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED1132', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED724', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED724', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED725', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED725', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED726', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED726', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED727', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED727', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED728', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED728', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED729', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED729', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED730', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED730', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED731', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/FED/FED731', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutData/TP/Electronics/VME', [{'path':'Hcal/TPTask/OccupancyCutData/Electronics/VME', 'description':"""Occupancy Distributions for Data with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutData/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/OccupancyCutData/Electronics/uTCA', 'description':"""Occupancy Distributions for Data with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtEmul/TP/Electronics/VME', [{'path':'Hcal/TPTask/EtEmul/Electronics/VME', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtEmul/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/EtEmul/Electronics/uTCA', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1100', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1102', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1104', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1106', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1108', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1110', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1112', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1114', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1116', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1118', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1120', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1122', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1132', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED724', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED725', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED726', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED727', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED728', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED729', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED730', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsLS/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED731', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmul/TP', [{'path':'Hcal/TPTask/OccupancyCutEmul/OccupancyCutEmul', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/Electronics/VME', [{'path':'Hcal/DigiTask/OccupancyCut/Electronics/VME', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/Electronics/uTCA', [{'path':'Hcal/DigiTask/OccupancyCut/Electronics/uTCA', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EvnMsm/RAW/Electronics/VME', [{'path':'Hcal/RawRunHarvesting/EvnMsm/Electronics/VME', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EvnMsm/RAW/Electronics/uTCA', [{'path':'Hcal/RawRunHarvesting/EvnMsm/Electronics/uTCA', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnEmul/TP', [{'path':'Hcal/TPTask/MsnEmul/MsnEmul', 'description':"""Distribution of channels missing from Emulator w.r.t. Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/depth/depth1', [{'path':'Hcal/DigiRunHarvesting/Dead/depth/depth1', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/depth/depth2', [{'path':'Hcal/DigiRunHarvesting/Dead/depth/depth2', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/depth/depth3', [{'path':'Hcal/DigiRunHarvesting/Dead/depth/depth3', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/depth/depth4', [{'path':'Hcal/DigiRunHarvesting/Dead/depth/depth4', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth1', [{'path':'Hcal/RecHitTask/Occupancy/depth/depth1', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth2', [{'path':'Hcal/RecHitTask/Occupancy/depth/depth2', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth3', [{'path':'Hcal/RecHitTask/Occupancy/depth/depth3', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth4', [{'path':'Hcal/RecHitTask/Occupancy/depth/depth4', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BcnMsm/RAW/Electronics/VME', [{'path':'Hcal/RawRunHarvesting/BcnMsm/Electronics/VME', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BcnMsm/RAW/Electronics/uTCA', [{'path':'Hcal/RawRunHarvesting/BcnMsm/Electronics/uTCA', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP', [{'path':'Hcal/TPTask/EtData/EtData', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/Electronics/VME', [{'path':'Hcal/DigiTask/TimingCut/Electronics/VME', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/Electronics/uTCA', [{'path':'Hcal/DigiTask/TimingCut/Electronics/uTCA', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnData/TP/Electronics/VME', [{'path':'Hcal/TPTask/MsnData/Electronics/VME', 'description':"""Distribution of channels missing from Data w.r.t. Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnData/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/MsnData/Electronics/uTCA', 'description':"""Distribution of channels missing from Data w.r.t. Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtEmul/TP', [{'path':'Hcal/TPTask/EtEmul/EtEmul', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HBM', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HBM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HBP', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HBP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HEM', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HEM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HEP', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HEP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HFM', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HFM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HFP', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HFP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HOM', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HOM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HOP', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HOP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Summary/TP', [{'path':'Hcal/TPRunHarvesting/Summary/Summary', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br>WHITE color stands for INAPPLICABLE flag<br>Flag(Y) vs FED(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorr/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCorr/TTSubdet/HBHE', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorr/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCorr/TTSubdet/HF', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/Electronics/VME', [{'path':'Hcal/RecHitTask/TimingCut/Electronics/VME', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/Electronics/uTCA', [{'path':'Hcal/RecHitTask/TimingCut/Electronics/uTCA', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1100', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1102', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1104', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1106', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1108', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1110', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1112', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1114', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1116', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1118', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1120', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1122', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/DigiSize/FED/FED1132', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/DigiSize/FED/FED724', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/DigiSize/FED/FED725', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/DigiSize/FED/FED726', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/DigiSize/FED/FED727', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/DigiSize/FED/FED728', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/DigiSize/FED/FED729', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/DigiSize/FED/FED730', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSize/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/DigiSize/FED/FED731', 'description':"""Digi Size Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmul/TP', [{'path':'Hcal/TPTask/OccupancyEmul/OccupancyEmul', 'description':"""Occupancy Distributions for Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmul/TP/Electronics/VME', [{'path':'Hcal/TPTask/OccupancyCutEmul/Electronics/VME', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmul/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/OccupancyCutEmul/Electronics/uTCA', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGCorr/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/FGCorr/TTSubdet/HBHE', 'description':"""Correction of Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGCorr/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/FGCorr/TTSubdet/HF', 'description':"""Correction of Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/HBHEPartition/HBHEa', [{'path':'Hcal/RecHitTask/TimingCut/HBHEPartition/HBHEa', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/HBHEPartition/HBHEb', [{'path':'Hcal/RecHitTask/TimingCut/HBHEPartition/HBHEb', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/HBHEPartition/HBHEc', [{'path':'Hcal/RecHitTask/TimingCut/HBHEPartition/HBHEc', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1100', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1100', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1102', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1102', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1104', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1104', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1106', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1106', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1108', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1108', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1110', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1110', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1112', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1112', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1114', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1114', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1116', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1116', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1118', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1118', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1120', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1120', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1122', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1122', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1132', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED1132', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED724', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED724', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED725', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED725', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED726', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED726', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED727', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED727', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED728', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED728', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED729', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED729', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED730', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED730', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED731', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/FED/FED731', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HBM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HBP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HEM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HEP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HFM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HFP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HOM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HOP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatio/TP/Electronics/VME', [{'path':'Hcal/TPTask/EtCorrRatio/Electronics/VME', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatio/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/EtCorrRatio/Electronics/uTCA', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW', [{'path':'Hcal/RawRunHarvesting/SummaryvsLS/SummaryvsLS', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1100', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1100', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1102', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1102', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1104', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1104', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1106', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1106', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1108', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1108', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1110', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1110', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1112', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1112', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1114', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1114', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1116', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1116', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1118', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1118', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1120', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1120', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1122', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1122', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED1132', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED1132', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED724', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED724', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED725', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED725', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED726', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED726', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED727', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED727', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED728', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED728', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED729', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED729', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED730', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED730', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Dead/DIGI/FED/FED731', [{'path':'Hcal/DigiRunHarvesting/Dead/FED/FED731', 'description':"""Dead Cells over a period of 1 Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth1', [{'path':'Hcal/DigiTask/TimingCut/depth/depth1', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth2', [{'path':'Hcal/DigiTask/TimingCut/depth/depth2', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth3', [{'path':'Hcal/DigiTask/TimingCut/depth/depth3', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth4', [{'path':'Hcal/DigiTask/TimingCut/depth/depth4', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/depth/depth1', [{'path':'Hcal/DigiTask/OccupancyCut/depth/depth1', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/depth/depth2', [{'path':'Hcal/DigiTask/OccupancyCut/depth/depth2', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/depth/depth3', [{'path':'Hcal/DigiTask/OccupancyCut/depth/depth3', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/depth/depth4', [{'path':'Hcal/DigiTask/OccupancyCut/depth/depth4', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/Electronics/VME', [{'path':'Hcal/DigiTask/Occupancy/Electronics/VME', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/Electronics/uTCA', [{'path':'Hcal/DigiTask/Occupancy/Electronics/uTCA', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/fC/SubdetPM/HBM', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/fC/SubdetPM/HBP', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/fC/SubdetPM/HEM', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/fC/SubdetPM/HEP', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/fC/SubdetPM/HFM', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/fC/SubdetPM/HFP', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/fC/SubdetPM/HOM', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'fC/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/fC/SubdetPM/HOP', 'description':"""fC per TS distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQualityvsLS/RAW', [{'path':'Hcal/RawTask/BadQualityvsLS/BadQualityvsLS', 'description':"""Distribution of Bad Channels vs Lumi Section <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/Electronics/VME', [{'path':'Hcal/RecHitTask/Occupancy/Electronics/VME', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/Electronics/uTCA', [{'path':'Hcal/RecHitTask/Occupancy/Electronics/uTCA', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth1', [{'path':'Hcal/RecHitTask/Energy/depth/depth1', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth2', [{'path':'Hcal/RecHitTask/Energy/depth/depth2', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth3', [{'path':'Hcal/RecHitTask/Energy/depth/depth3', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth4', [{'path':'Hcal/RecHitTask/Energy/depth/depth4', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth1', [{'path':'Hcal/DigiTask/SumQ/depth/depth1', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth2', [{'path':'Hcal/DigiTask/SumQ/depth/depth2', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth3', [{'path':'Hcal/DigiTask/SumQ/depth/depth3', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth4', [{'path':'Hcal/DigiTask/SumQ/depth/depth4', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyData/TP', [{'path':'Hcal/TPTask/OccupancyData/OccupancyData', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HBM', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HBM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HBP', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HBP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HEM', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HEM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HEP', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HEP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HFM', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HFM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HFP', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HFP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HOM', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HOM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HOP', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HOP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1100', [{'path':'Hcal/RawTask/BadQuality/FED/FED1100', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1102', [{'path':'Hcal/RawTask/BadQuality/FED/FED1102', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1104', [{'path':'Hcal/RawTask/BadQuality/FED/FED1104', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1106', [{'path':'Hcal/RawTask/BadQuality/FED/FED1106', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1108', [{'path':'Hcal/RawTask/BadQuality/FED/FED1108', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1110', [{'path':'Hcal/RawTask/BadQuality/FED/FED1110', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1112', [{'path':'Hcal/RawTask/BadQuality/FED/FED1112', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1114', [{'path':'Hcal/RawTask/BadQuality/FED/FED1114', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1116', [{'path':'Hcal/RawTask/BadQuality/FED/FED1116', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1118', [{'path':'Hcal/RawTask/BadQuality/FED/FED1118', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1120', [{'path':'Hcal/RawTask/BadQuality/FED/FED1120', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1122', [{'path':'Hcal/RawTask/BadQuality/FED/FED1122', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED1132', [{'path':'Hcal/RawTask/BadQuality/FED/FED1132', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED724', [{'path':'Hcal/RawTask/BadQuality/FED/FED724', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED725', [{'path':'Hcal/RawTask/BadQuality/FED/FED725', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED726', [{'path':'Hcal/RawTask/BadQuality/FED/FED726', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED727', [{'path':'Hcal/RawTask/BadQuality/FED/FED727', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED728', [{'path':'Hcal/RawTask/BadQuality/FED/FED728', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED729', [{'path':'Hcal/RawTask/BadQuality/FED/FED729', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED730', [{'path':'Hcal/RawTask/BadQuality/FED/FED730', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/FED/FED731', [{'path':'Hcal/RawTask/BadQuality/FED/FED731', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1100', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1102', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1104', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1106', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1108', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1110', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1112', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1114', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1116', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1118', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1120', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1122', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED1132', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED724', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED725', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED726', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED727', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED728', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED729', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED730', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/OccupancyCut/FED/FED731', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth1', [{'path':'Hcal/RecHitTask/OccupancyCut/depth/depth1', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth2', [{'path':'Hcal/RecHitTask/OccupancyCut/depth/depth2', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth3', [{'path':'Hcal/RecHitTask/OccupancyCut/depth/depth3', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth4', [{'path':'Hcal/RecHitTask/OccupancyCut/depth/depth4', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1100', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1100', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1102', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1102', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1104', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1104', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1106', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1106', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1108', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1108', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1110', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1110', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1112', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1112', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1114', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1114', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1116', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1116', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1118', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1118', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1120', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1120', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1122', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1122', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1132', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED1132', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED724', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED724', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED725', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED725', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED726', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED726', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED727', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED727', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED728', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED728', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED729', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED729', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED730', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED730', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED731', [{'path':'Hcal/RecHitTask/TimingCut/FED/FED731', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1100', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1100', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1102', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1102', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1104', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1104', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1106', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1106', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1108', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1108', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1110', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1110', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1112', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1112', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1114', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1114', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1116', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1116', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1118', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1118', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1120', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1120', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1122', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1122', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1132', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED1132', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED724', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED724', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED725', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED725', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED726', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED726', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED727', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED727', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED728', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED728', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED729', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED729', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED730', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED730', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED731', [{'path':'Hcal/RecHitTask/OccupancyCut/FED/FED731', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1100', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1100', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1102', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1102', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1104', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1104', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1106', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1106', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1108', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1108', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1110', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1110', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1112', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1112', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1114', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1114', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1116', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1116', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1118', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1118', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1120', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1120', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1122', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1122', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1132', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1132', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED724', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED724', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED725', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED725', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED726', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED726', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED727', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED727', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED728', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED728', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED729', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED729', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED730', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED730', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED731', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED731', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/Shape/FED/FED1100', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/Shape/FED/FED1102', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/Shape/FED/FED1104', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/Shape/FED/FED1106', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/Shape/FED/FED1108', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/Shape/FED/FED1110', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/Shape/FED/FED1112', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/Shape/FED/FED1114', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/Shape/FED/FED1116', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/Shape/FED/FED1118', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/Shape/FED/FED1120', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/Shape/FED/FED1122', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/Shape/FED/FED1132', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/Shape/FED/FED724', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/Shape/FED/FED725', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/Shape/FED/FED726', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/Shape/FED/FED727', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/Shape/FED/FED728', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/Shape/FED/FED729', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/Shape/FED/FED730', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Shape/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/Shape/FED/FED731', 'description':"""Signal Shape.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/Electronics/VME', [{'path':'Hcal/TPTask/EtData/Electronics/VME', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/EtData/Electronics/uTCA', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP', [{'path':'Hcal/TPTask/EtMsm/EtMsm', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnEmul/TP/Electronics/VME', [{'path':'Hcal/TPTask/MsnEmul/Electronics/VME', 'description':"""Distribution of channels missing from Emulator w.r.t. Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnEmul/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/MsnEmul/Electronics/uTCA', 'description':"""Distribution of channels missing from Emulator w.r.t. Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth1', [{'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth1', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth2', [{'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth2', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth3', [{'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth3', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth4', [{'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth4', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1100', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1102', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1104', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1106', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1108', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1110', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1112', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1114', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1116', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1118', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1120', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1122', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/TimingCut/FED/FED1132', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/TimingCut/FED/FED724', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/TimingCut/FED/FED725', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/TimingCut/FED/FED726', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/TimingCut/FED/FED727', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/TimingCut/FED/FED728', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/TimingCut/FED/FED729', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/TimingCut/FED/FED730', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/TimingCut/FED/FED731', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnData/TP', [{'path':'Hcal/TPTask/MsnData/MsnData', 'description':"""Distribution of channels missing from Data w.r.t. Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '00 Run Summary', [{'path':'Hcal/DigiRunHarvesting/SummaryvsLS/SummaryvsLS', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawRunHarvesting/SummaryvsLS/SummaryvsLS', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecoRunHarvesting/Summary/Summary', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br>WHITE color stands for INAPPLICABLE flag<br>Flag(Y) vs FED(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPRunHarvesting/Summary/Summary', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br>WHITE color stands for INAPPLICABLE flag<br>Flag(Y) vs FED(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '01 RAW Bad Quality', [{'path':'Hcal/RawTask/BadQualityvsBX/BadQualityvsBX', 'description':"""Distribution of Bad Channels vs Bunch Crossing <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawTask/BadQualityvsLS/BadQualityvsLS', 'description':"""Distribution of Bad Channels vs Lumi Section <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '02 RAW Bad Quality depth', [{'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth1', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth2', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth3', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawRunHarvesting/BadQuality/depth/depth4', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '03 RAW Bcn(Evn) Mismatches', [{'path':'Hcal/RawRunHarvesting/BcnMsm/Electronics/VME', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawRunHarvesting/BcnMsm/Electronics/uTCA', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RawRunHarvesting/EvnMsm/Electronics/VME', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawRunHarvesting/EvnMsm/Electronics/uTCA', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '04 DIGI Occupancy', [{'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth1', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth2', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth3', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiRunHarvesting/Occupancy/depth/depth4', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '05 DIGI Occupancy vs LS', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '06 DIGI Occupancy Cut', [{'path':'Hcal/DigiTask/OccupancyCut/depth/depth1', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCut/depth/depth2', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyCut/depth/depth3', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCut/depth/depth4', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '11 DIGI Amplitude vs LS', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '13 DIGI Timing', [{'path':'Hcal/DigiTask/TimingCut/depth/depth1', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/depth/depth2', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCut/depth/depth3', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/depth/depth4', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '14 DIGI Timing', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '17 DIGI Timing vs LS', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1100', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1102', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1104', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1106', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1108', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1110', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1112', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1114', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1116', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1118', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1120', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1122', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1132', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED724', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED725', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED726', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED727', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED728', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED729', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED730', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED731', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '18 RECO Energy', [{'path':'Hcal/RecHitTask/Energy/Subdet/HB', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/Energy/Subdet/HE', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/Energy/Subdet/HF', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/Energy/Subdet/HO', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '19 RECO Energy', [{'path':'Hcal/RecHitTask/Energy/depth/depth1', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/Energy/depth/depth2', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/Energy/depth/depth3', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/Energy/depth/depth4', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '23 RECO Occupancy', [{'path':'Hcal/RecHitTask/Occupancy/depth/depth1', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/Occupancy/depth/depth2', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/Occupancy/depth/depth3', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/Occupancy/depth/depth4', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '25 RECO Occupancy Cut', [{'path':'Hcal/RecHitTask/OccupancyCut/depth/depth1', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/OccupancyCut/depth/depth2', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/OccupancyCut/depth/depth3', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/OccupancyCut/depth/depth4', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '29 RECO Timing', [{'path':'Hcal/RecHitTask/TimingCut/depth/depth1', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/depth/depth2', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCut/depth/depth3', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/depth/depth4', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '30 RECO Timing', [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HBM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HBP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HEM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HEP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HFM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HFP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HOM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/SubdetPM/HOP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '31 RECO Timing vs LS', [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1100', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1102', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1104', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1106', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1108', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1110', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1112', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1114', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1116', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1118', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1120', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1122', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED1132', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED724', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED725', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED726', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED727', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED728', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED729', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED730', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCutvsLS/FED/FED731', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '34 RECO HBHEabc Timing', [{'path':'Hcal/RecHitTask/TimingCut/HBHEPartition/HBHEa', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingCut/HBHEPartition/HBHEb', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingCut/HBHEPartition/HBHEc', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '35 RECO Timing vs Energy', [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HBM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HBP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HEM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HEP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HFM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HFP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HOM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal/RecHitTask/TimingvsEnergy/SubdetPM/HOP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '36 TP Et Correlation', [{'path':'Hcal/TPTask/EtCorr/TTSubdet/HBHE', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCorr/TTSubdet/HF', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '37 TP Et Correlation Ratio', [{'path':'Hcal/TPTask/EtCorrRatio/EtCorrRatio', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '40 TP Et Distributions', [{'path':'Hcal/TPTask/EtData/EtData', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtEmul/EtEmul', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '41 TP Et Distributions', [{'path':'Hcal/TPTask/EtData/TTSubdet/HBHE', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtData/TTSubdet/HF', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/EtEmul/TTSubdet/HBHE', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtEmul/TTSubdet/HF', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '42 TP Et(FG) Mismatches', [{'path':'Hcal/TPTask/EtMsm/EtMsm', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/FGMsm/FGMsm', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '44 TP Occupancy', [{'path':'Hcal/TPTask/OccupancyData/OccupancyData', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyEmul/OccupancyEmul', 'description':"""Occupancy Distributions for Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '45 TP Occupancy Cut', [{'path':'Hcal/TPTask/OccupancyCutData/OccupancyCutData', 'description':"""Occupancy Distributions for Data with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyCutEmul/OccupancyCutEmul', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
