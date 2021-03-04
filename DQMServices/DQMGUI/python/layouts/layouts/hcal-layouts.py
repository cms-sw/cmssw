from .adapt_to_new_backend import *
dqmitems={}

def hcallayout(i, p, *rows): i['Hcal/Layouts/' + p] = rows


hcallayout(dqmitems, 'EtEmul/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtEmul/TTSubdet/HBHE', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtEmul/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtEmul/TTSubdet/HF', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmul/TP/Electronics/VME', [{'path':'Hcal/TPTask/OccupancyEmul/Electronics/VME', 'description':"""Occupancy Distributions for Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmul/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/OccupancyEmul/Electronics/uTCA', 'description':"""Occupancy Distributions for Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCutEmulvsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCutEmulvsBX/TTSubdet/HBHE', 'description':"""Et Emulator vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCutEmulvsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCutEmulvsBX/TTSubdet/HF', 'description':"""Et Emulator vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BcnMsm/RAW/Electronics/VME', [{'path':'Hcal/RawTask/BcnMsm/Electronics/VME', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BcnMsm/RAW/Electronics/uTCA', [{'path':'Hcal/RawTask/BcnMsm/Electronics/uTCA', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/DIGI/Subdet/HB', [{'path':'Hcal/DigiTask/Occupancyvsieta/Subdet/HB', 'description':"""Occupancy vs ieta (No cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/DIGI/Subdet/HE', [{'path':'Hcal/DigiTask/Occupancyvsieta/Subdet/HE', 'description':"""Occupancy vs ieta (No cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/DIGI/Subdet/HF', [{'path':'Hcal/DigiTask/Occupancyvsieta/Subdet/HF', 'description':"""Occupancy vs ieta (No cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/DIGI/Subdet/HO', [{'path':'Hcal/DigiTask/Occupancyvsieta/Subdet/HO', 'description':"""Occupancy vs ieta (No cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsmRatiovsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtMsmRatiovsBX/TTSubdet/HBHE', 'description':"""Rate of the Et Mismatches vs BX.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsmRatiovsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtMsmRatiovsBX/TTSubdet/HF', 'description':"""Rate of the Et Mismatches vs BX.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/QIE10', [{'path':'Hcal/QIE10Task/ADC/ADC', 'description':"""QIE10 ADC Distribution <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#QIE10_Task'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQualityvsBX/RAW', [{'path':'Hcal/RawTask/BadQualityvsBX/BadQualityvsBX', 'description':"""Distribution of Bad Channels vs Bunch Crossing <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/DIGI/Subdet/HB', [{'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HB', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/DIGI/Subdet/HE', [{'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HE', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/DIGI/Subdet/HF', [{'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HF', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/DIGI/Subdet/HO', [{'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HO', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1100', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1100', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1102', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1102', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1104', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1104', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1106', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1106', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1108', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1108', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1110', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1110', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1112', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1112', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1114', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1114', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1116', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1116', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1118', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1118', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1120', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1120', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1122', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1122', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED1132', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED1132', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED724', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED724', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED725', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED725', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED726', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED726', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED727', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED727', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED728', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED728', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED729', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED729', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED730', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED730', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW/FED/FED731', [{'path':'Hcal/RawTask/SummaryvsLS/FED/FED731', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtData/TTSubdet/HBHE', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtData/TTSubdet/HF', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1100', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1102', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1104', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1106', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1108', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1110', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1112', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1114', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1116', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1118', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1120', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1122', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED1132', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED724', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED725', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED726', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED727', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED728', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED729', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED730', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/SummaryvsLS/FED/FED731', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/DIGI/Subdet/HB', [{'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HB', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/DIGI/Subdet/HE', [{'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HE', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/DIGI/Subdet/HF', [{'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HF', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/DIGI/Subdet/HO', [{'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HO', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HB', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HE', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HF', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/DIGI/Subdet/HO', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP/Electronics/VME', [{'path':'Hcal/TPTask/FGMsm/Electronics/VME', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/FGMsm/Electronics/uTCA', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatio/TP', [{'path':'Hcal/TPTask/EtCorrRatio/EtCorrRatio', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCutDatavsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCutDatavsBX/TTSubdet/HBHE', 'description':"""Et Data vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCutDatavsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCutDatavsBX/TTSubdet/HF', 'description':"""Et Data vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HBM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HBP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HEM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HEP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HFM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HFP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HOM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HOP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP/Electronics/VME', [{'path':'Hcal/TPTask/EtMsm/Electronics/VME', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/EtMsm/Electronics/uTCA', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutData/TP', [{'path':'Hcal/TPTask/OccupancyCutData/OccupancyCutData', 'description':"""Occupancy Distributions for Data with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/DIGI/Subdet/HB', [{'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HB', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/DIGI/Subdet/HE', [{'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HE', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/DIGI/Subdet/HF', [{'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HF', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/DIGI/Subdet/HO', [{'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HO', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmulvsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyEmulvsBX/TTSubdet/HBHE', 'description':"""Emul Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmulvsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyEmulvsBX/TTSubdet/HF', 'description':"""Emul Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP', [{'path':'Hcal/TPTask/FGMsm/FGMsm', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCutDatavsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCutDatavsLS/TTSubdet/HBHE', 'description':"""Et Data vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCutDatavsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCutDatavsLS/TTSubdet/HF', 'description':"""Et Data vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HBM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HBP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HEM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HEP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HFM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HFP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HOM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsBX/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HOP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/Electronics/VME', [{'path':'Hcal/TPTask/EtData/Electronics/VME', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/EtData/Electronics/uTCA', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

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

hcallayout(dqmitems, 'OccupancyCutDatavsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyCutDatavsLS/TTSubdet/HBHE', 'description':"""Data Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutDatavsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyCutDatavsLS/TTSubdet/HF', 'description':"""Data Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP', [{'path':'Hcal/TPTask/SummaryvsLS/SummaryvsLS', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/DIGI', [{'path':'Hcal/DigiTask/SummaryvsLS/SummaryvsLS', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1100', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1100', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1102', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1102', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1104', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1104', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1106', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1106', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1108', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1108', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1110', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1110', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1112', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1112', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1114', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1114', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1116', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1116', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1118', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1118', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1120', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1120', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1122', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1122', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED1132', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED1132', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED724', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED724', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED725', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED725', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED726', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED726', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED727', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED727', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED728', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED728', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED729', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED729', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED730', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED730', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/TP/FED/FED731', [{'path':'Hcal/TPTask/SummaryvsLS/FED/FED731', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutEmulvsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/MsnCutEmulvsBX/TTSubdet/HBHE', 'description':"""Number of Channels missing from Emulator w.r.t. Data vs BX. (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutEmulvsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/MsnCutEmulvsBX/TTSubdet/HF', 'description':"""Number of Channels missing from Emulator w.r.t. Data vs BX. (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/Electronics/VME', [{'path':'Hcal/DigiTask/OccupancyCut/Electronics/VME', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/DIGI/Electronics/uTCA', [{'path':'Hcal/DigiTask/OccupancyCut/Electronics/uTCA', 'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HBM', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HBP', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HEM', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HEP', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HFM', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HFP', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HOM', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/OccupancyCutvsiphivsLS/SubdetPM/HOP', 'description':"""Occupancy Distribution iphi vs LS (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HBM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HBP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HEM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HEP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HFM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HFP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HOM', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/SumQ/SubdetPM/HOP', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutDatavsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/MsnCutDatavsBX/TTSubdet/HBHE', 'description':"""Number of Channels missing from Data w.r.t. Emulator vs BX. (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutDatavsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/MsnCutDatavsBX/TTSubdet/HF', 'description':"""Number of Channels missing from Data w.r.t. Emulator vs BX. (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EvnMsm/RAW/Electronics/VME', [{'path':'Hcal/RawTask/EvnMsm/Electronics/VME', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EvnMsm/RAW/Electronics/uTCA', [{'path':'Hcal/RawTask/EvnMsm/Electronics/uTCA', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQvsLS/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/DIGI/Subdet/HB', [{'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HB', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/DIGI/Subdet/HE', [{'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HE', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/DIGI/Subdet/HF', [{'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HF', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/DIGI/Subdet/HO', [{'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HO', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnEmul/TP', [{'path':'Hcal/TPTask/MsnEmul/MsnEmul', 'description':"""Distribution of channels missing from Emulator w.r.t. Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/Electronics/VME', [{'path':'Hcal/DigiTask/TimingCut/Electronics/VME', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/Electronics/uTCA', [{'path':'Hcal/DigiTask/TimingCut/Electronics/uTCA', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth1', [{'path':'Hcal/DigiTask/Occupancy/depth/depth1', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth2', [{'path':'Hcal/DigiTask/Occupancy/depth/depth2', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth3', [{'path':'Hcal/DigiTask/Occupancy/depth/depth3', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/DIGI/depth/depth4', [{'path':'Hcal/DigiTask/Occupancy/depth/depth4', 'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsmRatiovsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtMsmRatiovsLS/TTSubdet/HBHE', 'description':"""Rate of the Et Mismatches vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsmRatiovsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtMsmRatiovsLS/TTSubdet/HF', 'description':"""Rate of the Et Mismatches vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyData2x3/TP', [{'path':'Hcal/TPTask/OccupancyData2x3/OccupancyData2x3', 'description':"""Data Occupancy for 2x3 TPs for HF <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtEmul/TP', [{'path':'Hcal/TPTask/EtEmul/EtEmul', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HBM', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HBP', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HEM', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HEP', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HFM', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HFP', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HOM', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/Occupancyvsiphi/SubdetPM/HOP', 'description':"""Occupancy vs iphi (no cuts) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtData/TP', [{'path':'Hcal/TPTask/EtData/EtData', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorr/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCorr/TTSubdet/HBHE', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorr/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCorr/TTSubdet/HF', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmul/TP', [{'path':'Hcal/TPTask/OccupancyCutEmul/OccupancyCutEmul', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

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

hcallayout(dqmitems, 'EtCutEmulvsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCutEmulvsLS/TTSubdet/HBHE', 'description':"""Et Emulator vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCutEmulvsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCutEmulvsLS/TTSubdet/HF', 'description':"""Et Emulator vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmul/TP/Electronics/VME', [{'path':'Hcal/TPTask/OccupancyCutEmul/Electronics/VME', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmul/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/OccupancyCutEmul/Electronics/uTCA', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGCorr/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/FGCorr/TTSubdet/HBHE', 'description':"""Correction of Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGCorr/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/FGCorr/TTSubdet/HF', 'description':"""Correction of Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutDatavsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyCutDatavsBX/TTSubdet/HBHE', 'description':"""Data Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutDatavsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyCutDatavsBX/TTSubdet/HF', 'description':"""Data Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HBM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HBP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HEM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HEP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HFM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HFP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HOM', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/ADC/SubdetPM/HOP', 'description':"""ADC Distributions per 1 TS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmulvsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyCutEmulvsLS/TTSubdet/HBHE', 'description':"""Emul Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmulvsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyCutEmulvsLS/TTSubdet/HF', 'description':"""Emul Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyDatavsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyDatavsBX/TTSubdet/HBHE', 'description':"""Data Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyDatavsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyDatavsBX/TTSubdet/HF', 'description':"""Data Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'LETDCvsADC/QIE10', [{'path':'Hcal/QIE10Task/LETDCvsADC/LETDCvsADC', 'description':"""QIE10 Leading Edge TDC vs ADC Distribution <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#QIE10_Task'>Details...</a>"""}])

hcallayout(dqmitems, 'Q2Q12vsLS/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/Q2Q12vsLS/FED/FED1118', 'description':"""Charge in TS2 over the sum of charges in TS1 and TS2 vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Q2Q12vsLS/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/Q2Q12vsLS/FED/FED1120', 'description':"""Charge in TS2 over the sum of charges in TS1 and TS2 vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Q2Q12vsLS/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/Q2Q12vsLS/FED/FED1122', 'description':"""Charge in TS2 over the sum of charges in TS1 and TS2 vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmul2x3/TP', [{'path':'Hcal/TPTask/OccupancyEmul2x3/OccupancyEmul2x3', 'description':"""Emulator Occupancy for 2x3 TPs for HF <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth1', [{'path':'Hcal/DigiTask/TimingCut/depth/depth1', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth2', [{'path':'Hcal/DigiTask/TimingCut/depth/depth2', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth3', [{'path':'Hcal/DigiTask/TimingCut/depth/depth3', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/DIGI/depth/depth4', [{'path':'Hcal/DigiTask/TimingCut/depth/depth4', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1100', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1100', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1102', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1102', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1104', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1104', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1106', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1106', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1108', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1108', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1110', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1110', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1112', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1112', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1114', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1114', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1116', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1116', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1118', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1118', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1120', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1120', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1122', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1122', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED1132', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED1132', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED724', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED724', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED725', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED725', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED726', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED726', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED727', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED727', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED728', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED728', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED729', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED729', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED730', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED730', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'DigiSizevsLS/DIGI/FED/FED731', [{'path':'Hcal/DigiTask/DigiSizevsLS/FED/FED731', 'description':"""Digi Size vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'LETDC/QIE10', [{'path':'Hcal/QIE10Task/LETDC/LETDC', 'description':"""QIE10 Leading Edge TDC Distribution <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#QIE10_Task'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnData/TP/Electronics/VME', [{'path':'Hcal/TPTask/MsnData/Electronics/VME', 'description':"""Distribution of channels missing from Data w.r.t. Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnData/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/MsnData/Electronics/uTCA', 'description':"""Distribution of channels missing from Data w.r.t. Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

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

hcallayout(dqmitems, 'TETDCvsADC/QIE10', [{'path':'Hcal/QIE10Task/TETDCvsADC/TETDCvsADC', 'description':"""QIE10 Trailing Edge TDC vs ADC Distribution <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#QIE10_Task'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth1', [{'path':'Hcal/RawTask/BadQuality/depth/depth1', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth2', [{'path':'Hcal/RawTask/BadQuality/depth/depth2', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth3', [{'path':'Hcal/RawTask/BadQuality/depth/depth3', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'BadQuality/RAW/depth/depth4', [{'path':'Hcal/RawTask/BadQuality/depth/depth4', 'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatio/TP/Electronics/VME', [{'path':'Hcal/TPTask/EtCorrRatio/Electronics/VME', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatio/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/EtCorrRatio/Electronics/uTCA', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmulvsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyEmulvsLS/TTSubdet/HBHE', 'description':"""Emul Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyEmulvsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyEmulvsLS/TTSubdet/HF', 'description':"""Emul Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth1', [{'path':'Hcal/DigiTask/SumQ/depth/depth1', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth2', [{'path':'Hcal/DigiTask/SumQ/depth/depth2', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth3', [{'path':'Hcal/DigiTask/SumQ/depth/depth3', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SumQ/DIGI/depth/depth4', [{'path':'Hcal/DigiTask/SumQ/depth/depth4', 'description':"""Signal Amplitude  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyData/TP', [{'path':'Hcal/TPTask/OccupancyData/OccupancyData', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyData/TP/Electronics/VME', [{'path':'Hcal/TPTask/OccupancyData/Electronics/VME', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyData/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/OccupancyData/Electronics/uTCA', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

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

hcallayout(dqmitems, 'EtCorrRatiovsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCorrRatiovsLS/TTSubdet/HBHE', 'description':"""Et Correlation Ratio vs LS Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatiovsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCorrRatiovsLS/TTSubdet/HF', 'description':"""Et Correlation Ratio vs LS Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyDatavsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyDatavsLS/TTSubdet/HBHE', 'description':"""Data Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyDatavsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyDatavsLS/TTSubdet/HF', 'description':"""Data Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutDatavsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/MsnCutDatavsLS/TTSubdet/HBHE', 'description':"""Number of Channels missing from Data w.r.t. Emulator vs LS. (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutDatavsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/MsnCutDatavsLS/TTSubdet/HF', 'description':"""Number of Channels missing from Data w.r.t. Emulator vs LS. (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutEmulvsLS/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/MsnCutEmulvsLS/TTSubdet/HBHE', 'description':"""Number of Channels missing from Emulator w.r.t. Data vs LS. (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnCutEmulvsLS/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/MsnCutEmulvsLS/TTSubdet/HF', 'description':"""Number of Channels missing from Emulator w.r.t. Data vs LS. (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

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

hcallayout(dqmitems, 'EtMsm/TP', [{'path':'Hcal/TPTask/EtMsm/EtMsm', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnEmul/TP/Electronics/VME', [{'path':'Hcal/TPTask/MsnEmul/Electronics/VME', 'description':"""Distribution of channels missing from Emulator w.r.t. Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnEmul/TP/Electronics/uTCA', [{'path':'Hcal/TPTask/MsnEmul/Electronics/uTCA', 'description':"""Distribution of channels missing from Emulator w.r.t. Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'KnownBadChannels/RunInfo/depth/depth1', [{'path':'Hcal/RunInfo/KnownBadChannels/depth/depth1', 'description':"""Channels that come from Hcal Channel Quality Object from Conditions. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Run_Info'>Details...</a>"""}])

hcallayout(dqmitems, 'KnownBadChannels/RunInfo/depth/depth2', [{'path':'Hcal/RunInfo/KnownBadChannels/depth/depth2', 'description':"""Channels that come from Hcal Channel Quality Object from Conditions. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Run_Info'>Details...</a>"""}])

hcallayout(dqmitems, 'KnownBadChannels/RunInfo/depth/depth3', [{'path':'Hcal/RunInfo/KnownBadChannels/depth/depth3', 'description':"""Channels that come from Hcal Channel Quality Object from Conditions. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Run_Info'>Details...</a>"""}])

hcallayout(dqmitems, 'KnownBadChannels/RunInfo/depth/depth4', [{'path':'Hcal/RunInfo/KnownBadChannels/depth/depth4', 'description':"""Channels that come from Hcal Channel Quality Object from Conditions. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Run_Info'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RAW', [{'path':'Hcal/RawTask/SummaryvsLS/SummaryvsLS', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

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

hcallayout(dqmitems, 'EtCorrRatiovsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCorrRatiovsBX/TTSubdet/HBHE', 'description':"""Et Correlation Ratio vs BX Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorrRatiovsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCorrRatiovsBX/TTSubdet/HF', 'description':"""Et Correlation Ratio vs BX Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'MsnData/TP', [{'path':'Hcal/TPTask/MsnData/MsnData', 'description':"""Distribution of channels missing from Data w.r.t. Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HBM', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HBM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HBP', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HBP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HEM', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HEM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HEP', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HEP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HFM', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HFM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HFP', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HFP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HOM', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HOM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/DIGI/SubdetPM/HOP', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HOP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorr2x3/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/EtCorr2x3/TTSubdet/HBHE', 'description':"""HF 2x3 Correlation <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtCorr2x3/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/EtCorr2x3/TTSubdet/HF', 'description':"""HF 2x3 Correlation <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmulvsBX/TP/TTSubdet/HBHE', [{'path':'Hcal/TPTask/OccupancyCutEmulvsBX/TTSubdet/HBHE', 'description':"""Emul Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutEmulvsBX/TP/TTSubdet/HF', [{'path':'Hcal/TPTask/OccupancyCutEmulvsBX/TTSubdet/HF', 'description':"""Emul Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS2/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/FG_TS2/TTSubdet/HBHE', 'description':"""FG Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS2/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/FG_TS2/TTSubdet/HF', 'description':"""FG Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1100', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1100', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1102', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1102', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1104', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1104', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1106', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1106', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1108', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1108', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1110', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1110', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1112', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1112', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1114', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1114', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1116', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1116', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1118', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1118', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1120', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1120', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1122', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1122', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED1132', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED1132', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED724', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED724', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED725', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED725', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED726', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED726', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED727', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED727', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED728', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED728', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED729', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED729', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED730', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED730', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO/FED/FED731', [{'path':'Hcal2/RecHitTask/SummaryvsLS/FED/FED731', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/RECO/Subdet/HB', [{'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HB', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/RECO/Subdet/HE', [{'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HE', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/RECO/Subdet/HF', [{'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HF', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsieta/RECO/Subdet/HO', [{'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HO', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_VME/DIGI_VMEvsuTCA/depth/depth1', [{'path':'Hcal2/DigiComparisonTask/Missing_VME/depth/depth1', 'description':"""Digis missing from VME collection and present in uTCA <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_VME/DIGI_VMEvsuTCA/depth/depth2', [{'path':'Hcal2/DigiComparisonTask/Missing_VME/depth/depth2', 'description':"""Digis missing from VME collection and present in uTCA <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_VME/DIGI_VMEvsuTCA/depth/depth3', [{'path':'Hcal2/DigiComparisonTask/Missing_VME/depth/depth3', 'description':"""Digis missing from VME collection and present in uTCA <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_VME/DIGI_VMEvsuTCA/depth/depth4', [{'path':'Hcal2/DigiComparisonTask/Missing_VME/depth/depth4', 'description':"""Digis missing from VME collection and present in uTCA <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS3/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/Et_TS3/TTSubdet/HBHE', 'description':"""Et Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS3/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/Et_TS3/TTSubdet/HF', 'description':"""Et Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/Electronics/VME', [{'path':'Hcal2/RecHitTask/OccupancyCut/Electronics/VME', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/Electronics/uTCA', [{'path':'Hcal2/RecHitTask/OccupancyCut/Electronics/uTCA', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HBM', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HBP', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HEM', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HEP', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HFM', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HFP', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HOM', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsiphi/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/Occupancyvsiphi/SubdetPM/HOP', 'description':"""Occupancy vs iphi (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/RECO/Subdet/HB', [{'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HB', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/RECO/Subdet/HE', [{'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HE', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/RECO/Subdet/HF', [{'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HF', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsLS/RECO/Subdet/HO', [{'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HO', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS2/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS2/Subdet/HB', 'description':"""ADC Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS2/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS2/Subdet/HE', 'description':"""ADC Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS2/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS2/Subdet/HF', 'description':"""ADC Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS2/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS2/Subdet/HO', 'description':"""ADC Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS0/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS0/Subdet/HB', 'description':"""ADC Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS0/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS0/Subdet/HE', 'description':"""ADC Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS0/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS0/Subdet/HF', 'description':"""ADC Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS0/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS0/Subdet/HO', 'description':"""ADC Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/RECO/Subdet/HB', [{'path':'Hcal2/RecHitTask/Occupancyvsieta/Subdet/HB', 'description':"""Occupancy vs ieta (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/RECO/Subdet/HE', [{'path':'Hcal2/RecHitTask/Occupancyvsieta/Subdet/HE', 'description':"""Occupancy vs ieta (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/RECO/Subdet/HF', [{'path':'Hcal2/RecHitTask/Occupancyvsieta/Subdet/HF', 'description':"""Occupancy vs ieta (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancyvsieta/RECO/Subdet/HO', [{'path':'Hcal2/RecHitTask/Occupancyvsieta/Subdet/HO', 'description':"""Occupancy vs ieta (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS9/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS9/Subdet/HB', 'description':"""ADC Correlation for TS9. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS9/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS9/Subdet/HE', 'description':"""ADC Correlation for TS9. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS9/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS9/Subdet/HF', 'description':"""ADC Correlation for TS9. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS9/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS9/Subdet/HO', 'description':"""ADC Correlation for TS9. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/Electronics/VME', [{'path':'Hcal2/RecHitTask/TimingCut/Electronics/VME', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/Electronics/uTCA', [{'path':'Hcal2/RecHitTask/TimingCut/Electronics/uTCA', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/Electronics/VME', [{'path':'Hcal2/RecHitTask/Occupancy/Electronics/VME', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/Electronics/uTCA', [{'path':'Hcal2/RecHitTask/Occupancy/Electronics/uTCA', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS3/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/FG_TS3/TTSubdet/HBHE', 'description':"""FG Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS3/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/FG_TS3/TTSubdet/HF', 'description':"""FG Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HBM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HBP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HEM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HEP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HFM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HFP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HOM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HOP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth1', [{'path':'Hcal2/RecHitTask/Energy/depth/depth1', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth2', [{'path':'Hcal2/RecHitTask/Energy/depth/depth2', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth3', [{'path':'Hcal2/RecHitTask/Energy/depth/depth3', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/depth/depth4', [{'path':'Hcal2/RecHitTask/Energy/depth/depth4', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1100', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1100', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1102', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1102', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1104', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1104', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1106', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1106', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1108', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1108', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1110', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1110', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1112', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1112', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1114', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1114', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1116', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1116', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1118', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1118', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1120', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1120', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1122', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1122', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED1132', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1132', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED724', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED724', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED725', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED725', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED726', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED726', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED727', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED727', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED728', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED728', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED729', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED729', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED730', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED730', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsLS/RECO/FED/FED731', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED731', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HB', [{'path':'Hcal2/RecHitTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HE', [{'path':'Hcal2/RecHitTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HF', [{'path':'Hcal2/RecHitTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyvsLS/RECO/Subdet/HO', [{'path':'Hcal2/RecHitTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS (no cuts applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HB', [{'path':'Hcal2/RecHitTask/Energy/Subdet/HB', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HE', [{'path':'Hcal2/RecHitTask/Energy/Subdet/HE', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HF', [{'path':'Hcal2/RecHitTask/Energy/Subdet/HF', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energy/RECO/Subdet/HO', [{'path':'Hcal2/RecHitTask/Energy/Subdet/HO', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HBM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HBP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HEM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HEP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HFM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HFP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HOM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingvsEnergy/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HOP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HBM', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HBP', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HEM', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HEP', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HFM', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HFP', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HOM', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsBX/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/TimingCutvsBX/SubdetPM/HOP', 'description':"""Timing $RECO vs BX (Cut is  applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1100', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1100', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1102', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1102', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1104', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1104', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1106', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1106', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1108', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1108', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1110', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1110', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1112', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1112', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1114', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1114', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1116', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1116', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1118', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1118', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1120', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1120', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1122', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1122', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED1132', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED1132', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED700', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED700', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED701', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED701', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED702', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED702', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED703', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED703', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED704', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED704', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED705', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED705', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED706', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED706', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED707', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED707', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED708', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED708', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED709', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED709', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED710', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED710', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED711', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED711', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED712', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED712', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED713', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED713', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED714', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED714', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED715', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED715', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED716', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED716', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED717', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED717', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED724', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED724', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED725', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED725', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED726', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED726', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED727', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED727', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED728', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED728', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED729', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED729', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED730', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED730', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA/FED/FED731', [{'path':'Hcal2/TPComparisonTask/FGMsm/FED/FED731', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HBM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HBP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HEM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HEP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HFM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HFP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HOM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphi/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HOP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth1', [{'path':'Hcal2/RecHitTask/TimingCut/depth/depth1', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth2', [{'path':'Hcal2/RecHitTask/TimingCut/depth/depth2', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth3', [{'path':'Hcal2/RecHitTask/TimingCut/depth/depth3', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/depth/depth4', [{'path':'Hcal2/RecHitTask/TimingCut/depth/depth4', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/depth/depth1', [{'path':'Hcal2/DigiComparisonTask/Mismatched/depth/depth1', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/depth/depth2', [{'path':'Hcal2/DigiComparisonTask/Mismatched/depth/depth2', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/depth/depth3', [{'path':'Hcal2/DigiComparisonTask/Mismatched/depth/depth3', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/depth/depth4', [{'path':'Hcal2/DigiComparisonTask/Mismatched/depth/depth4', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnVME/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADCMsnVME/Subdet/HB', 'description':"""ADC Distribution for Missing VME Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnVME/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADCMsnVME/Subdet/HE', 'description':"""ADC Distribution for Missing VME Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnVME/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADCMsnVME/Subdet/HF', 'description':"""ADC Distribution for Missing VME Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnVME/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADCMsnVME/Subdet/HO', 'description':"""ADC Distribution for Missing VME Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_uTCA/DIGI_VMEvsuTCA/depth/depth1', [{'path':'Hcal2/DigiComparisonTask/Missing_uTCA/depth/depth1', 'description':"""Digis missing from uTCA collection and present in VME <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_uTCA/DIGI_VMEvsuTCA/depth/depth2', [{'path':'Hcal2/DigiComparisonTask/Missing_uTCA/depth/depth2', 'description':"""Digis missing from uTCA collection and present in VME <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_uTCA/DIGI_VMEvsuTCA/depth/depth3', [{'path':'Hcal2/DigiComparisonTask/Missing_uTCA/depth/depth3', 'description':"""Digis missing from uTCA collection and present in VME <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing_uTCA/DIGI_VMEvsuTCA/depth/depth4', [{'path':'Hcal2/DigiComparisonTask/Missing_uTCA/depth/depth4', 'description':"""Digis missing from uTCA collection and present in VME <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS0/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/FG_TS0/TTSubdet/HBHE', 'description':"""FG Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS0/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/FG_TS0/TTSubdet/HF', 'description':"""FG Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS8/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS8/Subdet/HB', 'description':"""ADC Correlation for TS8. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS8/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS8/Subdet/HE', 'description':"""ADC Correlation for TS8. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS8/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS8/Subdet/HF', 'description':"""ADC Correlation for TS8. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS8/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS8/Subdet/HO', 'description':"""ADC Correlation for TS8. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnuTCA/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADCMsnuTCA/Subdet/HB', 'description':"""ADC Distribution for Missing uTCA Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnuTCA/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADCMsnuTCA/Subdet/HE', 'description':"""ADC Distribution for Missing uTCA Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnuTCA/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADCMsnuTCA/Subdet/HF', 'description':"""ADC Distribution for Missing uTCA Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADCMsnuTCA/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADCMsnuTCA/Subdet/HO', 'description':"""ADC Distribution for Missing uTCA Digis <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS0/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/Et_TS0/TTSubdet/HBHE', 'description':"""Et Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS0/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/Et_TS0/TTSubdet/HF', 'description':"""Et Correlation for TS0. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth1', [{'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth1', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth2', [{'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth2', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth3', [{'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth3', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/depth/depth4', [{'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth4', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1100', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1100', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1102', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1102', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1104', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1104', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1106', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1106', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1108', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1108', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1110', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1110', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1112', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1112', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1114', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1114', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1116', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1116', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1118', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1118', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1120', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1120', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1122', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1122', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED1132', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED1132', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED724', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED724', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED725', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED725', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED726', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED726', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED727', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED727', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED728', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED728', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED729', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED729', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED730', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED730', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/FED/FED731', [{'path':'Hcal2/RecHitTask/TimingCut/FED/FED731', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Et/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/Et/TTSubdet/HBHE', 'description':"""Et Correlation over all TS. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Et/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/Et/TTSubdet/HF', 'description':"""Et Correlation over all TS. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1100', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1100', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1102', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1102', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1104', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1104', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1106', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1106', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1108', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1108', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1110', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1110', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1112', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1112', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1114', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1114', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1116', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1116', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1118', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1118', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1120', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1120', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1122', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1122', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED1132', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED1132', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED724', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED724', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED725', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED725', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED726', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED726', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED727', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED727', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED728', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED728', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED729', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED729', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED730', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED730', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCut/RECO/FED/FED731', [{'path':'Hcal2/RecHitTask/OccupancyCut/FED/FED731', 'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC/Subdet/HB', 'description':"""ADC Correlation for all Time Slices VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC/Subdet/HE', 'description':"""ADC Correlation for all Time Slices VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC/Subdet/HF', 'description':"""ADC Correlation for all Time Slices VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC/Subdet/HO', 'description':"""ADC Correlation for all Time Slices VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HBM', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HBP', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HEM', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HEP', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HFM', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HFP', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HOM', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsiphivsLS/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphivsLS/SubdetPM/HOP', 'description':"""Occupancy Distribution iphi vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA', [{'path':'Hcal2/TPComparisonTask/Missing/Missing_VME', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA', [{'path':'Hcal2/TPComparisonTask/Missing/Missing_uTCA', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS1/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS1/Subdet/HB', 'description':"""ADC Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS1/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS1/Subdet/HE', 'description':"""ADC Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS1/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS1/Subdet/HF', 'description':"""ADC Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS1/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS1/Subdet/HO', 'description':"""ADC Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/HBHEPartition/HBHEa', [{'path':'Hcal2/RecHitTask/TimingCut/HBHEPartition/HBHEa', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/HBHEPartition/HBHEb', [{'path':'Hcal2/RecHitTask/TimingCut/HBHEPartition/HBHEb', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCut/RECO/HBHEPartition/HBHEc', [{'path':'Hcal2/RecHitTask/TimingCut/HBHEPartition/HBHEc', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS1/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/FG_TS1/TTSubdet/HBHE', 'description':"""FG Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'FG_TS1/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/FG_TS1/TTSubdet/HF', 'description':"""FG Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HBM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HBP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HEM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HEP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HFM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HFP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HOM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsLS/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HOP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1100', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1100', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1102', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1102', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1104', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1104', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1106', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1106', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1108', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1108', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1110', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1110', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1112', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1112', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1114', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1114', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1116', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1116', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1118', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1118', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1120', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1120', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1122', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1122', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED1132', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED1132', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED700', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED700', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED701', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED701', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED702', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED702', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED703', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED703', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED704', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED704', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED705', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED705', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED706', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED706', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED707', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED707', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED708', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED708', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED709', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED709', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED710', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED710', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED711', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED711', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED712', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED712', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED713', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED713', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED714', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED714', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED715', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED715', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED716', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED716', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED717', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED717', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED724', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED724', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED725', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED725', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED726', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED726', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED727', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED727', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED728', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED728', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED729', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED729', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED730', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED730', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Mismatched/DIGI_VMEvsuTCA/FED/FED731', [{'path':'Hcal2/DigiComparisonTask/Mismatched/FED/FED731', 'description':"""Digis for which ADCs are mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth1', [{'path':'Hcal2/RecHitTask/Occupancy/depth/depth1', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth2', [{'path':'Hcal2/RecHitTask/Occupancy/depth/depth2', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth3', [{'path':'Hcal2/RecHitTask/Occupancy/depth/depth3', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/depth/depth4', [{'path':'Hcal2/RecHitTask/Occupancy/depth/depth4', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS1/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/Et_TS1/TTSubdet/HBHE', 'description':"""Et Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS1/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/Et_TS1/TTSubdet/HF', 'description':"""Et Correlation for TS1. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS6/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS6/Subdet/HB', 'description':"""ADC Correlation for TS6. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS6/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS6/Subdet/HE', 'description':"""ADC Correlation for TS6. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS6/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS6/Subdet/HF', 'description':"""ADC Correlation for TS6. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS6/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS6/Subdet/HO', 'description':"""ADC Correlation for TS6. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1100', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1100', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1102', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1102', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1104', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1104', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1106', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1106', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1108', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1108', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1110', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1110', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1112', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1112', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1114', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1114', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1116', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1116', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1118', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1118', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1120', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1120', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1122', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1122', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED1132', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED1132', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED724', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED724', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED725', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED725', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED726', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED726', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED727', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED727', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED728', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED728', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED729', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED729', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED730', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED730', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Occupancy/RECO/FED/FED731', [{'path':'Hcal2/RecHitTask/Occupancy/FED/FED731', 'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1100', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1100', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1102', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1102', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1104', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1104', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1106', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1106', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1108', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1108', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1110', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1110', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1112', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1112', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1114', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1114', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1116', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1116', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1118', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1118', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1120', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1120', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1122', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1122', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED1132', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED1132', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED700', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED700', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED701', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED701', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED702', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED702', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED703', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED703', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED704', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED704', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED705', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED705', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED706', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED706', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED707', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED707', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED708', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED708', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED709', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED709', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED710', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED710', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED711', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED711', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED712', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED712', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED713', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED713', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED714', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED714', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED715', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED715', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED716', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED716', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED717', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED717', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED724', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED724', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED725', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED725', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED726', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED726', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED727', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED727', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED728', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED728', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED729', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED729', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED730', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED730', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/DIGI_VMEvsuTCA/FED/FED731', [{'path':'Hcal2/DigiComparisonTask/Missing/FED/FED731', 'description':"""Channels missing from one ELECTRONICS UNIT and present in the other! Note, every plot designates 1 FED, therfore if uHBHE FED shows up that means that this channel is present in VME but absent in uHBHE <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsieta/RECO/Subdet/HB', [{'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HB', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsieta/RECO/Subdet/HE', [{'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HE', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsieta/RECO/Subdet/HF', [{'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HF', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsieta/RECO/Subdet/HO', [{'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HO', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HBM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HBP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HEM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HEP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HFM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HFP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HOM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'TimingCutvsiphi/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HOP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1100', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1100', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1102', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1102', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1104', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1104', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1106', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1106', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1108', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1108', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1110', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1110', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1112', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1112', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1114', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1114', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1116', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1116', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1118', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1118', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1120', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1120', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1122', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1122', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED1132', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED1132', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED700', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED700', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED701', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED701', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED702', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED702', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED703', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED703', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED704', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED704', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED705', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED705', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED706', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED706', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED707', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED707', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED708', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED708', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED709', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED709', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED710', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED710', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED711', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED711', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED712', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED712', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED713', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED713', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED714', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED714', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED715', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED715', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED716', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED716', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED717', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED717', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED724', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED724', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED725', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED725', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED726', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED726', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED727', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED727', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED728', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED728', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED729', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED729', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED730', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED730', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA/FED/FED731', [{'path':'Hcal2/TPComparisonTask/EtMsm/FED/FED731', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS7/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS7/Subdet/HB', 'description':"""ADC Correlation for TS7. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS7/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS7/Subdet/HE', 'description':"""ADC Correlation for TS7. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS7/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS7/Subdet/HF', 'description':"""ADC Correlation for TS7. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS7/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS7/Subdet/HO', 'description':"""ADC Correlation for TS7. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'EtMsm/TP_VMEvsuTCA', [{'path':'Hcal2/TPComparisonTask/EtMsm/EtMsm', 'description':"""Distrubution of channels that have Et mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HBM', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HBP', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HEM', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HEP', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HFM', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HFP', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HOM', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsBX/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/OccupancyCutvsBX/SubdetPM/HOP', 'description':"""Occupancy vs BX (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'SummaryvsLS/RECO', [{'path':'Hcal2/RecHitTask/SummaryvsLS/SummaryvsLS', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS3/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS3/Subdet/HB', 'description':"""ADC Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS3/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS3/Subdet/HE', 'description':"""ADC Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS3/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS3/Subdet/HF', 'description':"""ADC Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS3/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS3/Subdet/HO', 'description':"""ADC Correlation for TS3. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1100', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1100', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1102', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1102', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1104', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1104', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1106', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1106', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1108', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1108', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1110', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1110', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1112', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1112', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1114', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1114', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1116', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1116', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1118', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1118', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1120', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1120', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1122', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1122', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED1132', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED1132', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED700', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED700', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED701', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED701', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED702', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED702', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED703', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED703', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED704', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED704', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED705', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED705', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED706', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED706', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED707', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED707', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED708', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED708', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED709', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED709', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED710', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED710', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED711', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED711', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED712', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED712', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED713', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED713', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED714', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED714', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED715', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED715', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED716', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED716', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED717', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED717', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED724', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED724', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED725', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED725', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED726', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED726', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED727', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED727', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED728', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED728', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED729', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED729', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED730', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED730', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Missing/TP_VMEvsuTCA/FED/FED731', [{'path':'Hcal2/TPComparisonTask/Missing/FED/FED731', 'description':"""Distribution of channels missing from 1 ELECTRONICS UNIT and present in the other <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/RECO/Subdet/HB', [{'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HB', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/RECO/Subdet/HE', [{'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HE', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/RECO/Subdet/HF', [{'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HF', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'OccupancyCutvsieta/RECO/Subdet/HO', [{'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HO', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'FGMsm/TP_VMEvsuTCA', [{'path':'Hcal2/TPComparisonTask/FGMsm/FGMsm', 'description':"""Distribution of channels that have FG bit mismatched <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS2/TP_VMEvsuTCA/TTSubdet/HBHE', [{'path':'Hcal2/TPComparisonTask/Et_TS2/TTSubdet/HBHE', 'description':"""Et Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'Et_TS2/TP_VMEvsuTCA/TTSubdet/HF', [{'path':'Hcal2/TPComparisonTask/Et_TS2/TTSubdet/HF', 'description':"""Et Correlation for TS2. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Trigger_Primitives_Comparison_Ta'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS4/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS4/Subdet/HB', 'description':"""ADC Correlation for TS4. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS4/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS4/Subdet/HE', 'description':"""ADC Correlation for TS4. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS4/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS4/Subdet/HF', 'description':"""ADC Correlation for TS4. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS4/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS4/Subdet/HO', 'description':"""ADC Correlation for TS4. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS5/DIGI_VMEvsuTCA/Subdet/HB', [{'path':'Hcal2/DigiComparisonTask/ADC_TS5/Subdet/HB', 'description':"""ADC Correlation for TS5. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS5/DIGI_VMEvsuTCA/Subdet/HE', [{'path':'Hcal2/DigiComparisonTask/ADC_TS5/Subdet/HE', 'description':"""ADC Correlation for TS5. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS5/DIGI_VMEvsuTCA/Subdet/HF', [{'path':'Hcal2/DigiComparisonTask/ADC_TS5/Subdet/HF', 'description':"""ADC Correlation for TS5. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'ADC_TS5/DIGI_VMEvsuTCA/Subdet/HO', [{'path':'Hcal2/DigiComparisonTask/ADC_TS5/Subdet/HO', 'description':"""ADC Correlation for TS5. VME(Y) vs uTCA(X) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Comparison_Task_VME_vs_uTCA'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HBM', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HBP', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HEM', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HEP', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HFM', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HFP', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HOM', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'EnergyvsBX/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/EnergyvsBX/SubdetPM/HOP', 'description':"""Energy vs BX (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HBM', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HBM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HBP', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HBP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HEM', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HEM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HEP', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HEP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HFM', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HFM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HFP', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HFP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HOM', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HOM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, 'Energyvsiphi/RECO/SubdetPM/HOP', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HOP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '00 Run Summary', [{'path':'Hcal/TPTask/SummaryvsLS/SummaryvsLS', 'description':"""Trigger Primitives Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SummaryvsLS/SummaryvsLS', 'description':"""DIGI Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RawTask/SummaryvsLS/SummaryvsLS', 'description':"""RAW Summary Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/SummaryvsLS/SummaryvsLS', 'description':"""RECO Summary. Summary. Anything that is not either WHITE or GREEN or Gray is BAD.<br> Color Scheme:<br><font color='green'>GOOD</font> for GOOD<br><font color='yellow'>PROBLEMATIC</font> for Problematic<br><font color='red'>BAD</font> for BAD<br><font color='black'>RESERVED</font> Not used at the moment <br><font color='gray'>NCDAQ with Gray Font</font> FED is excluded from cDAQ<br>WHITE color stands for INAPPLICABLE flag<br>FED(Y) vs LS(X). All the Monitoring Tasks are summarized. For details...  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '01 RAW Bad Quality', [{'path':'Hcal/RawTask/BadQualityvsBX/BadQualityvsBX', 'description':"""Distribution of Bad Channels vs Bunch Crossing <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawTask/BadQualityvsLS/BadQualityvsLS', 'description':"""Distribution of Bad Channels vs Lumi Section <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '02 RAW Bad Quality depth', 
	[
		{
			'path':'Hcal/RawTask/BadQuality/depth/depth1', 
			'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/RawTask/BadQuality/depth/depth2', 
			'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/RawTask/BadQuality/depth/depth3', 
			'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/RawTask/BadQuality/depth/depth4', 
			'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/RawTask/BadQuality/depth/depth5', 
			'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/RawTask/BadQuality/depth/depth6', 
			'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/RawTask/BadQuality/depth/depth7', 
			'description':"""Channels that were marked as Bad Quality by Unpacker. It includes, but not limited to, CapId nonrotation, validity bits checks, etc...Statistics over the whole Run is combined. Either all LSs up to the current one or up to end of the Run <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '03 RAW Bcn(Evn) Mismatches', [{'path':'Hcal/RawTask/BcnMsm/Electronics/VME', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawTask/BcnMsm/Electronics/uTCA', 'description':"""BX Mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}], [{'path':'Hcal/RawTask/EvnMsm/Electronics/VME', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}, {'path':'Hcal/RawTask/EvnMsm/Electronics/uTCA', 'description':"""Event Number mismatches between individual uHTR and AMC13 <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Raw_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '04 DIGI Occupancy', 
	[
		{
			'path':'Hcal/DigiTask/Occupancy/depth/depth1', 
			'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/Occupancy/depth/depth2', 
			'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/Occupancy/depth/depth3', 
			'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/DigiTask/Occupancy/depth/depth4', 
			'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/Occupancy/depth/depth5', 
			'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/Occupancy/depth/depth6', 
			'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/DigiTask/Occupancy/depth/depth7', 
			'description':"""Occupancy.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '05 DIGI Occupancy vs LS', [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HB', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HE', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HF', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyvsLS/Subdet/HO', 'description':"""Occupancy vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '06 DIGI Occupancy Cut', 
	[
		{
			'path':'Hcal/DigiTask/OccupancyCut/depth/depth1', 
			'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/OccupancyCut/depth/depth2', 
			'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/OccupancyCut/depth/depth3', 
			'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/DigiTask/OccupancyCut/depth/depth4', 
			'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/OccupancyCut/depth/depth5', 
			'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/OccupancyCut/depth/depth6', 
			'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/DigiTask/OccupancyCut/depth/depth7', 
			'description':"""Occupancy after a cut.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '07 DIGI Occupancy Cut vs BX', [{'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HB', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HE', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HF', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsBX/Subdet/HO', 'description':"""Occupancy vs BX <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '08 DIGI Occupancy Cut vs LS', [{'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HB', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HE', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HF', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsLS/Subdet/HO', 'description':"""Occupancy Cut vs LS <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '09 DIGI Occupancy Cut vs iphi', [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HBM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HBP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HEM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HEP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HFM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HFP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HOM', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsiphi/SubdetPM/HOP', 'description':"""Occupancy vs iphi (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '10 DIGI Occupancy Cut vs ieta', [{'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HB', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HE', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HF', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/OccupancyCutvsieta/Subdet/HO', 'description':"""Occupancy vs ieta (Cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '11 DIGI Amplitude vs LS', [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HBP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HEP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HFP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOM', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsLS/SubdetPM/HOP', 'description':"""Signal Amplitude vs LS (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '12 DIGI Amplitude vs BX', [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HBM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HBP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HEM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HEP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HFM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HFP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HOM', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/SumQvsBX/SubdetPM/HOP', 'description':"""Signal Amplitude vs BX (cut is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '13 DIGI Timing', 
	[
		{
			'path':'Hcal/DigiTask/TimingCut/depth/depth1', 
			'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/TimingCut/depth/depth2', 
			'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/TimingCut/depth/depth3', 
			'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/DigiTask/TimingCut/depth/depth4', 
			'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/TimingCut/depth/depth5', 
			'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal/DigiTask/TimingCut/depth/depth6', 
			'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal/DigiTask/TimingCut/depth/depth7', 
			'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '14 DIGI Timing', [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HBP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HEP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HFP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOM', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCut/SubdetPM/HOP', 'description':"""Charge Weighted DIGI Timing (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '15 DIGI Timing vs iphi', [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HBM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HBP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HEM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HEP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HFM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HFP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HOM', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCutvsiphi/SubdetPM/HOP', 'description':"""Charge weighted DIGI Timing vs iphi (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '16 DIGI Timing vs ieta', [{'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HB', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HE', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HF', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingCutvsieta/Subdet/HO', 'description':"""Charge weighted DIGI Timing vs ieta (Cut on the signal amplitude is applied).  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '17 DIGI Timing vs LS', [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1100', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1102', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1104', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1106', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1108', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1110', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1112', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1114', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1116', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1118', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED1120', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1122', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED1132', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED724', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED725', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED726', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED727', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED728', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED729', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}, {'path':'Hcal/DigiTask/TimingvsLS/FED/FED730', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}], [{'path':'Hcal/DigiTask/TimingvsLS/FED/FED731', 'description':"""Timing either @DIGI level vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '18 RECO Energy', [{'path':'Hcal2/RecHitTask/Energy/Subdet/HB', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energy/Subdet/HE', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/Energy/Subdet/HF', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energy/Subdet/HO', 'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '19 RECO Energy', 
	[
		{
			'path':'Hcal2/RecHitTask/Energy/depth/depth1', 
			'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Energy/depth/depth2', 
			'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Energy/depth/depth3', 
			'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
	{
			'path':'Hcal2/RecHitTask/Energy/depth/depth4', 
			'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Energy/depth/depth5', 
			'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Energy/depth/depth6', 
			'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal2/RecHitTask/Energy/depth/depth7', 
			'description':"""Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '20 RECO Energy vs LS', [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HBM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HBP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HEM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HEP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HFM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HFP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HOM', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/EnergyvsLS/SubdetPM/HOP', 'description':"""Energy vs LS (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '21 RECO Energy vs ieta', [{'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HB', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HE', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HF', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energyvsieta/Subdet/HO', 'description':"""Energy vs ieta (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '22 RECO Energy vs iphi', [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HBM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HBP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HEM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HEP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HFM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HFP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HOM', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/Energyvsiphi/SubdetPM/HOP', 'description':"""Energy vs iphi (Cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '23 RECO Occupancy', 
	[
		{
			'path':'Hcal2/RecHitTask/Occupancy/depth/depth1', 
			'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Occupancy/depth/depth2', 
			'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Occupancy/depth/depth3', 
			'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal2/RecHitTask/Occupancy/depth/depth4', 
			'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Occupancy/depth/depth5', 
			'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/Occupancy/depth/depth6', 
			'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal2/RecHitTask/Occupancy/depth/depth7', 
			'description':"""Occupancy Distribution  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '25 RECO Occupancy Cut', 
	[
		{
			'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth1', 
			'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth2', 
			'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth3', 
			'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth4', 
			'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth5', 
			'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth6', 
			'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal2/RecHitTask/OccupancyCut/depth/depth7', 
			'description':"""Occupancy Distribution (cut is applied on energy)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '26 RECO Occupancy Cut vs LS', [{'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HB', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HE', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HF', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsLS/Subdet/HO', 'description':"""Occupancy vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '27 RECO Occupancy Cut vs ieta', [{'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HB', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HE', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HF', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsieta/Subdet/HO', 'description':"""Occupancy Distribution vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '28 RECO Occupancy Cut vs iphi', [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HBM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HBP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HEM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HEP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HFM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HFP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HOM', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/OccupancyCutvsiphi/SubdetPM/HOP', 'description':"""Occupancy Distribution vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '29 RECO Timing', 
	[
		{
			'path':'Hcal2/RecHitTask/TimingCut/depth/depth1', 
			'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/TimingCut/depth/depth2', 
			'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/TimingCut/depth/depth3', 
			'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal2/RecHitTask/TimingCut/depth/depth4', 
			'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/TimingCut/depth/depth5', 
			'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}, {
			'path':'Hcal2/RecHitTask/TimingCut/depth/depth6', 
			'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	], [
		{
			'path':'Hcal2/RecHitTask/TimingCut/depth/depth7', 
			'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""
		}
	]
)

hcallayout(dqmitems, '30 RECO Timing', [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HBM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HBP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HEM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HEP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HFM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HFP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HOM', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCut/SubdetPM/HOP', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '31 RECO Timing vs LS', [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1100', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1102', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1104', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1106', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1108', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1110', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1112', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1114', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1116', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1118', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1120', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1122', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED1132', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED724', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED725', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED726', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED727', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED728', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED729', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED730', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCutvsLS/FED/FED731', 'description':"""Timing @RECO vs LS (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '32 RECO Timing vs ieta', [{'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HB', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HE', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HF', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsieta/Subdet/HO', 'description':"""Timing @RECO vs ieta (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '33 RECO Timing vs iphi', [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HBM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HBP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HEM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HEP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HFM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HFP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HOM', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCutvsiphi/SubdetPM/HOP', 'description':"""Timing @RECO vs iphi (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '34 RECO HBHEabc Timing', [{'path':'Hcal2/RecHitTask/TimingCut/HBHEPartition/HBHEa', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingCut/HBHEPartition/HBHEb', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingCut/HBHEPartition/HBHEc', 'description':"""Timing @RECO (cut is applied)  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '35 RECO Timing vs Energy', [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HBM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HBP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HEM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HEP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HFM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HFP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}], [{'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HOM', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}, {'path':'Hcal2/RecHitTask/TimingvsEnergy/SubdetPM/HOP', 'description':"""Timing @RECO vs Energy Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#RecHit_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '36 TP Et Correlation', [{'path':'Hcal/TPTask/EtCorr/TTSubdet/HBHE', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCorr/TTSubdet/HF', 'description':"""Et Correlation Distributions. Emulator(Y) vs Data(X). Channels not present in respective Collections are plotted as Et=-2. 1x1 for HF is used  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '37 TP Et Correlation Ratio', [{'path':'Hcal/TPTask/EtCorrRatio/EtCorrRatio', 'description':"""Et Correlation Ratio. It is always min(etd, ete)/max(etd, ete).  Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '38 TP Et Correlation Ratio vs LS', [{'path':'Hcal/TPTask/EtCorrRatiovsLS/TTSubdet/HBHE', 'description':"""Et Correlation Ratio vs LS Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCorrRatiovsLS/TTSubdet/HF', 'description':"""Et Correlation Ratio vs LS Correlation Ratio is defined as min(Et_d,Et_e)/max(Et_d, Et_e) - namely, as the min/max between emulator and data Et. <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '40 TP Et Distributions', [{'path':'Hcal/TPTask/EtData/EtData', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtEmul/EtEmul', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '41 TP Et Distributions', [{'path':'Hcal/TPTask/EtData/TTSubdet/HBHE', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtData/TTSubdet/HF', 'description':"""Et Data Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/EtEmul/TTSubdet/HBHE', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtEmul/TTSubdet/HF', 'description':"""Et Emulator Distributions.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '42 TP Et(FG) Mismatches', [{'path':'Hcal/TPTask/EtMsm/EtMsm', 'description':"""Distribution of channels with mismatched Et  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/FGMsm/FGMsm', 'description':"""Distribution of channels with mismatched Fine Grain Bit <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '43 TP Et Mismatches Rate vs LS', [{'path':'Hcal/TPTask/EtMsmRatiovsLS/TTSubdet/HBHE', 'description':"""Rate of the Et Mismatches vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtMsmRatiovsLS/TTSubdet/HF', 'description':"""Rate of the Et Mismatches vs LS.  <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '44 TP Occupancy', [{'path':'Hcal/TPTask/OccupancyData/OccupancyData', 'description':"""Occupancy Distributions for Data <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyEmul/OccupancyEmul', 'description':"""Occupancy Distributions for Emulator <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '45 TP Occupancy Cut', [{'path':'Hcal/TPTask/OccupancyCutData/OccupancyCutData', 'description':"""Occupancy Distributions for Data with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyCutEmul/OccupancyCutEmul', 'description':"""Occupancy Distributions for Emulator with a cut on Et <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '46 TP Occupancy vs BX', [{'path':'Hcal/TPTask/OccupancyDatavsBX/TTSubdet/HBHE', 'description':"""Data Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyDatavsBX/TTSubdet/HF', 'description':"""Data Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/OccupancyEmulvsBX/TTSubdet/HBHE', 'description':"""Emul Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyEmulvsBX/TTSubdet/HF', 'description':"""Emul Occupancy vs BX (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '47 TP Occupancy vs LS', [{'path':'Hcal/TPTask/OccupancyDatavsLS/TTSubdet/HBHE', 'description':"""Data Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyDatavsLS/TTSubdet/HF', 'description':"""Data Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/OccupancyEmulvsLS/TTSubdet/HBHE', 'description':"""Emul Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyEmulvsLS/TTSubdet/HF', 'description':"""Emul Occupancy vs LS (no cuts applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '48 TP Occupancy Cut vs BX', [{'path':'Hcal/TPTask/OccupancyCutDatavsBX/TTSubdet/HBHE', 'description':"""Data Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyCutDatavsBX/TTSubdet/HF', 'description':"""Data Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/OccupancyCutEmulvsBX/TTSubdet/HBHE', 'description':"""Emul Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyCutEmulvsBX/TTSubdet/HF', 'description':"""Emul Occupancy vs BX (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '49 TP Occupancy Cut vs LS', [{'path':'Hcal/TPTask/OccupancyCutDatavsLS/TTSubdet/HBHE', 'description':"""Data Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyCutDatavsLS/TTSubdet/HF', 'description':"""Data Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/OccupancyCutEmulvsLS/TTSubdet/HBHE', 'description':"""Emul Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/OccupancyCutEmulvsLS/TTSubdet/HF', 'description':"""Emul Occupancy vs LS (cut applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '50 TP Et Data vs BX(LS)', [{'path':'Hcal/TPTask/EtCutDatavsBX/TTSubdet/HBHE', 'description':"""Et Data vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCutDatavsBX/TTSubdet/HF', 'description':"""Et Data vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/EtCutDatavsLS/TTSubdet/HBHE', 'description':"""Et Data vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCutDatavsLS/TTSubdet/HF', 'description':"""Et Data vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '51 TP Et Emul vs BX(LS)', [{'path':'Hcal/TPTask/EtCutEmulvsBX/TTSubdet/HBHE', 'description':"""Et Emulator vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCutEmulvsBX/TTSubdet/HF', 'description':"""Et Emulator vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/EtCutEmulvsLS/TTSubdet/HBHE', 'description':"""Et Emulator vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCutEmulvsLS/TTSubdet/HF', 'description':"""Et Emulator vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '51 TP Et Emul vs BX(LS)', [{'path':'Hcal/TPTask/EtCutEmulvsBX/TTSubdet/HBHE', 'description':"""Et Emulator vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCutEmulvsBX/TTSubdet/HF', 'description':"""Et Emulator vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}], [{'path':'Hcal/TPTask/EtCutEmulvsLS/TTSubdet/HBHE', 'description':"""Et Emulator vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}, {'path':'Hcal/TPTask/EtCutEmulvsLS/TTSubdet/HF', 'description':"""Et Emulator vs LS (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])

hcallayout(dqmitems, '52 TP uHTR-L1T mismatch', [{'path':'Hcal/TPTask/EtMsm_uHTR_L1T/EtMsm_uHTR_L1T', 'description':"""TP mismatches between uHTR and L1T <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#TP_Task_Description'>Details...</a>"""}])


hcallayout(dqmitems, '53 LED pin diode ADC vs BX', [{'path':'Hcal/DigiTask/LED/LED_ADCvsBX', 'description':"""Et Emulator vs BX (cut is applied) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""}])


hcallayout(dqmitems, '54 (CapId-BX)%4', 
	[
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HBM', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HBP', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HEM', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HEP', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
	],
	[
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HFM', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HFP', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HOM', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
		{
			'path':'Hcal/DigiTask/CapID/SubdetPM/HOP', 'description':"""(CapId-BX)%4 (should have exactly one value) <a href='https://twiki.cern.ch/twiki/bin/view/CMS/HcalDQMRun2TaskDescription#Digi_Task_Description'>Details...</a>"""
		},
	]
)

hcallayout(dqmitems, '55 HF TDC cut efficiency', 
	[
		{
			'path':'Hcal/TPTask/TDCCutEfficiency_depth', 'description':"""Efficiency of HF dual anode TDC cut"""
		},
		{
			'path':'Hcal/TPTask/TDCCutEfficiency_ieta', 'description':"""Efficiency of HF dual anode TDC cut"""
		},
	]
)
apply_dqm_items_to_new_back_end(dqmitems, __file__)
