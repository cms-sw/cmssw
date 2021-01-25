from .adapt_to_new_backend import *
dqmitems={}

# BVB
# 20150311 For HLT currently the DQM shifter is asked to do nothing
#          That's why we remove at least the shift workspace for HLT now
#          We do this be disabling the method body in the method definitions.

# The shift workspace is split in 2 folders:
# One for cosmics, one for collisions
# (BVB: Personally I think this is very clear and hence a good idea)

# Cosmics
def hltlayout(i, p, *rows):
    pass
    ##i["00 Shift/HLT/Cosmics/" + p] = rows

hltlayout(dqmitems,"01 HLT_Commissioning_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_Commissioning_Pass_Any",
      'description': "Shows total number of HLT Commissioning trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"02 HLT_Cosmics_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_Cosmics_Pass_Any",
      'description': "Shows total number of HLT Cosmics trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"03 HLT_ForwardTriggers_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_ForwardTriggers_Pass_Any",
      'description': "Shows total number of HLT ForwardTriggers trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"04 HLT_HcalHPDNoise_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_HcalHPDNoise_Pass_Any",
      'description': "Shows total number of HLT HcalHPDNoise trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"05 HLT_HcalNZS_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_HcalNZS_Pass_Any",
      'description': "Shows total number of HLT HcalNZS trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

# Collisions
def hltlayout(i, p, *rows):
    pass
    ##i["00 Shift/HLT/Collisions/" + p] = rows

hltlayout(dqmitems,"01 HLT_Jet_Xsec",
    [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_Jet_Xsec",
      'description': "Shows total number of Jet PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"02 HLT_SingleElectron_Xsec",
    [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_SingleElectron_Xsec",
      'description': "Shows total number of SingleElectron PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"03 HLT_SingleMu_Xsec",
    [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_SingleMu_Xsec",
      'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"04 HLT_Jet_Occupancy",
    [{'path': "HLT/OccupancyPlots/PD_EtaVsPhi/HLT_Jet_EtaVsPhiFine",
      'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"05 HLT_SingleElectron_Occupancy",
    [{'path': "HLT/OccupancyPlots/PD_EtaVsPhi/HLT_SingleElectron_EtaVsPhiFine",
      'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"06 HLT_SingleMu_Occupancy",
    [{'path': "HLT/OccupancyPlots/PD_EtaVsPhi/HLT_SingleMu_EtaVsPhiFine",
      'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
