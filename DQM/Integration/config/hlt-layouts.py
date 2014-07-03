def hltlayout(i, p, *rows): i["HLT/Layouts/" + p] = DQMItem(layout=rows)

def hlt_evInfo_single(i, dir, name):
  i["HLT/Layouts/00-FourVector-Summary/%s" % name] = \
    DQMItem(layout=[["HLT/%s/%s" % (dir, name)]]) 



# list of summary GT histograms (dqmitems, dirPath , histoName)
hlt_evInfo_single(dqmitems, "FourVectorHLT", "HLT1Electron_etaphi")

########################################
########## TPG Summary #################
#######################################

def tpgSummary_l1t(i, p, *rows): 
   i["L1T/Layouts/TPG-Summary-L1T/" + p] = DQMItem(layout=rows)


tpgSummary_l1t(dqmitems, "01 - L1 Predeadtime Rate - Physics",
           [{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/Physics Trigger Rate", 'description':"Physics Predeadtime"}])
           

tpgSummary_l1t(dqmitems, "02 - L1 Predeadtime Rate - Technical",
           [{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_005", 'description':"Technical Predeadtime"}])
           

tpgSummary_l1t(dqmitems, "03.01 - Muon Timing DT vs CSC",
           [{'path': "L1T/L1TGMT/bx_DT_vs_CSC", 'description':"Muon Timing"}])

tpgSummary_l1t(dqmitems, "03.02 - Muon Timing DT vs RPC",
           [{'path': "L1T/L1TGMT/bx_DT_vs_RPC", 'description':"Muon Timing"}])

tpgSummary_l1t(dqmitems, "03.03 - Muon Timing DT vs RPC",
           [{'path': "L1T/L1TGMT/bx_CSC_vs_RPC", 'description':"Muon Timing"}])




######################## TPG HLT ################################3

def tpgSummary_hlt(i, p, *rows): 
   i["HLT/Layouts/TPG-Summary-HLT/" + p] = DQMItem(layout=rows)


tpgSummary_hlt(dqmitems, "01 - HLT Postdeadtime Rate",
           [{'path': "HLT/HLTScalers_EvF/hltRateNorm", 'description':"HLT Rate Postdeadtime"}])

tpgSummary_hlt(dqmitems, "02 - HLT MinBiasBSC Rate",
           [{'path': "HLT/HLTScalers_EvF/RateHistory/norm_rate_p093", 'description':"HLT MinBias BSC Rate per lumi sec"}])

tpgSummary_hlt(dqmitems, "03 - HLT ZeroBias Rate",
           [{'path': "HLT/HLTScalers_EvF/RateHistory/norm_rate_p092", 'description':"HLT Zero Bias Rate per lumi sec"}])


############  Muon

tpgSummary_hlt(dqmitems, "04 - Muon POG HLT",
           [{'path': "HLT/HLTMonMuon/L3Triggers/Level3/HLTMuonL3_etaphi", 'description':"Muons Passing HLT_Mu3 Occupancy phi vs eta"}])

tpgSummary_hlt(dqmitems, "05 - Muon POG HLT",
           [{'path': "HLT/HLTMonMuon/L3Triggers/Level3/HLTMuonL3_pt", 'description':"Muons Passing HLT_Mu3 Occpancy vs Pt"}])

tpgSummary_hlt(dqmitems, "06 - Muon POG L1 Passthru",
           [{'path': "HLT/HLTMonMuon/L1PassThrough/Level1/HLTMuonL1_etaphi", 'description':"Muons Passing HLT_L1Mu Occpancy phi vs eta"}])

tpgSummary_hlt(dqmitems, "07 - Muon POG L1 Passthru",
           [{'path': "HLT/HLTMonMuon/L1PassThrough/Level1/HLTMuonL1_pt", 'description':"Muons Passing HLT_L1Mu Occpancy vs Pt"}])

######### Jet Met 

tpgSummary_hlt(dqmitems, "08 - JET POG HLT",
           [{'path': "HLT/JetMET/All/HLT_Jet15U/HLT_Jet15U_EtaPhi", 'description':"HLT_Jet15U occupancy eta vs phi"}])

tpgSummary_hlt(dqmitems, "09 - JET POG HLT",
           [{'path': "HLT/JetMET/All/HLT_Jet15U/HLT_Jet15U_Et", 'description':"HLT_Jet15U occupancy Et"}])

tpgSummary_hlt(dqmitems, "10 - JET POG HLT L1 Passthru",
           [{'path': "HLT/JetMET/All/HLT_L1Jet6U/HLT_L1Jet6U_EtaPhi", 'description':"HLT_L1Jet6U occupancy, eta vs phi"}])

tpgSummary_hlt(dqmitems, "11 - JET POG HLT L1 Passthru",
           [{'path': "HLT/JetMET/All/HLT_L1Jet6U/HLT_L1Jet6U_Et", 'description':"HLT_L1Jet6U occupancy, Et "}])

######### EGamma

tpgSummary_hlt(dqmitems, "12 - EG POG HLT ",
           [{'path': "HLT/FourVector/paths/HLT_Ele10_LW_L1R/HLT_Ele10_LW_L1R_wrt__l1Etal1PhiL1On", 'description':"HLT_Ele10_LW_L1R occupancy, eta vs phi"}])

tpgSummary_hlt(dqmitems, "13 - EG POG HLT",
           [{'path': "HLT/FourVector/paths/HLT_Ele10_LW_L1R/HLT_Ele10_LW_L1R_wrt__l1EtL1", 'description':""}])


######## Tau

tpgSummary_hlt(dqmitems, "14 - Tau POG HLT",
           [{'path': "HLT/FourVector/paths/HLT_SingleLooseIsoTau20/HLT_SingleLooseIsoTau20_wrt__l1EtL1", 'description':"SingleLooseIsoTau20 et"}])

tpgSummary_hlt(dqmitems, "15 - Tau POG HLT",
           [{'path': "HLT/FourVector/paths/HLT_SingleLooseIsoTau20/HLT_SingleLooseIsoTau20_wrt__l1Etal1PhiL1OnUM", 'description':"SingleLooseIsoTau20 eta vs phi"}])

