                  
# def pixellumilayout(dqmitems,layout, *plots):
#     destination_1='PixelLumi/Layouts/'
#     destination=''
    
#     parts_of_layout=layout.split('/')
#     layout_name=parts_of_layout[len(parts_of_layout)-1]
#     for plot_list in plots:
#       for plot in plot_list:
#         plot_path_parts=plot.split('/')
#         plot_name=plot_path_parts[len(plot_path_parts)-1]
#         overlay=''
#         description=''
#         source=plot
#         destination=destination_1+destination+plot_name
#         name=layout_name

#         print('register_layout(source=\'%s\', destination=\'%s\', name=\'%s\', description=\'%s\', overlay=\'%s\')'%(source, destination, name, description, overlay))
        
dqmitems=''
print('from ..layouts.layout_manager import register_layout\n')


def rateSummarylayout(dqmitems,layout, *plots):
    destination_1=''
    parts_of_layout=layout.split('/')
    layout_name=parts_of_layout[len(parts_of_layout)-1]
    for plot_list in plots:
        for plot in plot_list:
            destination=''
            destination_path=''
            if(plot != None):
               if(isinstance(plot, str)):
                  plot_path_parts=plot.split('/')
               else:
                  plot_path_parts=plot['path'].split('/')
                  plot_name=plot_path_parts[len(plot_path_parts)-1]
                  if(len(parts_of_layout) > 1):
                     destination_path=parts_of_layout[0:len(parts_of_layout)-1]
                     destination_path.append(plot_name)
                     destination='/'.join(destination_path)
                  destination=destination_1+destination+plot_name
                  source=plot['path']
                  description=''
                  if 'description' in plot:
                    description=plot['description']
                  description=description.replace('<a', '')
                  description=description.replace('</a>', '')
                  overlay=''
                  if 'overlay' in plot:
                    overlay=plot['overlay']
                  name=layout_name
                  print('register_layout(source=\'%s\', destination=\'%s\', name=\'%s\', description=\'%s\', overlay=\'%s\')'%(source, destination, name, description, overlay))
rateSummarylayout(
  dqmitems,
  "HLT/Layouts/highestRate Summary",
  [{'path':'HLT/TriggerRates/HLT/HLT_PFJet200_v2 accept', 'description':'# accepts per LS (divide by 23.3 to get rate in Hz)'},
   {'path':'HLT/TriggerRates/HLT/HLT_Ele27_eta2p1_WPLoose_Gsf_v1 accept', 'description':'# accepts per LS (divide by 23.3 to get rate in Hz)'}],
  [{'path':'HLT/TriggerRates/HLT/HLT_Photon30_R9Id90_HE10_IsoM_v2 accept', 'description':'# accepts per LS (divide by 23.3 to get rate in Hz)'},
   {'path':'HLT/TriggerRates/HLT/HLT_IsoMu24_eta2p1_v2 accept', 'description':'# accepts per LS (divide by 23.3 to get rate in Hz)'}]
                 )

def hlt_secondaryObjMonFolder(i, path, name):
  i["HLT/Layouts/secondaryObjectMonitor/%s" % name] = \
    DQMItem(layout=[[path]])
def hlt_secondaryObjMonFolder(dqmitems,p, n):
    layout=''
    plots=[[{'path': "HLT/Layouts/secondaryObjectMonitor/%s" % n}]]
    destination_1='Collisions/HLTFeedBack/'
    parts_of_layout=layout.split('/')
    layout_name=parts_of_layout[len(parts_of_layout)-1]
    for plot_list in plots:
        for plot in plot_list:
            destination=''
            destination_path=''
            if(plot != None):
               if(isinstance(plot, str)):
                  plot_path_parts=plot.split('/')
               else:
                  plot_path_parts=plot['path'].split('/')
                  plot_name=plot_path_parts[len(plot_path_parts)-1]
                  if(len(parts_of_layout) > 1):
                     destination_path=parts_of_layout[0:len(parts_of_layout)-1]
                     destination_path.append(plot_name)
                     destination='/'.join(destination_path)
                  destination=destination_1+destination+plot_name
                  source=plot['path']
                  description=''
                  if 'description' in plot:
                    description=plot['description']
                  description=description.replace('<a', '')
                  description=description.replace('</a>', '')
                  overlay=''
                  if 'overlay' in plot:
                    overlay=plot['overlay']
                  name=layout_name
                  print('register_layout(source=\'%s\', destination=\'%s\', description=\'%s\', overlay=\'%s\')'%(source, destination, description, overlay))

hlt_secondaryObjMonFolder(dqmitems,"HLT/Tracking/iter2Merged/GeneralProperties/TrackEta_ImpactPoint_GenTk", "TrackEta_ImpactPoint_GenTk")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/bJet_phi", "bJet_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/Photon_eta","Photon_eta")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/Photon_phi","Photon_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/Muon_eta","Muon_eta")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/Muon_phi","Muon_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/L2Muon_eta","L2Muon_eta")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/L2Muon_phi","L2Muon_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/L2NoBPTXMuon_eta","L2NoBPTXMuon_eta")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/L2NoBPTXMuon_phi","L2NoBPTXMuon_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/Electron_eta","Electron_eta")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/Electron_phi","Electron_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/CaloMET_phi","CaloMET_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/PFMET_phi","PFMET_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/bJet_phi","bJet_phi")
hlt_secondaryObjMonFolder(dqmitems,"HLT/ObjectMonitor/Backup/bJet_eta","bJet_eta")
hlt_secondaryObjMonFolder(dqmitems,"HLT/Tracking/iter2Merged/GeneralProperties/TrackEta_ImpactPoint_GenTk","TrackEta_ImpactPoint_GenTk__iter2Merged")
hlt_secondaryObjMonFolder(dqmitems,"HLT/Tracking/pixelTracks/GeneralProperties/TrackEta_ImpactPoint_GenTk","TrackEta_ImpactPoint_GenTk__pixelTracks")
hlt_secondaryObjMonFolder(dqmitems,"HLT/Tracking/iter2Merged/GeneralProperties/TrackPhi_ImpactPoint_GenTk","MergedTrackPhi_ImpactPoint_GenTk__iter2Merged")
hlt_secondaryObjMonFolder(dqmitems,"HLT/Tracking/pixelTracks/GeneralProperties/TrackPhi_ImpactPoint_GenTk","TrackPhi_ImpactPoint_GenTk__pixelTracks")
hlt_secondaryObjMonFolder(dqmitems,"HLT/Tracking/iter2Merged/GeneralProperties/NumberOfTracks_GenTk","NumberOfTracks_GenTk__iter2Merged")
hlt_secondaryObjMonFolder(dqmitems,"HLT/Tracking/pixelTracks/GeneralProperties/NumberOfTracks_GenTk","NumberOfTracks_GenTk__pixelTracks")
hlt_secondaryObjMonFolder(dqmitems,"HLT/SiStrip/MechanicalView/TIB/Summary_ClusterCharge_OffTrack__TIB","Summary_ClusterCharge_OffTrack__TIB")
hlt_secondaryObjMonFolder(dqmitems,"HLT/SiStrip/MechanicalView/TIB/Summary_ClusterStoNCorr_OnTrack__TIB","Summary_ClusterStoNCorr_OnTrack__TIB")
hlt_secondaryObjMonFolder(dqmitems,"HLT/SiStrip/MechanicalView/TOB/Summary_ClusterStoNCorr_OnTrack__TOB","Summary_ClusterStoNCorr_OnTrack__TOB")
hlt_secondaryObjMonFolder(dqmitems,"HLT/SiStrip/MechanicalView/TID/MINUS/Summary_ClusterStoNCorr_OnTrack__TID__MINUS","Summary_ClusterStoNCorr_OnTrack__TID__MINUS")
hlt_secondaryObjMonFolder(dqmitems,"HLT/SiStrip/MechanicalView/TID/PLUS/Summary_ClusterStoNCorr_OnTrack__TID__PLUS","Summary_ClusterStoNCorr_OnTrack__TID__PLUS")
hlt_secondaryObjMonFolder(dqmitems,"HLT/SiStrip/MechanicalView/TEC/MINUS/Summary_ClusterStoNCorr_OnTrack__TEC__MINUS","Summary_ClusterStoNCorr_OnTrack__TEC__MINUS")
hlt_secondaryObjMonFolder(dqmitems,"HLT/SiStrip/MechanicalView/TEC/PLUS/Summary_ClusterStoNCorr_OnTrack__TEC__PLUS","Summary_ClusterStoNCorr_OnTrack__TEC__PLUS")

def hlt_evInfo_single(dqmitems,dir, n):
    layout=''
    plots=[[{'path': "HLT/%s/%s" % (dir, n)}]]
    destination_1='HLT/Layouts/00-FourVector-Summary/'
    parts_of_layout=layout.split('/')
    layout_name=parts_of_layout[len(parts_of_layout)-1]
    for plot_list in plots:
        for plot in plot_list:
            destination=''
            destination_path=''
            if(plot != None):
               if(isinstance(plot, str)):
                  plot_path_parts=plot.split('/')
               else:
                  plot_path_parts=plot['path'].split('/')
                  plot_name=plot_path_parts[len(plot_path_parts)-1]
                  if(len(parts_of_layout) > 1):
                     destination_path=parts_of_layout[0:len(parts_of_layout)-1]
                     destination_path.append(plot_name)
                     destination='/'.join(destination_path)
                  destination=destination_1+destination+plot_name
                  source=plot['path']
                  description=''
                  if 'description' in plot:
                    description=plot['description']
                  description=description.replace('<a', '')
                  description=description.replace('</a>', '')
                  overlay=''
                  if 'overlay' in plot:
                    overlay=plot['overlay']
                  name=layout_name
                  print('register_layout(source=\'%s\', destination=\'%s\', description=\'%s\', overlay=\'%s\')'%(source, destination, description, overlay))
# list of summary GT histograms (dqmitems, dirPath , histoName)
hlt_evInfo_single(dqmitems, "FourVectorHLT", "HLT1Electron_etaphi")

########################################
########## TPG Summary #################
#######################################

def tpgSummary_l1t(dqmitems,layout, *plots):
    destination_1='L1T/Layouts/TPG-Summary-L1T/'
    parts_of_layout=layout.split('/')
    layout_name=parts_of_layout[len(parts_of_layout)-1]
    for plot_list in plots:
        for plot in plot_list:
            destination=''
            destination_path=''
            if(plot != None):
               if(isinstance(plot, str)):
                  plot_path_parts=plot.split('/')
               else:
                  plot_path_parts=plot['path'].split('/')
                  plot_name=plot_path_parts[len(plot_path_parts)-1]
                  if(len(parts_of_layout) > 1):
                     destination_path=parts_of_layout[0:len(parts_of_layout)-1]
                     destination_path.append(plot_name)
                     destination='/'.join(destination_path)
                  destination=destination_1+destination+plot_name
                  source=plot['path']
                  description=''
                  if 'description' in plot:
                    description=plot['description']
                  description=description.replace('<a', '')
                  description=description.replace('</a>', '')
                  overlay=''
                  if 'overlay' in plot:
                    overlay=plot['overlay']
                  name=layout_name
                  print('register_layout(source=\'%s\', destination=\'%s\', name=\'%s\', description=\'%s\', overlay=\'%s\')'%(source, destination, name, description, overlay))

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
def tpgSummary_hlt(dqmitems,layout, *plots):
    destination_1='HLT/Layouts/TPG-Summary-HLT/'
    parts_of_layout=layout.split('/')
    layout_name=parts_of_layout[len(parts_of_layout)-1]
    for plot_list in plots:
        for plot in plot_list:
            destination=''
            destination_path=''
            if(plot != None):
               if(isinstance(plot, str)):
                  plot_path_parts=plot.split('/')
               else:
                  plot_path_parts=plot['path'].split('/')
                  plot_name=plot_path_parts[len(plot_path_parts)-1]
                  if(len(parts_of_layout) > 1):
                     destination_path=parts_of_layout[0:len(parts_of_layout)-1]
                     destination_path.append(plot_name)
                     destination='/'.join(destination_path)
                  destination=destination_1+destination+plot_name
                  source=plot['path']
                  description=''
                  if 'description' in plot:
                    description=plot['description']
                  description=description.replace('<a', '')
                  description=description.replace('</a>', '')
                  overlay=''
                  if 'overlay' in plot:
                    overlay=plot['overlay']
                  name=layout_name
                  print('register_layout(source=\'%s\', destination=\'%s\', name=\'%s\', description=\'%s\', overlay=\'%s\')'%(source, destination, name, description, overlay))

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
