                  
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

def trigvalegammaZ(dqmitems,layout, *plots):
    destination_1='HLT/HLTEgammaValidation/Zee Preselection/'
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
###
# """
# Layout file for trigger release validation
#  facilitating organization/regrouping of subsystem histograms
# -use subsystem top-folder as specified in previous stages eg
#  def trigval<subsys>(i, p, *rows): i["HLT/<subsys>/Preselection" + p] = DQMItem(layout=rows)
# """
###

###---- EGAMMA selection goes here: ----

trigvalegammaZ(dqmitems,"doubleEle5SWL1R/total",
           [{'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_by_step_MC_matched", 'description':"per-event efficiency (MC matched) for doubleEle5SWL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_by_step", 'description':"per-event efficiency (MC matched) for doubleEle5SWL1R"}])

trigvalegammaZ(dqmitems,"doubleEle5SWL1R/kinematics",
           [{'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/gen_et", 'description':"per-event efficiency (MC matched) for doubleEle5SWL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/gen_eta", 'description':"per-event efficiency (MC matched) for doubleEle5SWL1R"}])

trigvalegammaZ(dqmitems,"doubleEle5SWL1R/L1 match",
           [{'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional in doubleEle5SWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional in doubleEle5SWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"doubleEle5SWL1R/Et cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter in doubleEle5SWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter in doubleEle5SWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"doubleEle5SWL1R/Hcal isolation",
           [{'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5HcalIsolFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter in doubleEle5SWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5HcalIsolFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter in doubleEle5SWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"doubleEle5SWL1R/pixel match",
           [{'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter in doubleEle5SWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_DoubleEle5_SW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter in doubleEle5SWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWL1R/total",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_by_step_MC_matched", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_by_step", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"}])

trigvalegammaZ(dqmitems,"Ele10LWL1R/kinematics",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/gen_et", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/gen_eta", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"}])

trigvalegammaZ(dqmitems,"Ele10LWL1R/L1 match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional in Ele10LWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWL1R/Et cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter in Ele10LWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWL1R/Hcal isolation",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter in Ele10LWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWL1R/pixel match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter in Ele10LWL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/total",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_by_step_MC_matched", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_by_step", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/kinematics",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/gen_et", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/gen_eta", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/L1 match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional_vs_et_MC_matched" , 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional in Ele10LWEleIdL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/Et cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter_vs_et_MC_matched",  'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter in Ele10LWEleIdL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/cluster shape cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter in Ele10LWEleIdL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/Hcal isolation",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter_vs_et_MC_matched",  'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter in Ele10LWEleIdL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/pixel match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter_vs_et_MC_matched",  'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter in Ele10LWEleIdL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/ 1oE - 1op cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter in Ele10LWEleIdL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/ delta-eta cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter in Ele10LWEleIdL1R_vs_et"}])

trigvalegammaZ(dqmitems,"Ele10LWEleIdL1R/ delta-phi cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMZee/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter in Ele10LWEleIdL1R_vs_et"}])

def hltlayoutW(dqmitems,layout, *plots):
    destination_1='HLT/HLTEgammaValidation/Wenu Preselection/'
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
###
hltlayoutW(dqmitems,"Ele10LWL1R/total",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_by_step_MC_matched", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_by_step", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"}])

hltlayoutW(dqmitems,"Ele10LWL1R/kinematics",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/gen_et", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/gen_eta", 'description':"per-event efficiency (MC matched) for Ele10LWL1R"}])

hltlayoutW(dqmitems,"Ele10LWL1R/L1 match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional in Ele10LWL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWL1R/Et cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter in Ele10LWL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWL1R/Hcal isolation",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter in Ele10LWL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWL1R/pixel match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter in Ele10LWL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter in Ele10LWL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/total",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_by_step_MC_matched", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_by_step", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/kinematics",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/gen_et", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/gen_eta", 'description':"per-event efficiency (MC matched) for Ele10LWEleIdL1R"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/L1 match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional_vs_et_MC_matched" , 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdL1MatchFilterRegional in Ele10LWEleIdL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/Et cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter_vs_et_MC_matched",  'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdEtFilter in Ele10LWEleIdL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/cluster shape cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdClusterShapeFilter in Ele10LWEleIdL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/Hcal isolation",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter_vs_et_MC_matched",  'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdHcalIsolFilter in Ele10LWEleIdL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/pixel match",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter_vs_et_MC_matched",  'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdPixelMatchFilter in Ele10LWEleIdL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/ 1oE - 1op cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdOneOEMinusOneOPFilter in Ele10LWEleIdL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/ delta-eta cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDetaFilter in Ele10LWEleIdL1R_vs_et"}])

hltlayoutW(dqmitems,"Ele10LWEleIdL1R/ delta-phi cut",
           [{'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter_vs_eta_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter in Ele10LWEleIdL1R_vs_eta"},
            {'path': "HLT/HLTEgammaValidation/HLT_Ele10_LW_EleId_L1RDQMWenu/efficiency_hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter_vs_et_MC_matched", 'description':"per-object (MC matched) for hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter in Ele10LWEleIdL1R_vs_et"}])

def hltLayoutGammaJet(dqmitems,layout, *plots):
    destination_1='HLT/HLTEgammaValidation/Photon Summary'
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
###

hltLayoutGammaJet(dqmitems,"/HLT_Photon10_L1R Efficiency vs Et",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon10_L1R_DQMGammaJet/final_eff_vs_et", 'description':"Efficiency of HLT_Photon10_L1R vs Et of generated photon"}])
hltLayoutGammaJet(dqmitems,"/HLT_Photon10_L1R Efficiency vs eta",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon10_L1R_DQMGammaJet/final_eff_vs_eta", 'description':"Efficiency of HLT_Photon10_L1R vs eta of generated photon"}])
hltLayoutGammaJet(dqmitems,"/L1 EgammaEt5 Efficiency vs et",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon10_L1R_DQMGammaJet/efficiency_hltL1sRelaxedSingleEgammaEt5_vs_et_MC_matched", 'description':"Efficiency of L1 EgammaEt5 vs et of generated photon"}])
hltLayoutGammaJet(dqmitems,"/L1 EgammaEt5 Efficiency vs eta",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon10_L1R_DQMGammaJet/efficiency_hltL1sRelaxedSingleEgammaEt5_vs_eta_MC_matched", 'description':"Efficiency of L1 EgammaEt5 vs eta of generated photon"}])
hltLayoutGammaJet(dqmitems,"/HLT_Photon15_LooseEcalIso_L1R Efficiency vs Et",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon15_LooseEcalIso_L1R_DQMGammaJet/final_eff_vs_et",'description':"Efficiency of HLT_Photon15_LooseEcalIso_L1R vs Et of generated photon"}])
hltLayoutGammaJet(dqmitems,"/HLT_Photon15_LooseEcalIso_L1R Efficiency vs eta",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon15_LooseEcalIso_L1R_DQMGammaJet/final_eff_vs_eta",'description':"Efficiency of HLT_Photon15_LooseEcalIso_L1R vs eta of generated photon"}])
hltLayoutGammaJet(dqmitems,"/HLT_Photon15_TrackIso_L1R Efficiency vs Et",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon15_TrackIso_L1R_DQMGammaJet/final_eff_vs_et",'description':"Efficiency of HLT_Photon15_TrackIso_L1R vs Et of generated photon"}])
hltLayoutGammaJet(dqmitems,"/HLT_Photon15_TrackIso_L1R Efficiency vs eta",
                  [{'path':"HLT/HLTEgammaValidation/HLT_Photon15_TrackIso_L1R_DQMGammaJet/final_eff_vs_eta",'description':"Efficiency of HLT_Photon15_TrackIso_L1R vs eta of generated photon"}])

###---- MUON selection goes here: ----

def trigvalmuon(dqmitems,layout, *plots):
    destination_1='HLT/Muon/Efficiency_Layouts/'
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
###
paths = ['HLT_DoubleMu0', 'HLT_DoubleMu3_v2', 'HLT_DoubleMu3_v3', 'HLT_DoubleMu3_v5', 'HLT_DoubleMu5_v1', 'HLT_DoubleMu6_v1', 'HLT_DoubleMu6_v3', 'HLT_DoubleMu7_v1', 'HLT_DoubleMu7_v3', 'HLT_IsoMu11_v4', 'HLT_IsoMu12_v1', 'HLT_IsoMu12_v3', 'HLT_IsoMu13_v4', 'HLT_IsoMu15_v4', 'HLT_IsoMu15_v5', 'HLT_IsoMu15_v7', 'HLT_IsoMu17_v4', 'HLT_IsoMu17_v5', 'HLT_IsoMu17_v7', 'HLT_IsoMu24_v1', 'HLT_IsoMu24_v3', 'HLT_IsoMu30_v1', 'HLT_IsoMu30_v3', 'HLT_IsoMu9_v4', 'HLT_L1DoubleMu0_v1', 'HLT_L1DoubleMu0_v2', 'HLT_L1DoubleMuOpen', 'HLT_L1Mu20', 'HLT_L1Mu7_v1', 'HLT_L1MuOpen_v2', 'HLT_L2DoubleMu0', 'HLT_L2DoubleMu0_v2', 'HLT_L2DoubleMu0_v4', 'HLT_L2DoubleMu20_NoVertex_v1', 'HLT_L2DoubleMu23_NoVertex_v3', 'HLT_L2DoubleMu35_NoVertex_v1', 'HLT_L2Mu0_NoVertex', 'HLT_L2Mu10_v1', 'HLT_L2Mu10_v3', 'HLT_L2Mu20_v1', 'HLT_L2Mu20_v3', 'HLT_L2Mu30_v1', 'HLT_L2Mu7_v1', 'HLT_L2MuOpen_NoVertex_v1', 'HLT_Mu0_v2', 'HLT_Mu11', 'HLT_Mu12_v1', 'HLT_Mu12_v3', 'HLT_Mu13_v1', 'HLT_Mu15_v1', 'HLT_Mu15_v2', 'HLT_Mu15_v4', 'HLT_Mu17_v1', 'HLT_Mu19_v1', 'HLT_Mu20_v1', 'HLT_Mu20_v3', 'HLT_Mu21_v1', 'HLT_Mu24_v1', 'HLT_Mu24_v3', 'HLT_Mu25_v1', 'HLT_Mu30_NoVertex_v1', 'HLT_Mu30_v1', 'HLT_Mu30_v3', 'HLT_Mu3_v2', 'HLT_Mu3_v3', 'HLT_Mu3_v5', 'HLT_Mu40_v1', 'HLT_Mu5', 'HLT_Mu5_v3', 'HLT_Mu5_v5', 'HLT_Mu7', 'HLT_Mu8_v1', 'HLT_Mu8_v3', 'HLT_Mu9']

for thisPath in paths:

    thisDir           = "HLT/Muon/Distributions/" + thisPath
    thisDocumentation = " (" + thisPath + " path) (<a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerValidationMuon\">documentation</a>)"

##     trigvalmuon(dqmitems, thisPath + "/1: Trigger Path Efficiency Summary",
##         [{'path': "HLT/Muon/Summary/Efficiency_Summary", 'description':"Percentage of total events in the sample which pass each muon trigger"}])
    trigvalmuon(dqmitems, thisPath + "/1: Efficiency L3 vs. Reco",
        [{'path': thisDir + "/recEffEta_Total", 'description':"Efficiency to find an L3 muon associated to a reconstructed muon vs. pT" + thisDocumentation}])
    trigvalmuon(dqmitems, thisPath + "/2: pT Turn-On L3 vs. Reco",
        [{'path': thisDir + "/recTurnOn1_Total", 'description':"Efficiency to find an L3 muon associated to a reconstructed muon vs. pT" + thisDocumentation}])
    trigvalmuon(dqmitems, thisPath + "/pT Turn-On L1 vs. Gen",
        [{'path': thisDir + "/genTurnOn1_L1", 'description':"Efficiency to find an L1 muon associated to a generated muon vs. pT" + thisDocumentation}])
    trigvalmuon(dqmitems, thisPath + "/pT Turn-On L2 vs. Gen",
        [{'path': thisDir + "/genTurnOn1_L2", 'description':"Efficiency to find a gen-matched L2 muon associated to a gen-matched L1 muon vs. pT" + thisDocumentation}])
    trigvalmuon(dqmitems, thisPath + "/pT Turn-On L3 vs. Gen",
        [{'path': thisDir + "/genTurnOn1_L3", 'description':"Efficiency to find a gen-matched L3 muon associated to a gen-matched L1 muon vs. pT" + thisDocumentation}])
    trigvalmuon(dqmitems, thisPath + "/Efficiency L1 vs. Gen",
        [{'path': thisDir + "/genEffEta_L1", 'description':"Efficiency to find an L1 muon associated to a generated muon vs. eta" + thisDocumentation}])
    trigvalmuon(dqmitems, thisPath + "/Efficiency L2 vs. Gen",
        [{'path': thisDir + "/genEffEta_L2", 'description':"Efficiency to find a gen-matched L2 muon associated to a gen-matched L1 muon vs. eta" + thisDocumentation}])
    trigvalmuon(dqmitems, thisPath + "/Efficiency L3 vs. Gen",
        [{'path': thisDir + "/genEffEta_L3", 'description':"Efficiency to find a gen-matched L3 muon associated to a gen-matched L1 muon vs. eta" + thisDocumentation}])
    if "Iso" in thisPath:
        trigvalmuon(dqmitems, thisPath + "/pT Turn-On L2 Isolated vs. Gen",
            [{'path': thisDir + "/genTurnOn1_L2Iso", 'description':"Efficiency to find an isolated gen-matched L2 muon associated to a gen-matched L1 muon vs. pT" + thisDocumentation}])
        trigvalmuon(dqmitems, thisPath + "/pT Turn-On L3 Isolated vs. Gen",
            [{'path': thisDir + "/genTurnOn1_L3Iso", 'description':"Efficiency to find an isolated gen-matched L3 muon associated to a gen-matched L1 muon vs. pT" + thisDocumentation}])
        trigvalmuon(dqmitems, thisPath + "/Efficiency L2 Isolated vs. Gen",
            [{'path': thisDir + "/genEffEta_L2Iso", 'description':"Efficiency to find an isolated gen-matched L2 muon associated to a gen-matched L1 muon vs. eta" + thisDocumentation}])
        trigvalmuon(dqmitems, thisPath + "/Efficiency L3 Isolated vs. Gen",
            [{'path': thisDir + "/genEffEta_L3Iso", 'description':"Efficiency to find an isolated gen-matched L3 muon associated to a gen-matched L1 muon vs. eta" + thisDocumentation}])

###---- TAU selection goes here: ----
def trigvaltau(dqmitems,layout, *plots):
    destination_1='HLT/TauRelVal/Summary For '
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
###

for lumi in ["8E29","1E31"]:
    trigvaltau(dqmitems,"MC_"+lumi+" Menu/Double Tau Path Performance",
               [{'path': "HLT/TauRelVal/MC_"+lumi+ "/DoubleTau/EfficiencyRefInput",
                 'description':"Efficiency of the Double Tau Path with ref to MC for "+lumi},
                {'path': "HLT/TauRelVal/MC_"+lumi+ "/DoubleTau/EfficiencyRefPrevious",
                 'description':"Efficiency of the Double Tau Path with ref to previous step( "+lumi+")"}

               ])
    trigvaltau(dqmitems,"MC_"+lumi+" Menu/Single Tau Path Performance",
               [
                {'path': "HLT/TauRelVal/MC_"+lumi+ "/SingleTau/EfficiencyRefInput",
                 'description':"Efficiency of the Single Tau Path with ref to MC for "+lumi},
                {'path': "HLT/TauRelVal/MC_"+lumi+ "/SingleTau/EfficiencyRefPrevious",
                 'description':"Efficiency of the Single Tau Path with ref to previous step( "+lumi+")"}
               ])
    trigvaltau(dqmitems,"MC_"+lumi+" Menu/L1 Efficency",
               [
                  {'path': "HLT/TauRelVal/MC_"+lumi+ "/L1/L1TauEtEff", 'description':"L1 Tau Efficiency vs pt with  ref to MC for "+lumi},
                  {'path': "HLT/TauRelVal/MC_"+lumi+ "/L1/L1TauEtaEff", 'description':"L1 Tau Efficiency vs pt with  ref to MC for "+lumi},
               ])

    trigvaltau(dqmitems,"MC_"+lumi+" Menu/L2 Efficency",
               [
                  {'path': "HLT/TauRelVal/MC_"+lumi+ "/L2/L2TauEtEff", 'description':"L2 Tau Efficiency vs pt with  ref to MC for "+lumi},
                  {'path': "HLT/TauRelVal/MC_"+lumi+ "/L2/L2TauEtaEff", 'description':"L2 Tau Efficiency vs pt with  ref to MC for "+lumi},
               ])

    trigvaltau(dqmitems,"MC_"+lumi+" Menu/L1 Resolution",
               [
                  {'path': "HLT/TauRelVal/MC_"+lumi+ "/L1/L1TauEtResol", 'description':"L1 Tau ET resolution with ref to MC  for "+lumi}
               ])

    trigvaltau(dqmitems,"MC_"+lumi+" Menu/L2 Resolution",
               [
                  {'path': "HLT/TauRelVal/MC_"+lumi+ "/L2/L2TauEtResol", 'description':"L2 Tau ET resolution with ref to MC  for "+lumi}
               ])

###---- JETMET selection goes here: ----
def trigvaljetmet(dqmitems,layout, *plots):
    destination_1='HLT/HLTJETMET/ValidationReport/'
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
###

trigvaljetmet(dqmitems,"HLTMET35 eff vs genMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET35/Gen Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET35 versus genMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTMET45 eff vs genMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET45/Gen Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET45 versus genMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTMET60 eff vs genMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET60/Gen Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET60 versus genMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTMET100 eff vs genMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET100/Gen Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET100 versus genMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTMET35 eff vs recMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET35/Reco Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET35 versus recMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTMET45 eff vs recMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET45/Reco Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET45 versus recMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTMET60 eff vs recMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET60/Reco Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET60 versus recMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTMET100 eff vs recMet RelVal",
        [{'path': "HLT/HLTJETMET/SingleMET100/Reco Missing ET Turn-On RelVal", 'description': "Trigger efficiency for HLTMET100 versus recMET wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet15U eff vs genJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet15U/Gen Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet15U versus genJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet30U eff vs genJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet30U/Gen Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet30U versus genJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet50U eff vs genJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet50U/Gen Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet50U versus genJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet70U eff vs genJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet70U/Gen Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet70U versus genJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet100U eff vs genJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet100U/Gen Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet100U versus genJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet15U eff vs recJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet15U/Reco Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet15U versus recJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet30U eff vs recJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet30U/Reco Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet30U versus recJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet50U eff vs recJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet50U/Reco Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet50U versus recJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet70U eff vs recJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet70U/Reco Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet70U versus recJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet100U eff vs recJet Pt RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet100U/Reco Jet Pt Turn-On RelVal", 'description': "Trigger efficiency for HLTJet100U versus recJet Pt wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet15U eff vs genJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet15U/Gen Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet15U versus genJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet30U eff vs genJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet30U/Gen Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet30U versus genJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet50U eff vs genJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet50U/Gen Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet50U versus genJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet70U eff vs genJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet70U/Gen Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet70U versus genJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet100U eff vs genJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet100U/Gen Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet100U versus genJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet15U eff vs recJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet15U/Reco Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet15U versus recJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet30U eff vs recJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet30U/Reco Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet30U versus recJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet50U eff vs recJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet50U/Reco Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet50U versus recJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet70U eff vs recJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet70U/Reco Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet70U versus recJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet100U eff vs recJet Eta RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet100U/Reco Jet Eta Turn-On RelVal", 'description': "Trigger efficiency for HLTJet100U versus recJet Eta wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet15U eff vs genJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet15U/Gen Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet15U versus genJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet30U eff vs genJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet30U/Gen Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet30U versus genJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet50U eff vs genJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet50U/Gen Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet50U versus genJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet70U eff vs genJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet70U/Gen Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet70U versus genJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet100U eff vs genJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet100U/Gen Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet100U versus genJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet15U eff vs recJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet15U/Reco Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet15U versus recJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet30U eff vs recJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet30U/Reco Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet30U versus recJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet50U eff vs recJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet50U/Reco Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet50U versus recJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet70U eff vs recJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet70U/Reco Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet70U versus recJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTJet100U eff vs recJet Phi RelVal",
        [{'path': "HLT/HLTJETMET/SingleJet100U/Reco Jet Phi Turn-On RelVal", 'description': "Trigger efficiency for HLTJet100U versus recJet Phi wrt full sample"}])
trigvaljetmet(dqmitems,"HLTHT300MHT100 eff vs genHT RelVal",
        [{'path': "HLT/HLTJETMET/HT300MHT100/Gen HT Turn-On RelVal", 'description': "Trigger efficiency for HLTHT300MHT100 versus genHT wrt full sample"}])
trigvaljetmet(dqmitems,"HLTHT300MHT100 eff vs recHT RelVal",
        [{'path': "HLT/HLTJETMET/HT300MHT100/Reco HT Turn-On RelVal", 'description': "Trigger efficiency for HLTHT300MHT100 versus recHT wrt full sample"}])

###---- BJET selection goes here: ----
#def trigvalbjet(i, p, *rows): i["HLT//Preselection" + p] = DQMItem(layout=rows)

###---- ALCA selection goes here: ----
def trigvalalca(dqmitems,layout, *plots):
    destination_1='HLT/AlCaEcalPi0/Preselection'
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
###

###---- TOP selection goes here: ----
def trigvaltopmuon(dqmitems,layout, *plots):
    destination_1='HLT/Top/TopValidationReport/Semileptonic_muon/'
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
###

trigvaltopmuon(dqmitems,"HLTMu9 eff vs eta",
        [{'path': "HLT/Top/Semileptonic_muon/EffVsEta_HLT_Mu9", 'description': "Trigger efficiency for HLTMu9 versus eta of the highest pt reconstructed muon with pt>20, eta<2.1 "}])
trigvaltopmuon(dqmitems,"HLTMu9 eff vs pt",
        [{'path': "HLT/Top/Semileptonic_muon/EffVsPt_HLT_Mu9", 'description': "Trigger efficiency for HLTMu9 versus pt of the highest pt reconstructed muon with pt>20, eta<2.1"}])

trigvaltopmuon(dqmitems,"HLTMu15 eff vs eta",
        [{'path': "HLT/Top/Semileptonic_muon/EffVsEta_HLT_Mu15", 'description': "Trigger efficiency for HLTMu15 versus eta of the highest pt reconstructed muon with pt>20, eta<2.1"}])
trigvaltopmuon(dqmitems,"HLTMu15 eff vs pt",
        [{'path': "HLT/Top/Semileptonic_muon/EffVsPt_HLT_Mu15", 'description': "Trigger efficiency for HLTMu15 versus pt of the highest pt reconstructed muon with pt>20, eta<2.1"}])

trigvaltopmuon(dqmitems,"Muon trigger efficiencies wrt gen",
        [{'path': "HLT/Top/Semileptonic_muon/Efficiencies_MuonTriggers_gen", 'description': "Muon trigger efficiencies wrt mc acceptance (1 muon from W decay, pt>10, eta<2.4)"}])

trigvaltopmuon(dqmitems,"Muon trigger efficiencies wrt gen+reco",
        [{'path': "HLT/Top/Semileptonic_muon/Efficiencies_MuonTriggers", 'description': "Muon trigger efficiencies wrt mc acceptance+offline  (acc: 1 muon from W, pt>10, eta<2.4; off: at least 1 rec muon, pt>20, eta<2.1 and 2 jets Et_raw>13, eta<2.4)"}])

def trigvaltopelectron(dqmitems,layout, *plots):
    destination_1='HLT/Top/TopValidationReport/Semileptonic_electron/'
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
###

trigvaltopelectron(dqmitems,"HLTEle15SWL1R eff vs eta",
        [{'path': "HLT/Top/Semileptonic_electron/EffVsEta_HLT_Ele15_SW_L1R", 'description': "Trigger efficiency for HLT_Ele15_SW_L1R versus eta of the highest pt reconstructed electron with pt>20, eta<2.4"}])

trigvaltopelectron(dqmitems,"HLTEle15SWL1R eff vs pt",
        [{'path': "HLT/Top/Semileptonic_electron/EffVsPt_HLT_Ele15_SW_L1R", 'description': "Trigger efficiency for HLT_Ele15_SW_L1R versus pt of the highest pt reconstructed electron with pt>20, eta<2.4"}])

trigvaltopelectron(dqmitems,"HLTEle15SWLooseTrackIsoL1R eff vs eta",
        [{'path': "HLT/Top/Semileptonic_electron/EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R", 'description': "Trigger efficiency for HLT_Ele15_SW_LooseTrackIso_L1R versus eta of the highest pt reconstructed electron with pt>20, eta<2.4"}])

trigvaltopelectron(dqmitems,"HLTEle15SWLooseTrackIsoL1R eff vs pt",
        [{'path': "HLT/Top/Semileptonic_electron/EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R", 'description': "Trigger efficiency for HLTEle15_SW_LooseTrackIso_L1R versus pt of the highest pt reconstructed electron with pt>20, eta<2.4"}])

trigvaltopelectron(dqmitems,"Electron trigger efficiencies wrt gen",
        [{'path': "HLT/Top/Semileptonic_electron/Efficiencies_Electrontriggers_gen", 'description': "Electron trigger efficiencies wrt mc acceptance  (1 electron from W decay, pt>10, eta<2.4)"}])

trigvaltopelectron(dqmitems,"Electron trigger efficiencies wrt gen+reco",
        [{'path': "HLT/Top/Semileptonic_electron/Efficiencies_Electrontriggers", 'description': "Electron trigger efficiencies wrt mc acceptance+offline  (acc: 1 electron from W, pt>10, eta<2.4; off: at least 1 rec electron, pt>20, eta<2.4 and 2 jets Et_raw>13, eta<2.4)"}])
##---- HEAVYFLAVOR selection goes here: ----
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# def trigvalbphys(items, title, histogram, description):
#   global bphysPlotNumber
#   DQMItem(layout=[[{'path':"HLT/HeavyFlavor/HLT/"+histogram, 'description':description}]], destination0="HLT/HeavyFlavor/HLTValidationReport/" + '%02d) '%bphysPlotNumber + title)
#   print("HLT/HeavyFlavor/HLTValidationReport/" + '%02d) '%bphysPlotNumber + title)
#   # print(str(items["HLT/HeavyFlavor/HLTValidationReport/" + '%02d) '%bphysPlotNumber + title]))
# #  items["HLT/HeavyFlavor/HLT2ValidationReport/" + '%02d) '%bphysPlotNumber + title] = DQMItem(layout=[[{'path':"HLT/HeavyFlavor/HLT2/"+histogram, 'description':description}]])
#   bphysPlotNumber+=1

# # SUMMARY PLOT

# trigvalbphys(dqmitems,
#   "Trigger Efficiencies in Di-global Events",
#   "effPathGlob_recoPt",
#   "Trigger path efficiencies in di-global muon events where the muons are associated to the generated muons as a function of di-muon pT"
# )

# # SINGLE MUON

# trigvalbphys(dqmitems,
#   "Glob\Gen Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effGlobGen_genEtaPhi",
#   "Efficiency to find a global muon associated to a generated muon as a function of generated muon Eta and Phi"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effGlobGen_genEtaPt",
#   "Efficiency to find a global muon associated to a generated muon as a function of generated muon Eta and pT"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effGlobGen_genEtaPtX",
#   "Efficiency to find a global muon associated to a generated muon as a function of generated muon Eta"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effGlobGen_genEtaPtY",
#   "Efficiency to find a global muon associated to a generated muon as a function of generated muon pT"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effGlobGen_genEtaPhiY",
#   "Efficiency to find a global muon associated to a generated muon as a function of generated muon Phi"
# )

# trigvalbphys(dqmitems,
#   "L1\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt1Glob_recoEtaPhi",
#   "Efficiency to find a L1 muon associated to a global+gen muon as a function of global muon Eta and Phi"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt1Glob_recoEtaPt",
#   "Efficiency to find a L1 muon associated to a global+gen muon as a function of global muon Eta and pT"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt1Glob_recoEtaPtX",
#   "Efficiency to find a L1 muon associated to a global+gen muon as a function of global muon Eta"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt1Glob_recoEtaPtY",
#   "Efficiency to find a L1 muon associated to a global+gen muon as a function of global muon pT"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt1Glob_recoEtaPhiY",
#   "Efficiency to find a L1 muon associated to a global+gen muon as a function of global muon Phi"
# )

# trigvalbphys(dqmitems,
#   "L2\L1 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt2Filt1_recoEtaPhi",
#   "Efficiency to find a L2 muon associated to a L1+global+gen muon as a function of global muon Eta and Phi"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt2Filt1_recoEtaPt",
#   "Efficiency to find a L2 muon associated to a L1+global+gen muon as a function of global muon Eta and pT"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt2Filt1_recoEtaPtX",
#   "Efficiency to find a L2 muon associated to a L1+global+gen muon as a function of global muon Eta"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt2Filt1_recoEtaPtY",
#   "Efficiency to find a L2 muon associated to a L1+global+gen muon as a function of global muon pT"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt2Filt1_recoEtaPhiY",
#   "Efficiency to find a L2 muon associated to a L1+global+gen muon as a function of global muon Phi"
# )

# trigvalbphys(dqmitems,
#   "L3\L2 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Filt2_recoEtaPhi",
#   "Efficiency to find a L3 muon associated to a L2+L1+global+gen muon as a function of global muon Eta and Phi"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Filt2_recoEtaPt",
#   "Efficiency to find a L3 muon associated to a L2+L1+global+gen muon as a function of global muon Eta and pT"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Filt2_recoEtaPtX",
#   "Efficiency to find a L3 muon associated to a L2+L1+global+gen muon as a function of global muon Eta"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Filt2_recoEtaPtY",
#   "Efficiency to find a L3 muon associated to a L2+L1+global+gen muon as a function of global muon pT"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Filt2_recoEtaPhiY",
#   "Efficiency to find a L3 muon associated to a L2+L1+global+gen muon as a function of global muon Phi"
# )

# trigvalbphys(dqmitems,
#   "L3\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Glob_recoEtaPhi",
#   "Efficiency to find a L3 muon associated to a global+gen muon as a function of global muon Eta and Phi"
# )
# trigvalbphys(dqmitems,
#   "L3\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Glob_recoEtaPt",
#   "Efficiency to find a L3 muon associated to a global+gen muon as a function of global muon Eta and pT"
# )
# trigvalbphys(dqmitems,
#   "L3\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Glob_recoEtaPtX",
#   "Efficiency to find a L3 muon associated to a global+gen muon as a function of global muon Eta"
# )
# trigvalbphys(dqmitems,
#   "L3\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Glob_recoEtaPtY",
#   "Efficiency to find a L3 muon associated to a global+gen muon as a function of global muon pT"
# )
# trigvalbphys(dqmitems,
#   "L3\Glob Single Muon Efficiency (HLT_Mu3)",
#   "HLT_Mu3/effFilt3Glob_recoEtaPhiY",
#   "Efficiency to find a L3 muon associated to a global+gen muon as a function of global muon Phi"
# )

# # DOUBLE MUON

# trigvalbphys(dqmitems,
#   "Glob\Gen Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effGlobDigenAND_genRapPt",
#   "Efficiency to find two global muons associated to two generated muons as a function of generated dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effGlobDigenAND_genRapPtX",
#   "Efficiency to find two global muons associated to two generated muons as a function of generated dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effGlobDigenAND_genRapPtY",
#   "Efficiency to find two global muons associated to two generated muons as a function of generated dimuon pT"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effGlobDigenAND_genPtDR",
#   "Efficiency to find two global muons associated to two generated muons (with pT>7) as a function of generated dimuon pT and DeltaR at Interaction Point"
# )
# trigvalbphys(dqmitems,
#   "Glob\Gen Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effGlobDigenAND_genPtDRY",
#   "Efficiency to find two global muons associated to two generated muons (with pT>7) as a function of generated dimuon DeltaR at Interaction Point"
# )

# trigvalbphys(dqmitems,
#   "L1\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt1DiglobAND_recoRapPt",
#   "Efficiency to find two L1 muons associated to two global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt1DiglobAND_recoRapPtX",
#   "Efficiency to find two L1 dimuon associated to two global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt1DiglobAND_recoRapPtY",
#   "Efficiency to find two L1 muons associated to two global+gen muons as a function of global dimuon pT"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt1DiglobAND_recoPtDRpos",
#   "Efficiency to find two L1 muons associated to two global+gen muons (with pT>7) as a function of global dimuon pT and DeltaR in the Muon System"
# )
# trigvalbphys(dqmitems,
#   "L1\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt1DiglobAND_recoPtDRposY",
#   "Efficiency to find two L1 muons associated to two global+gen muons (with pT>7) as a function of global dimuon deltaR in the Muon System"
# )

# trigvalbphys(dqmitems,
#   "L2\L1 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt2Difilt1AND_recoRapPt",
#   "Efficiency to find two L2 muons associated to two L1+global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt2Difilt1AND_recoRapPtX",
#   "Efficiency to find two L2 muons associated to two L1+global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt2Difilt1AND_recoRapPtY",
#   "Efficiency to find two L2 muons associated to two L1+global+gen muons as a function of global dimuon pT"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt2Difilt1AND_recoPtDRpos",
#   "Efficiency to find two L2 muons associated to two L1+global+gen muons (with pT>7) as a function of global dimuon pT and DeltaR in the Muon System"
# )
# trigvalbphys(dqmitems,
#   "L2\L1 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt2Difilt1AND_recoPtDRposY",
#   "Efficiency to find two L2 muons associated to two L1+global+gen muons (with pT>7) as a function of global dimuon DeltaR in the Muon System"
# )

# trigvalbphys(dqmitems,
#   "L3\L2 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3Difilt2AND_recoRapPt",
#   "Efficiency to find two L3 muons associated to two L2+L1+global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3Difilt2AND_recoRapPtX",
#   "Efficiency to find two L3 muons associated to two L2+L1+global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3Difilt2AND_recoRapPtY",
#   "Efficiency to find two L3 muons associated to two L2+L1+global+gen muons as a function of global dimuon pT"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3Difilt2AND_recoPtDR",
#   "Efficiency to find two L3 muons associated to two L2+L1+global+gen muons (with pT>7) as a function of global dimuon pT and DeltaR at Interaction Point"
# )
# trigvalbphys(dqmitems,
#   "L3\L2 Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3Difilt2AND_recoPtDRY",
#   "Efficiency to find two L3 muons associated to two L2+L1+global+gen muons (with pT>7) as a function of global dimuon DeltaR at Interaction Point"
# )

# trigvalbphys(dqmitems,
#   "L3\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3DiglobAND_recoRapPt",
#   "Efficiency to find two L3 muons associated to two global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "L3\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3DiglobAND_recoRapPtX",
#   "Efficiency to find two L3 muons associated to two global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "L3\Glob Dimuon Efficiency (HLT_DoubleMu0)",
#   "HLT_DoubleMu0/effFilt3DiglobAND_recoRapPtY",
#   "Efficiency to find two L3 muons associated to two global+gen muons as a function of global dimuon pT"
# )

# # TRIGGER PATH

# trigvalbphys(dqmitems,
#   "HLT_Mu3 Efficiency in Di-Global Events",
#   "HLT_Mu3/effPathDiglobOR_recoRapPt",
#   "Efficiency to find an HLT_Mu3 muon associated to one of the global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "HLT_Mu3 Efficiency in Di-Global Events",
#   "HLT_Mu3/effPathDiglobOR_recoRapPtX",
#   "Efficiency to find an HLT_Mu3 muon associated to one of the global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "HLT_Mu3 Efficiency in Di-Global Events",
#   "HLT_Mu3/effPathDiglobOR_recoRapPtY",
#   "Efficiency to find an HLT_Mu3 muon associated to one of the global+gen muons as a function of global dimuon pT"
# )

# trigvalbphys(dqmitems,
#   "HLT_Mu5 Efficiency in Di-Global Events",
#   "HLT_Mu5/effPathDiglobOR_recoRapPt",
#   "Efficiency to find an HLT_Mu5 muon associated to one of the global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "HLT_Mu5 Efficiency in Di-Global Events",
#   "HLT_Mu5/effPathDiglobOR_recoRapPtX",
#   "Efficiency to find an HLT_Mu5 muon associated to one of the global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "HLT_Mu5 Efficiency in Di-Global Events",
#   "HLT_Mu5/effPathDiglobOR_recoRapPtY",
#   "Efficiency to find an HLT_Mu5 muon associated to one of the global+gen muons as a function of global dimuon pT"
# )

# trigvalbphys(dqmitems,
#   "HLT_Mu9 Efficiency in Di-Global Events",
#   "HLT_Mu9/effPathDiglobOR_recoRapPt",
#   "Efficiency to find an HLT_Mu9 muon associated to one of the global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "HLT_Mu9 Efficiency in Di-Global Events",
#   "HLT_Mu9/effPathDiglobOR_recoRapPtX",
#   "Efficiency to find an HLT_Mu9 muon associated to one of the global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "HLT_Mu9 Efficiency in Di-Global Events",
#   "HLT_Mu9/effPathDiglobOR_recoRapPtY",
#   "Efficiency to find an HLT_Mu9 muon associated to one of the global+gen muons as a function of global dimuon pT"
# )

# trigvalbphys(dqmitems,
#   "HLT_DoubleMu0 Efficiency in Di-Global Events",
#   "HLT_DoubleMu0/effPathDiglobAND_recoRapPt",
#   "Efficiency to find two HLT_DoubleMu0 muons associated to two global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "HLT_DoubleMu0 Efficiency in Di-Global Events",
#   "HLT_DoubleMu0/effPathDiglobAND_recoRapPtX",
#   "Efficiency to find two HLT_DoubleMu0 muons associated to two global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "HLT_DoubleMu0 Efficiency in Di-Global Events",
#   "HLT_DoubleMu0/effPathDiglobAND_recoRapPtY",
#   "Efficiency to find two HLT_DoubleMu0 muons associated to two global+gen muons as a function of global dimuon pT"
# )

# trigvalbphys(dqmitems,
#   "HLT_DoubleMu3 Efficiency in Di-Global Events",
#   "HLT_DoubleMu3/effPathDiglobAND_recoRapPt",
#   "Efficiency to find two HLT_DoubleMu3 muons associated to two global+gen muons as a function of global dimuon Rapidity and pT"
# )
# trigvalbphys(dqmitems,
#   "HLT_DoubleMu3 Efficiency in Di-Global Events",
#   "HLT_DoubleMu3/effPathDiglobAND_recoRapPtX",
#   "Efficiency to find two HLT_DoubleMu3 muons associated to two global+gen muons as a function of global dimuon Rapidity"
# )
# trigvalbphys(dqmitems,
#   "HLT_DoubleMu3 Efficiency in Di-Global Events",
#   "HLT_DoubleMu3/effPathDiglobAND_recoRapPtY",
#   "Efficiency to find two HLT_DoubleMu3 muons associated to two global+gen muons as a function of global dimuon pT"
# )
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
###---- SUSYEXO selection goes here: ----
def trigvalsusybsm(dqmitems,layout, *plots):
    destination_1='HLT/SusyExo/00 SusyExoValidationReport/00 Global Efficiencies/'
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
############# Mc Selections ################

### RA1

# L1
trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA1/L1_RA1_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA1/L1_RA1_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA1/L1_RA1_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA1/L1_RA1_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA1/L1_RA1_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA1/L1_RA1_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA1/L1_RA1_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/00 RA1/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA1/Hlt_RA1_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA2

# L1
trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA2/L1_RA2_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA2/L1_RA2_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA2/L1_RA2_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA2/L1_RA2_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA2/L1_RA2_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA2/L1_RA2_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA2/L1_RA2_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/01 RA2/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA2/Hlt_RA2_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA3

# L1
trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA3/L1_RA3_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA3/L1_RA3_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA3/L1_RA3_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA3/L1_RA3_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA3/L1_RA3_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA3/L1_RA3_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA3/L1_RA3_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/02 RA3/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA3/Hlt_RA3_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA4_e

# L1
trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/L1_RA4_e_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/L1_RA4_e_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/L1_RA4_e_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/L1_RA4_e_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/L1_RA4_e_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/L1_RA4_e_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/L1_RA4_e_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/03 RA4_e/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA4_e/Hlt_RA4_e_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA4_m

# L1
trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/L1_RA4_m_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/L1_RA4_m_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/L1_RA4_m_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/L1_RA4_m_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/L1_RA4_m_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/L1_RA4_m_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/L1_RA4_m_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/04 RA4_m/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA4_m/Hlt_RA4_m_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA5RA6_2e

# L1
trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/L1_RA5RA6_2e_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/L1_RA5RA6_2e_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/L1_RA5RA6_2e_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/L1_RA5RA6_2e_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/L1_RA5RA6_2e_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/L1_RA5RA6_2e_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/L1_RA5RA6_2e_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/05 RA5RA6_2e/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2e/Hlt_RA5RA6_2e_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA5RA6_1e1m

# L1
trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/L1_RA5RA6_1e1m_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/L1_RA5RA6_1e1m_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/L1_RA5RA6_1e1m_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/L1_RA5RA6_1e1m_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/L1_RA5RA6_1e1m_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/L1_RA5RA6_1e1m_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/L1_RA5RA6_1e1m_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/06 RA5RA6_1e1m/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_1e1m/Hlt_RA5RA6_1e1m_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA5RA6_2m

# L1
trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/L1_RA5RA6_2m_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/L1_RA5RA6_2m_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/L1_RA5RA6_2m_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/L1_RA5RA6_2m_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/L1_RA5RA6_2m_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/L1_RA5RA6_2m_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/L1_RA5RA6_2m_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/07 RA5RA6_2m/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA5RA6_2m/Hlt_RA5RA6_2m_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA7_3e

# L1
trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/L1_RA7_3e_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/L1_RA7_3e_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/L1_RA7_3e_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/L1_RA7_3e_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/L1_RA7_3e_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/L1_RA7_3e_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/L1_RA7_3e_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/08 RA7_3e/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_3e/Hlt_RA7_3e_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA7_2e1m

# L1
trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/L1_RA7_2e1m_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/L1_RA7_2e1m_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/L1_RA7_2e1m_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/L1_RA7_2e1m_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/L1_RA7_2e1m_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/L1_RA7_2e1m_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/L1_RA7_2e1m_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/09 RA7_2e1m/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_2e1m/Hlt_RA7_2e1m_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA7_1e2m

# L1
trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/L1_RA7_1e2m_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/L1_RA7_1e2m_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/L1_RA7_1e2m_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/L1_RA7_1e2m_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/L1_RA7_1e2m_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/L1_RA7_1e2m_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/L1_RA7_1e2m_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/10 RA7_1e2m/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_1e2m/Hlt_RA7_1e2m_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### RA7_3m

# L1
trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/00 L1/00 L1_EG",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/L1_RA7_3m_EG", 'description': "Efficiency for L1 e-gamma bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/00 L1/01 L1_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/L1_RA7_3m_Mu", 'description': "Efficiency for L1 muon bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/00 L1/02 L1_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/L1_RA7_3m_Jet", 'description': "Efficiency for L1 jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/00 L1/03 L1_ETM_ETT_HTT",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/L1_RA7_3m_ETM_ETT_HTT", 'description': "Efficiency for L1 ETM, ETT  and HTT bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/00 L1/04 L1_TauJet",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/L1_RA7_3m_TauJet", 'description': "Efficiency for L1 tau jet bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/00 L1/05 L1_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/L1_RA7_3m_XTrigger", 'description': "Efficiency for L1 cross trigger bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/00 L1/06 L1_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/L1_RA7_3m_Overflow", 'description': "Efficiency for other L1 bits. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

# HLT
trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/00 Hlt_Ele",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_Ele", 'description': "Efficiency for HLT electron paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/01 Hlt_Photon",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_Photon", 'description': "Efficiency for HLT photon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/02 Hlt_Mu",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_Mu", 'description': "Efficiency for HLT muon paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/03 Hlt_Jet",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_Jet", 'description': "Efficiency for HLT jet paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/04 Hlt_MET_HT",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_MET_HT", 'description': "Efficiency for HLT MET and HT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/05 Hlt_Tau_BTag",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_Tau_BTag", 'description': "Efficiency for HLT tau and b-tag paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/06 Hlt_XTrigger",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_XTrigger", 'description': "Efficiency for HLT cross trigger paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,"01 McSelection/11 RA7_3m/01 HLT/07 Hlt_Others",
               [{'path': "HLT/SusyExo/McSelection/RA7_3m/Hlt_RA7_3m_Overflow", 'description': "Efficiency for other HLT paths. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

############# Reco Distributions
def trigvalsusybsm(dqmitems,layout, *plots):
    destination_1='HLT/SusyExo/00 SusyExoValidationReport/01 Reco Distributions/'
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
########General

### Reco Electrons
trigvalsusybsm(dqmitems,
               "00 General/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/General/ElecMult",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/General/Elec1Pt",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/General/Elec2Pt",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/General/Elec1Eta",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/General/Elec2Eta",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/General/Elec1Phi",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/General/Elec2Phi",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "00 General/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/General/MuonMult",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/General/Muon1Pt",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/General/Muon2Pt",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/General/Muon1Eta",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/General/Muon2Eta",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/General/Muon1Phi",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/General/Muon2Phi",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "00 General/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/General/JetMult",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/General/Jet1Pt",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/General/Jet2Pt",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/General/Jet1Eta",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/General/Jet2Eta",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/General/Jet1Phi",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/General/Jet2Phi",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "00 General/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/General/PhotonMult",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/General/Photon1Pt",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/General/Photon2Pt",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/General/Photon1Eta",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/General/Photon2Eta",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/General/Photon1Phi",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/General/Photon2Phi",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "00 General/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/General/MET",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/General/METphi",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/General/METx",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/General/METy",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/General/METSignificance",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "00 General/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/General/SumEt",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Jet110

### Reco Electrons
trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Jet110",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Jet110",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Jet110",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Jet110",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Jet110",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Jet110",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Jet110",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Jet110",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Jet110",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Jet110",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Jet110",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Jet110",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Jet110",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Jet110",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Jet110",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Jet110",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Jet110",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Jet110",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Jet110",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Jet110",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Jet110",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Jet110",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Jet110",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Jet110",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Jet110",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Jet110",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Jet110",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Jet110",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Jet110",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Jet110",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Jet110",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Jet110",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Jet110",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "01 HLT_Jet110/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Jet110",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Jet140

### Reco Electrons
trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Jet140",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Jet140",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Jet140",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Jet140",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Jet140",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Jet140",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Jet140",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Jet140",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Jet140",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Jet140",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Jet140",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Jet140",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Jet140",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Jet140",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Jet140",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Jet140",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Jet140",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Jet140",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Jet140",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Jet140",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Jet140",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Jet140",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Jet140",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Jet140",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Jet140",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Jet140",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Jet140",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Jet140",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Jet140",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Jet140",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Jet140",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Jet140",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Jet140",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "02 HLT_Jet140/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Jet140",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_MET60

### Reco Electrons
trigvalsusybsm(dqmitems,
               "03 HLT_MET60/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_MET60",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_MET60",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_MET60",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_MET60",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_MET60",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_MET60",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_MET60",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "03 HLT_MET60/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_MET60",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_MET60",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_MET60",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_MET60",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_MET60",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_MET60",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_MET60",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "03 HLT_MET60/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_MET60",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_MET60",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_MET60",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_MET60",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_MET60",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_MET60",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_MET60",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "03 HLT_MET60/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_MET60",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_MET60",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_MET60",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_MET60",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_MET60",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_MET60",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_MET60",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "03 HLT_MET60/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_MET60",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_MET60",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_MET60",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_MET60",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_MET60",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "03 HLT_MET60/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_MET60",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_MET100

### Reco Electrons
trigvalsusybsm(dqmitems,
               "04 HLT_MET100/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_MET100",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_MET100",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_MET100",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_MET100",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_MET100",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_MET100",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_MET100",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "04 HLT_MET100/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_MET100",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_MET100",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_MET100",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_MET100",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_MET100",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_MET100",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_MET100",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "04 HLT_MET100/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_MET100",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_MET100",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_MET100",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_MET100",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_MET100",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_MET100",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_MET100",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "04 HLT_MET100/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_MET100",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_MET100",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_MET100",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_MET100",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_MET100",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_MET100",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_MET100",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "04 HLT_MET100/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_MET100",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_MET100",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_MET100",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_MET100",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_MET100",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "04 HLT_MET100/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_MET100",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_HT200

### Reco Electrons
trigvalsusybsm(dqmitems,
               "05 HLT_HT200/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_HT200",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_HT200",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_HT200",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_HT200",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_HT200",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_HT200",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_HT200",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "05 HLT_HT200/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_HT200",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_HT200",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_HT200",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_HT200",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_HT200",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_HT200",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_HT200",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "05 HLT_HT200/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_HT200",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_HT200",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_HT200",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_HT200",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_HT200",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_HT200",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_HT200",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "05 HLT_HT200/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_HT200",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_HT200",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_HT200",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_HT200",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_HT200",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_HT200",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_HT200",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "05 HLT_HT200/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_HT200",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_HT200",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_HT200",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_HT200",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_HT200",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "05 HLT_HT200/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_HT200",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_HT240

### Reco Electrons
trigvalsusybsm(dqmitems,
               "06 HLT_HT240/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_HT240",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_HT240",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_HT240",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_HT240",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_HT240",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_HT240",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_HT240",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "06 HLT_HT240/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_HT240",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_HT240",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_HT240",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_HT240",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_HT240",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_HT240",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_HT240",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "06 HLT_HT240/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_HT240",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_HT240",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_HT240",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_HT240",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_HT240",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_HT240",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_HT240",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "06 HLT_HT240/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_HT240",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_HT240",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_HT240",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_HT240",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_HT240",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_HT240",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_HT240",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "06 HLT_HT240/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_HT240",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_HT240",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_HT240",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_HT240",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_HT240",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "06 HLT_HT240/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_HT240",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_HT300_MHT100

### Reco Electrons
trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_HT300_MHT100",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_HT300_MHT100",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_HT300_MHT100",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_HT300_MHT100",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_HT300_MHT100",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_HT300_MHT100",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_HT300_MHT100",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_HT300_MHT100",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_HT300_MHT100",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_HT300_MHT100",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_HT300_MHT100",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_HT300_MHT100",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "07 HLT_HT300_MHT100/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_HT300_MHT100",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Mu9

### Reco Electrons
trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Mu9",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Mu9",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Mu9",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Mu9",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Mu9",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Mu9",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Mu9",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Mu9",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Mu9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Mu9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Mu9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Mu9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Mu9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Mu9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Mu9",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Mu9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Mu9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Mu9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Mu9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Mu9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Mu9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Mu9",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Mu9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Mu9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Mu9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Mu9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Mu9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Mu9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Mu9",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Mu9",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Mu9",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Mu9",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Mu9",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "08 HLT_Mu9/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Mu9",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Mu11

### Reco Electrons
trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Mu11",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Mu11",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Mu11",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Mu11",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Mu11",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Mu11",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Mu11",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Mu11",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Mu11",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Mu11",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Mu11",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Mu11",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Mu11",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Mu11",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Mu11",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Mu11",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Mu11",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Mu11",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Mu11",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Mu11",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Mu11",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Mu11",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Mu11",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Mu11",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Mu11",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Mu11",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Mu11",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Mu11",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Mu11",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Mu11",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Mu11",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Mu11",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Mu11",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "09 HLT_Mu11/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Mu11",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_IsoMu9

### Reco Electrons
trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_IsoMu9",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_IsoMu9",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_IsoMu9",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_IsoMu9",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_IsoMu9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_IsoMu9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_IsoMu9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_IsoMu9",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_IsoMu9",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_IsoMu9",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_IsoMu9",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_IsoMu9",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "10 HLT_IsoMu9/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_IsoMu9",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_DoubleMu3

### Reco Electrons
trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_DoubleMu3",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_DoubleMu3",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_DoubleMu3",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_DoubleMu3",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_DoubleMu3",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_DoubleMu3",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_DoubleMu3",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_DoubleMu3",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_DoubleMu3",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_DoubleMu3",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_DoubleMu3",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_DoubleMu3",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "11 HLT_DoubleMu3/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_DoubleMu3",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Ele15_SW_LooseTrackIso_L1R

### Reco Electrons
trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "12 HLT_Ele15_SW_LooseTrackIso_L1R/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Ele15_SW_LooseTrackIso_L1R",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Ele20_SW_L1R

### Reco Electrons
trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Ele20_SW_L1R",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Ele20_SW_L1R",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Ele20_SW_L1R",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Ele20_SW_L1R",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Ele20_SW_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Ele20_SW_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Ele20_SW_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Ele20_SW_L1R",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Ele20_SW_L1R",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Ele20_SW_L1R",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Ele20_SW_L1R",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Ele20_SW_L1R",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "13 HLT_Ele20_SW_L1R/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Ele20_SW_L1R",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_DoubleEle10_SW_L1R

### Reco Electrons
trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_DoubleEle10_SW_L1R",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_DoubleEle10_SW_L1R",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_DoubleEle10_SW_L1R",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_DoubleEle10_SW_L1R",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_DoubleEle10_SW_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_DoubleEle10_SW_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_DoubleEle10_SW_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_DoubleEle10_SW_L1R",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_DoubleEle10_SW_L1R",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_DoubleEle10_SW_L1R",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_DoubleEle10_SW_L1R",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_DoubleEle10_SW_L1R",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "14 HLT_DoubleEle10_SW_L1R/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_DoubleEle10_SW_L1R",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Photon25_L1R

### Reco Electrons
trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Photon25_L1R",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Photon25_L1R",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Photon25_L1R",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Photon25_L1R",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Photon25_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Photon25_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Photon25_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Photon25_L1R",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Photon25_L1R",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Photon25_L1R",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Photon25_L1R",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Photon25_L1R",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "15 HLT_Photon25_L1R/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Photon25_L1R",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Photon30_L1R_1E31

### Reco Electrons
trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Photon30_L1R_1E31",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Photon30_L1R_1E31",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Photon30_L1R_1E31",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Photon30_L1R_1E31",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Photon30_L1R_1E31",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Photon30_L1R_1E31",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Photon30_L1R_1E31",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Photon30_L1R_1E31",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Photon30_L1R_1E31",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Photon30_L1R_1E31",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Photon30_L1R_1E31",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Photon30_L1R_1E31",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "16 HLT_Photon30_L1R_1E31/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Photon30_L1R_1E31",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_DoublePhoton15_L1R

### Reco Electrons
trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_DoublePhoton15_L1R",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_DoublePhoton15_L1R",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_DoublePhoton15_L1R",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_DoublePhoton15_L1R",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_DoublePhoton15_L1R",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_DoublePhoton15_L1R",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_DoublePhoton15_L1R",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_DoublePhoton15_L1R",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_DoublePhoton15_L1R",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_DoublePhoton15_L1R",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_DoublePhoton15_L1R",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_DoublePhoton15_L1R",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "17 HLT_DoublePhoton15_L1R/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_DoublePhoton15_L1R",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_BTagIP_Jet80

### Reco Electrons
trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_BTagIP_Jet80",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_BTagIP_Jet80",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_BTagIP_Jet80",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_BTagIP_Jet80",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_BTagIP_Jet80",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_BTagIP_Jet80",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_BTagIP_Jet80",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_BTagIP_Jet80",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_BTagIP_Jet80",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_BTagIP_Jet80",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_BTagIP_Jet80",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_BTagIP_Jet80",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "18 HLT_BTagIP_Jet80/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_BTagIP_Jet80",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_BTagIP_Jet120

### Reco Electrons
trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_BTagIP_Jet120",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_BTagIP_Jet120",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_BTagIP_Jet120",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_BTagIP_Jet120",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_BTagIP_Jet120",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_BTagIP_Jet120",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_BTagIP_Jet120",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_BTagIP_Jet120",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_BTagIP_Jet120",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_BTagIP_Jet120",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_BTagIP_Jet120",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_BTagIP_Jet120",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "19 HLT_BTagIP_Jet120/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_BTagIP_Jet120",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_L2Mu5_Photon9

### Reco Electrons
trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_L2Mu5_Photon9",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_L2Mu5_Photon9",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_L2Mu5_Photon9",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_L2Mu5_Photon9",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_L2Mu5_Photon9",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_L2Mu5_Photon9",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_L2Mu5_Photon9",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_L2Mu5_Photon9",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_L2Mu5_Photon9",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_L2Mu5_Photon9",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_L2Mu5_Photon9",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_L2Mu5_Photon9",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "20 HLT_L2Mu5_Photon9/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_L2Mu5_Photon9",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_L2Mu8_HT50

### Reco Electrons
trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_L2Mu8_HT50",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_L2Mu8_HT50",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_L2Mu8_HT50",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_L2Mu8_HT50",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_L2Mu8_HT50",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_L2Mu8_HT50",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_L2Mu8_HT50",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_L2Mu8_HT50",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_L2Mu8_HT50",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_L2Mu8_HT50",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_L2Mu8_HT50",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_L2Mu8_HT50",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "21 HLT_L2Mu8_HT50/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_L2Mu8_HT50",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Ele10_LW_L1R_HT180

### Reco Electrons
trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Ele10_LW_L1R_HT180",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Ele10_LW_L1R_HT180",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Ele10_LW_L1R_HT180",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Ele10_LW_L1R_HT180",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Ele10_LW_L1R_HT180",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Ele10_LW_L1R_HT180",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Ele10_LW_L1R_HT180",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Ele10_LW_L1R_HT180",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Ele10_LW_L1R_HT180",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Ele10_LW_L1R_HT180",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Ele10_LW_L1R_HT180",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Ele10_LW_L1R_HT180",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "22 HLT_Ele10_LW_L1R_HT180/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Ele10_LW_L1R_HT180",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

######## HLT_Ele10_SW_L1R_TripleJet30

### Reco Electrons
trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/00 Elec Mult",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/ElecMult_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Electron Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/01 Elec Pt",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/02 Elec Eta",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/03 Elec Phi",
               [{'path': "HLT/SusyExo/RecoElectrons/HLT/Elec1Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the leading electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoElectrons/HLT/Elec2Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the second electron. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Muons
trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/04 Muon Mult",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/MuonMult_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Muon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/05 Muon Pt",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/06 Muon Eta",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/07 Muon Phi",
               [{'path': "HLT/SusyExo/RecoMuons/HLT/Muon1Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMuons/HLT/Muon2Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Jets
trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/08 Jet Mult",
               [{'path': "HLT/SusyExo/RecoJets/HLT/JetMult_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Jet Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/09 Jet Pt",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/10 Jet Eta",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/11 Jet Phi",
               [{'path': "HLT/SusyExo/RecoJets/HLT/Jet1Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoJets/HLT/Jet2Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco Photons
trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/12 Photon Mult",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/PhotonMult_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Photon Multiplicity. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/13 Photon Pt",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Pt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Pt Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/14 Photon Eta",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Eta_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Eta Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/15 Photon Phi",
               [{'path': "HLT/SusyExo/RecoPhotons/HLT/Photon1Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the leading muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoPhotons/HLT/Photon2Phi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "Phi Distribution of the second muon. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

### Reco MET
trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/16 MET",
               [{'path': "HLT/SusyExo/RecoMET/HLT/MET_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "MET distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/17 MET x-y-phi",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METPhi_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "MET phi distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}],
               [{'path': "HLT/SusyExo/RecoMET/HLT/METx_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "MET x distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."},
                {'path': "HLT/SusyExo/RecoMET/HLT/METy_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "MET y distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/18 METSignificance",
               [{'path': "HLT/SusyExo/RecoMET/HLT/METSignificance_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "MET Significance distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

trigvalsusybsm(dqmitems,
               "23 HLT_Ele10_SW_L1R_TripleJet30/19 SumEt",
               [{'path': "HLT/SusyExo/RecoMET/HLT/SumEt_HLT_Ele10_SW_L1R_TripleJet30",
                 'description': "SumEt distribution. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSUSYBSMHLTOfflinePerformance\">here</a>."}])

###---- HIGGS selection goes here: ----

def trigvalhiggsHWW(dqmitems,layout, *plots):
    destination_1='HLT/Higgs/HiggsValidationReport/HWW/'
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

trigvalhiggsHWW(dqmitems,"mumu selection: HLTMu9 eff vs eta ",
        [{'path': "HLT/Higgs/HWW/EffVsEta_HLT_Mu9", 'description': "Trigger efficiency for HLTMu9 in the dimuon channel vs eta of the highest pt reconstructed muon ( Event selection: at least 2 globalMuons pt>10,20, eta<2.4, opp charge)"}])

trigvalhiggsHWW(dqmitems,"mumu selection : HLTMu9 eff vs pt",
        [{'path': "HLT/Higgs/HWW/EffVsPt_HLT_Mu9", 'description': "Trigger efficiency for HLTMu9 in the dimuon channel vs Pt of the highest pt reconstructed muon (Event selection: at least 2 globalMuons pt>10,20, eta<2.4, opp charge)"}])

trigvalhiggsHWW(dqmitems,"ee selection :HLTEle10LWL1R eff vs eta ",
        [{'path': "HLT/Higgs/HWW/EffVsEta_HLT_Ele10_LW_L1R", 'description':" Trigger efficiency for  HLTEle10LWL1R in the ee channel vs eta of the highest pt reco electron (Event selection: at least 2 electrons pt>10,20,eta<2.4,opp charge,H/E<0.05,0.6< E/p<2.5)"}])

trigvalhiggsHWW(dqmitems,"ee selection: EffVsPt HLT_Ele10LWL1R ",
        [{'path': "HLT/Higgs/HWW/EffVsPt_HLT_Ele10_LW_L1R", 'description': "Trigger efficiency for HLTEle10_LW_L1R in the ee channel vs Pt of the highest pt reco electron (Event selection: at least 2 electrons pt>10,20, eta<2.4,opp charge, H/E<0.05,0.6< E/p<2.5) "}])

trigvalhiggsHWW(dqmitems,"emu selection: EffVsEta HLTMu9 ",
        [{'path': "HLT/Higgs/HWW/EffVsEta_HLT_Mu9_EM", 'description': "Trigger efficiency for HLTMu9 in the emu channel vs muon eta ( Event selection applied: at least 2 leptons pt>10,20,eta<2.4, opp. charge, muons:globalMuons, electrons:H/E<0.05,0.6< E/p<2.5)"}])

trigvalhiggsHWW(dqmitems,"emu selection: EffVsPt HLTEle10LWL1R ",
        [{'path': "HLT/Higgs/HWW/EffVsPt_HLT_Ele10_LW_L1R_EM", 'description': "Trigger efficiency for HLTEle10_LW_L1R in the emu channel vs electron eta ( Event selection applied: at least 2 leptons pt>10,20,eta<2.4, opp. charge, muons:globalMuons, electrons:H/E<0.05,0.6< E/p<2.5)"}])

trigvalhiggsHWW(dqmitems,"mumu selection : global Efficiencies ",
        [{'path': "HLT/Higgs/HWW/Efficiencies_MuonTriggers", 'description': "Muon Trigger efficiencies in the dimuon channel wrt the events passing the selection (at least 2 globalMuons pt>10,20, eta<2.4, opp charge)"}])

trigvalhiggsHWW(dqmitems,"ee selection: global Efficiencies ",
        [{'path': "HLT/Higgs/HWW/Efficiencies_ElectronTriggers", 'description': "Electron Trigger efficiencies in the dielectron channel wrt the events passing the selection ( at least 2 electrons pt>10,20,eta<2.4, opp. charge, H/E<0.05,0.6< E/p<2.5)"}])

trigvalhiggsHWW(dqmitems,"emu selection: global Efficiencies ",
        [{'path': "HLT/Higgs/HWW/TriggerEfficiencies_EmuChannel", 'description': "Trigger efficiencies in the emu channel wrt the events passing the selection ( at least 2 leptons pt>10,20,eta<2.4, opp. charge, muons:globalMuons, electrons:H/E<0.05, 0.6< E/p<2.5)"}])

def trigvalhiggsHgg(dqmitems,layout, *plots):
    destination_1='HLT/Higgs/HiggsValidationReport/Hgg/'
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

trigvalhiggsHgg(dqmitems,"HLTDoublePhoton10L1R eff vs eta",
        [{'path': "HLT/Higgs/Hgg/EffVsEta_HLT_DoublePhoton10_L1R", 'description': "Trigger efficiency for HLTDoublePhoton10 versus eta of the highest pt reconstructed photon in the event passing the selection (at least 2 reconstructed photons pt>20, eta<2.4)"}])
trigvalhiggsHgg(dqmitems,"HLTDoublePhoton10L1R vs pt",
        [{'path': "HLT/Higgs/Hgg/EffVsPt_HLT_DoublePhoton10_L1R", 'description': "Trigger efficiency for HLTDoublePhoton10 versus pt of the highest pt reconstructed photon in the event passing the selection (at least 2 reconstructed photons pt>20, eta<2.4)"}])

trigvalhiggsHgg(dqmitems,"Photon global Efficiencies ",
        [{'path': "HLT/Higgs/Hgg/Efficiencies_PhotonTriggers", 'description': "Photon Trigger efficiencies  wrt the events passing the selection (at least 2 reco photons pt>20, eta<2.4)"}])

def trigvalhiggsH2tau(dqmitems,layout, *plots):
    destination_1='HLT/Higgs/HiggsValidationReport/H2tau/'
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
                  
trigvalhiggsH2tau(dqmitems,"semimu channel: HLTMu3 eff vs eta",
        [{'path': "HLT/Higgs/H2tau/EffVsEta_HLT_Mu3", 'description': "Trigger efficiency for HLTMu3 versus eta of the generated muon from tau decay passing the selection (1 muon from tau pt>15, eta<2.4)"}])

trigvalhiggsH2tau(dqmitems,"semimu channel: HLTMu3 eff vs pt",
        [{'path': "HLT/Higgs/H2tau/EffVsPt_HLT_Mu3", 'description': "Trigger efficiency for HLTMu3 versus pt of the generated muon from tau decay passing the selection (1 muon from tau pt>15, eta<2.4)"}])

trigvalhiggsH2tau(dqmitems,"semielec channel: HLTEle10LW eff vs eta",
        [{'path': "HLT/Higgs/H2tau/EffVsEta_HLT_Ele10_LW_L1R", 'description': "Trigger efficiency for HLTEle10LWL1R versus eta of the generated electron from tau decay passing the selection (1 electron from tau pt>15, eta<2.4)"}])

trigvalhiggsH2tau(dqmitems,"semielec channel: HLTEle10LW eff vs pt",
        [{'path': "HLT/Higgs/H2tau/EffVsPt_HLT_Ele10_LW_L1R", 'description': "Trigger efficiency for HLTEle10LWL1R versus Pt of the generated electron from tau decay passing the selection (1 electron from tau pt>15, eta<2.4)"}])

trigvalhiggsH2tau(dqmitems,"semimuonic channel : global Efficiencies ",
        [{'path': "HLT/Higgs/H2tau/Efficiencies_MuonTriggers", 'description': "Muon Trigger efficiencies  wrt the events passing the muon selection (1 muon from tau decay pt>15, eta<2.4"}])

trigvalhiggsH2tau(dqmitems,"semielectronic channel: global Efficiencies ",
        [{'path': "HLT/Higgs/H2tau/Efficiencies_ElectronTriggers", 'description': "Electron Trigger efficiencies  wrt the events passing the electron selection ( 1 electron from tau pt>15,eta<2.4)"}])

def trigvalhiggsHZZ(dqmitems,layout, *plots):
    destination_1='HLT/Higgs/HiggsValidationReport/HZZ/'
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

trigvalhiggsHZZ(dqmitems,"4mu selection: HLTMu9 eff vs eta ",
        [{'path': "HLT/Higgs/HZZ/EffVsEta_HLT_Mu9", 'description': "Trigger efficiency for HLTMu9 vs eta of the highest pt reconstructed muon in the event passing the selection (at least 4 globalMuons pt>10, eta<2.4, opp charge"}])

trigvalhiggsHZZ(dqmitems,"4mu selection :HLTMu9 eff vs pt ",
        [{'path': "HLT/Higgs/HZZ/EffVsPt_HLT_Mu9", 'description': "Trigger efficiency for HLTMu9  vs Pt of the highest pt reconstructed muon in the event passing the selection (at least 4 globalMuons pt>10, eta<2.4, opp charge"}])

trigvalhiggsHZZ(dqmitems,"4e selection :HLTEle10LWL1R eff vs eta ",
        [{'path': "HLT/Higgs/HZZ/EffVsEta_HLT_Ele10_LW_L1R", 'description': "Trigger efficiency for HLTEle10_LW_L1R  vs eta of the highest pt reconstructed electron in the event passing the selection (at least 4 gsfElectrons pt>10, eta<2.4,opp charge, H/E<0.05, 0.6< E/p<2.5) "}])

trigvalhiggsHZZ(dqmitems,"4e selection: HLT_Ele10LWL1R  eff vs pt",
        [{'path': "HLT/Higgs/HZZ/EffVsPt_HLT_Ele10_LW_L1R", 'description': "Trigger efficiency for HLTEle10_LW_L1R  vs Pt of the highest pt reconstructed electron in the event passing the selection (at least 4 gsfElectrons pt>10, eta<2.4,opp charge, H/E<0.05, 0.6< E/p<2.5) "}])

trigvalhiggsHZZ(dqmitems,"2e2mu selection: HLTMu9 eff vs eta ",
        [{'path': "HLT/Higgs/HZZ/EffVsEta_HLT_Mu9_EM", 'description': "Trigger efficiency for HLTMu9  vs eta of the highest pt reconstructed muon in the event passing the selection ( at least 2 muons and 2 electrons  pt>10,eta<2.4, opp. charge, muons:globalMuons, electrons:H/E<0.05, 0.6< E/p<2.5)"}])

trigvalhiggsHZZ(dqmitems,"2e2mu selection: HLTEle10LWL1R eff vs pt ",
        [{'path': "HLT/Higgs/HZZ/EffVsPt_HLT_Ele10_LW_L1R_EM", 'description': "Trigger efficiency for HLTEle10_LW_L1R  vs pt of the highest pt reconstructed electron in the event passing the selection ( at least 2 muons and 2 electrons pt>10,eta<2.4, opp. charge, muons:globalMuons, electrons:H/E<0.05,0.6< E/p<2.5)"}])

trigvalhiggsHZZ(dqmitems,"4mu selection : global Efficiencies ",
        [{'path': "HLT/Higgs/HZZ/Efficiencies_MuonTriggers", 'description': "Muon Trigger efficiencies  wrt the events passing the selection (at least 4 globalMuons pt>10, eta<2.4, opp charge"}])

trigvalhiggsHZZ(dqmitems,"4e selection: global Efficiencies ",
        [{'path': "HLT/Higgs/HZZ/Efficiencies_ElectronTriggers", 'description': "Electron Trigger efficiencies wrt the events passing the selection ( at least 4 electrons pt>10,eta<2.4, opp. charge, H/E<0.05, 0.6< E/p<2.5)"}])

trigvalhiggsHZZ(dqmitems,"2e2mu selection: global Efficiencies ",
        [{'path': "HLT/Higgs/HZZ/TriggerEfficiencies_EmuChannel", 'description': "Trigger efficiencies  wrt the events passing the selection ( at least 2 muons and 2 electrons pt>10,20,eta<2.4, opp. charge, muons:globalMuons, electrons:H/E<0.05, 0.6< E/p<2.5)"}])

def trigvalhiggsHtaunu(dqmitems,layout, *plots):
    destination_1='HLT/Higgs/HiggsValidationReport/Htaunu/'
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
trigvalhiggsHtaunu(dqmitems,"Tau global Efficiencies ",
        [{'path': "HLT/Higgs/Htaunu/globalEfficiencies", 'description': "Tau trigger efficiencies  wrt the events passing the selection ( at least 1 gen tau pt>100,eta<2.4)"}])

###---- QCD selection goes here: ----
#def trigvalqcd(i, p, *rows): i["HLT//Preselection" + p] = DQMItem(layout=rows)
