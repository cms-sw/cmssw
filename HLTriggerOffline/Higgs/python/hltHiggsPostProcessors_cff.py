import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Higgs.hltHiggsPostProcessor_cfi import *

# Build the standard strings to the DQM
def efficiency_string(objtype,plot_type,triggerpath):
    # --- IMPORTANT: Add here a elif if you are introduce a new collection
    #                (see EVTColContainer::getTypeString) 
    if objtype == "Mu":
        objtypeLatex="#mu"
    elif objtype == "Photon": 
        objtypeLatex="#gamma"
    elif objtype == "Ele": 
        objtypeLatex="e"
    elif objtype == "MET" :
        objtypeLatex="MET"
    elif objtype == "PFMET" :
        objtypeLatex="PFMET"
    elif objtype == "PFTau": 
        objtypeLatex="#tau"
    elif objtype == "Jet": 
        objtypeLatex="jet"
    else:
        objtypeLatex=objtype

    numer_description = "# gen %s passed the %s" % (objtypeLatex,triggerpath)
    denom_description = "# gen %s " % (objtypeLatex)

    if plot_type == "TurnOn1":
        title = "pT Turn-On"
        xAxis = "p_{T} of Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sMaxPt1" % (objtype)
    elif plot_type == "TurnOn2":
        title = "Next-to-Leading pT Turn-On"
        xAxis = "p_{T} of Next-to-Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sMaxPt2" % (objtype)
    elif plot_type == "EffEta":
        title = "#eta Efficiency"
        xAxis = "#eta of Generated %s " % (objtype)
        input_type = "gen%sEta" % (objtype)
    elif plot_type == "EffPhi":
        title = "#phi Efficiency"
        xAxis = "#phi of Generated %s " % (objtype)
        input_type = "gen%sPhi" % (objtype)     
    elif "TurnOn" in plot_type:
        title = "%sth Leading pT Turn-On" % (plot_type[-1])
        xAxis = "p_{T} of %sth Leading Generated %s (GeV/c)" % (plot_type[-1], objtype)
        input_type = "gen%sMaxPt%s" % (objtype, plot_type[-1])
    elif plot_type == "EffdEtaqq":
        title = "#Delta #eta_{qq} Efficiency"
        xAxis = "#Delta #eta_{qq} of Generated %s " % (objtype)
        input_type = "gen%sdEtaqq" % (objtype)
    elif plot_type == "Effmqq":
        title = "m_{qq} Efficiency"
        xAxis = "m_{qq} of Generated %s " % (objtype)
        input_type = "gen%smqq" % (objtype)
    elif plot_type == "EffdPhibb":
        title = "#Delta #phi_{bb} Efficiency"
        xAxis = "#Delta #phi_{bb} of Generated %s " % (objtype)
        input_type = "gen%sdPhibb" % (objtype)
    elif plot_type == "EffCSV1":
        title = "CSV1 Efficiency"
        xAxis = "CSV1 of Generated %s " % (objtype)
        input_type = "gen%sCSV1" % (objtype)
    elif plot_type == "EffCSV2":
        title = "CSV2 Efficiency"
        xAxis = "CSV2 of Generated %s " % (objtype)
        input_type = "gen%sCSV2" % (objtype)
    elif plot_type == "EffCSV3":
        title = "CSV3 Efficiency"
        xAxis = "CSV3 of Generated %s " % (objtype)
        input_type = "gen%sCSV3" % (objtype)
    elif plot_type == "EffmaxCSV":
        title = "max CSV Efficiency"
        xAxis = "max CSV of Generated %s " % (objtype)
        input_type = "gen%smaxCSV" % (objtype)
        
    yAxis = "%s / %s" % (numer_description, denom_description)
    all_titles = "%s for trigger %s; %s; %s" % (title, triggerpath,
                                        xAxis, yAxis)
    return "Eff_%s_%s '%s' %s_%s %s" % (input_type,triggerpath,
            all_titles,input_type,triggerpath,input_type)

# Adding the reco objects
def get_reco_strings(strings):
    reco_strings = []
    for entry in strings:
        reco_strings.append(entry
                            .replace("Generated", "Reconstructed")
                            .replace("Gen", "Reco")
                            .replace("gen", "rec"))
    return reco_strings


plot_types = ["TurnOn1", "TurnOn2", "EffEta", "EffPhi"]
#--- IMPORTANT: Update this collection whenever you introduce a new object
#               in the code (from EVTColContainer::getTypeString)
obj_types  = ["Mu","Ele","Photon","MET","PFMET","PFTau","Jet"]
#--- IMPORTANT: Trigger are extracted from the hltHiggsValidator_cfi.py module
triggers = [] 
efficiency_strings = []

# Extract the triggers used in the hltHiggsValidator 
from HLTriggerOffline.Higgs.hltHiggsValidator_cfi import hltHiggsValidator as _config
triggers = set([])
for an in _config.analysis:
    s = _config.__getattribute__(an)
    vstr = s.__getattribute__("hltPathsToCheck")
    map(lambda x: triggers.add(x.replace("_v","")),vstr)
triggers = list(triggers)

#------------------------------------------------------------

# Generating the list with all the efficiencies
for type in plot_types:
    for obj in obj_types:
        for trig in triggers:
            efficiency_strings.append(efficiency_string(obj,type,trig))


#add the summary plots
efficiency_summary_strings = []
for an in _config.analysis:
    efficiency_summary_strings.append("EffSummaryPaths_"+an+"_gen ' Efficiency of paths used in "+an+" ; trigger path ' SummaryPaths_"+an+"_gen_passingHLT SummaryPaths_"+an+"_gen")
    for trig in triggers:
        efficiency_summary_strings.append("Eff_trueVtxDist_"+an+"_gen_"+trig+" ' Efficiency of "+trig+" vs nb of interactions ; nb events passing each path ' trueVtxDist_"+an+"_gen_"+trig+" trueVtxDist_"+an+"_gen")

efficiency_strings.extend(efficiency_summary_strings)
efficiency_strings.extend(get_reco_strings(efficiency_strings))

hltHiggsPostHWW = hltHiggsPostProcessor.clone()
hltHiggsPostHWW.subDirs = ['HLT/Higgs/HWW']
hltHiggsPostHWW.efficiencyProfile = efficiency_strings

hltHiggsPostHZZ = hltHiggsPostProcessor.clone()
hltHiggsPostHZZ.subDirs = ['HLT/Higgs/HZZ']
hltHiggsPostHZZ.efficiencyProfile = efficiency_strings

hltHiggsPostHgg = hltHiggsPostProcessor.clone()
hltHiggsPostHgg.subDirs = ['HLT/Higgs/Hgg']
hltHiggsPostHgg.efficiencyProfile = efficiency_strings

hltHiggsPostHggControlPaths = hltHiggsPostProcessor.clone()
hltHiggsPostHggControlPaths.subDirs = ['HLT/Higgs/HggControlPaths']
hltHiggsPostHggControlPaths.efficiencyProfile = efficiency_strings

hltHiggsPostDoubleHinTaus = hltHiggsPostProcessor.clone()
hltHiggsPostDoubleHinTaus.subDirs = ['HLT/Higgs/DoubleHinTaus']
hltHiggsPostDoubleHinTaus.efficiencyProfile = efficiency_strings

hltHiggsPostHiggsDalitz = hltHiggsPostProcessor.clone()
hltHiggsPostHiggsDalitz.subDirs = ['HLT/Higgs/HiggsDalitz']
hltHiggsPostHiggsDalitz.efficiencyProfile = efficiency_strings

hltHiggsPostH2tau = hltHiggsPostProcessor.clone()
hltHiggsPostH2tau.subDirs = ['HLT/Higgs/H2tau']
hltHiggsPostH2tau.efficiencyProfile = efficiency_strings

hltHiggsPostHtaunu = hltHiggsPostProcessor.clone()
hltHiggsPostHtaunu.subDirs = ['HLT/Higgs/Htaunu']
hltHiggsPostHtaunu.efficiencyProfile = efficiency_strings


efficiency_strings_TTHbbej = []
#add the summary plots
for an in _config.analysis:
    for trig in triggers:
        efficiency_strings_TTHbbej.append("Eff_HtDist_"+an+"_gen_"+trig+" ' Efficiency of "+trig+" vs sum pT of jets ; sum pT of jets ' HtDist_"+an+"_gen_"+trig+" HtDist_"+an+"_gen")

efficiency_strings_TTHbbej.extend(get_reco_strings(efficiency_strings_TTHbbej))
efficiency_strings_TTHbbej.extend(efficiency_strings)

hltHiggsPostTTHbbej = hltHiggsPostProcessor.clone()
hltHiggsPostTTHbbej.subDirs = ['HLT/Higgs/TTHbbej']
hltHiggsPostTTHbbej.efficiencyProfile = efficiency_strings_TTHbbej

hltHiggsPostAHttH = hltHiggsPostProcessor.clone()
hltHiggsPostAHttH.subDirs = ['HLT/Higgs/AHttH']
hltHiggsPostAHttH.efficiencyProfile = efficiency_strings

#Specific plots for VBFHbb_2btag  
#dEtaqq, mqq, dPhibb, CVS1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
NminOneCutNames = ("EffdEtaqq", "Effmqq", "EffdPhibb", "EffCSV1", "EffCSV2", "EffCSV3",  "EffmaxCSV", "", "", "TurnOn1", "TurnOn2", "TurnOn3", "TurnOn4")
plot_types = []
NminOneCuts =_config.VBFHbb_2btag.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            if( NminOneCutNames[iCut] == "EffmaxCSV" ):
                plot_types.pop()
            plot_types.append(NminOneCutNames[iCut])

efficiency_strings = []
for type in plot_types:
    for obj in ["Jet"]:
        for trig in triggers:
            efficiency_strings.append(efficiency_string(obj,type,trig))

efficiency_strings = get_reco_strings(efficiency_strings)
efficiency_strings.extend(get_reco_strings(efficiency_summary_strings))

hltHiggsPostVBFHbb_2btag = hltHiggsPostProcessor.clone()
hltHiggsPostVBFHbb_2btag.subDirs = ['HLT/Higgs/VBFHbb_2btag']
hltHiggsPostVBFHbb_2btag.efficiencyProfile = efficiency_strings

#Specific plots for VBFHbb_1btag  
#dEtaqq, mqq, dPhibb, CVS1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
plot_types = []
NminOneCuts = _config.VBFHbb_1btag.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            if( NminOneCutNames[iCut] == "EffmaxCSV" ):
                plot_types.pop()
            plot_types.append(NminOneCutNames[iCut])
                       
efficiency_strings = []
for type in plot_types:
    for obj in ["Jet"]:
        for trig in triggers:
            efficiency_strings.append(efficiency_string(obj,type,trig))

efficiency_strings = get_reco_strings(efficiency_strings)
efficiency_strings.extend(get_reco_strings(efficiency_summary_strings))
    
hltHiggsPostVBFHbb_1btag = hltHiggsPostProcessor.clone()
hltHiggsPostVBFHbb_1btag.subDirs = ['HLT/Higgs/VBFHbb_1btag']
hltHiggsPostVBFHbb_1btag.efficiencyProfile = efficiency_strings

#Specific plots for VBFHbb_0btag  
#dEtaqq, mqq, dPhibb, CVS1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
plot_types = []
NminOneCuts = _config.VBFHbb_0btag.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            if( NminOneCutNames[iCut] == "EffmaxCSV" ):
                plot_types.pop()
            plot_types.append(NminOneCutNames[iCut])

efficiency_strings = []
for type in plot_types:
    for obj in ["Jet"]:
        for trig in triggers:
            efficiency_strings.append(efficiency_string(obj,type,trig))

efficiency_strings = get_reco_strings(efficiency_strings)
efficiency_strings.extend(get_reco_strings(efficiency_summary_strings))

hltHiggsPostVBFHbb_0btag = hltHiggsPostProcessor.clone()
hltHiggsPostVBFHbb_0btag.subDirs = ['HLT/Higgs/VBFHbb_0btag']
hltHiggsPostVBFHbb_0btag.efficiencyProfile = efficiency_strings


#Specific plots for ZnnHbb
#Jet plots
plot_types = ["EffEta", "EffPhi"]
NminOneCuts = _config.ZnnHbb.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            plot_types.append(NminOneCutNames[iCut])
    
efficiency_strings = []
for type in plot_types:
    for obj in ["Jet"]:
        for trig in triggers:
            efficiency_strings.append(efficiency_string(obj,type,trig))
        
efficiency_strings = get_reco_strings(efficiency_strings)

#PFMET plots
plot_types = ["TurnOn1", "EffPhi"]
efficiency_strings2 = []
for type in plot_types:
    for obj in ["PFMET"]:
        for trig in triggers:
            efficiency_strings2.append(efficiency_string(obj,type,trig))

efficiency_strings2 = get_reco_strings(efficiency_strings2)
efficiency_strings += efficiency_strings2
efficiency_strings.extend(get_reco_strings(efficiency_summary_strings))
    
hltHiggsPostZnnHbb = hltHiggsPostProcessor.clone()
hltHiggsPostZnnHbb.subDirs = ['HLT/Higgs/ZnnHbb']
hltHiggsPostZnnHbb.efficiencyProfile = efficiency_strings


#Specific plots for X4b
#Jet plots
NminOneCuts = _config.X4b.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            plot_types.append(NminOneCutNames[iCut])

efficiency_strings = []
for type in plot_types:
    for obj in ["Jet"]:
        for trig in triggers:
            efficiency_strings.append(efficiency_string(obj,type,trig))
        
efficiency_strings = get_reco_strings(efficiency_strings)
efficiency_strings.extend(get_reco_strings(efficiency_summary_strings))

hltHiggsPostX4b = hltHiggsPostProcessor.clone()
hltHiggsPostX4b.subDirs = ['HLT/Higgs/X4b']
hltHiggsPostX4b.efficiencyProfile = efficiency_strings

#Specific plots for WH -> ev + bb
efficiency_strings_WHToENuBB = list(efficiency_strings_TTHbbej)
#add the summary plots
for an in _config.analysis:
    for trig in triggers:
        efficiency_strings_WHToENuBB.append("Eff_HtDist_"+an+"_gen_"+trig+" ' Efficiency of "+trig+" vs sum pT of jets ; sum pT of jets ' HtDist_"+an+"_gen_"+trig+" HtDist_"+an+"_gen")

efficiency_strings_WHToENuBB.extend(get_reco_strings(efficiency_strings_WHToENuBB))

hltHiggsPostWHToENuBB = hltHiggsPostProcessor.clone()
hltHiggsPostWHToENuBB.subDirs = ['HLT/Higgs/WHToENuBB']
hltHiggsPostWHToENuBB.efficiencyProfile = efficiency_strings_WHToENuBB

#Specific plots for MSSMHbb
#Jet plots
NminOneCuts = _config.MSSMHbb.NminOneCuts
if NminOneCuts:
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            plot_types.append(NminOneCutNames[iCut])

efficiency_strings = []
for type in plot_types:
    for obj in ["Jet"]:
        for trig in triggers:
            efficiency_strings.append(efficiency_string(obj,type,trig))

efficiency_strings = get_reco_strings(efficiency_strings)
efficiency_strings.extend(get_reco_strings(efficiency_summary_strings))

hltHiggsPostMSSMHbb = hltHiggsPostProcessor.clone()
hltHiggsPostMSSMHbb.subDirs = ['HLT/Higgs/MSSMHbb']
hltHiggsPostMSSMHbb.efficiencyProfile = efficiency_strings




hltHiggsPostProcessors = cms.Sequence(
        hltHiggsPostHWW+
        hltHiggsPostHZZ+
        hltHiggsPostHgg+
        hltHiggsPostHggControlPaths+
        hltHiggsPostHtaunu+
        hltHiggsPostH2tau+
        hltHiggsPostTTHbbej+
        hltHiggsPostAHttH+
        hltHiggsPostVBFHbb_0btag+
        hltHiggsPostVBFHbb_1btag+
        hltHiggsPostVBFHbb_2btag+
        hltHiggsPostZnnHbb+
        hltHiggsPostDoubleHinTaus+
        hltHiggsPostHiggsDalitz+
        hltHiggsPostX4b+
        hltHiggsPostWHToENuBB+
        hltHiggsPostMSSMHbb

)


