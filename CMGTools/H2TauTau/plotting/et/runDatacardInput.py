import subprocess
import os
import time

processes = set()
max_processes = 4

doSM = True
doMSSM = False

catsSM = []
catsMSSM = []

if doSM:
    catsSM = {
    'Xcat_IncX && mt<30 && Xcat_J0_lowX':'0jet_low',
    'Xcat_IncX && mt<30 && Xcat_J0_mediumX':'0jet_medium',
    'Xcat_IncX && mt<30 && Xcat_J0_highX':'0jet_high',
    'Xcat_IncX && mt<30 && met>30 && Xcat_J1_mediumX':'1jet_medium',
    'Xcat_IncX && mt<30 && met>30 && Xcat_J1_high_mediumhiggsX':'1jet_high_mediumhiggs',
    'Xcat_IncX && mt<30 && met>30 && Xcat_J1_high_lowhiggsX':'1jet_high_lowhiggs',
    'Xcat_IncX && mt<30 && Xcat_VBF_looseX':'vbf_loose',
    'Xcat_IncX && mt<30 && Xcat_VBF_tightX':'vbf_tight',
}

if doMSSM:
    catsMSSM = {
    'Xcat_IncX && mt<30 && Xcat_J1BX':'btag',
    'Xcat_IncX && mt<30 && Xcat_0BX':'nobtag'
}


dirUp = '/data/steggema/Sep19EleTau_Up/' #NOTE: Must end with Up
dirDown = '/data/steggema/Sep19EleTau_Down/' #NOTE: Must end with Down
dirNom = '/data/steggema/Sep19EleTau/'

dirGGH = '/data/steggema/Sep19EleTau/'


embedded = True

embOpt = ''
if embedded:
    embOpt = '-E'

mssmBinnings = ['2013', 'fine', 'default']

allArgs = []
for cat in catsSM:
    args = ['python', 'plot_H2TauTauDataMC_TauEle_All.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt]


    argsUp = ['python', 'plot_H2TauTauMC.py', dirUp, 'tauEle_2012_up_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'Higgs;Ztt', '-c', 'TauEle']
    argsDown = ['python', 'plot_H2TauTauMC.py', dirDown, 'tauEle_2012_down_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'Higgs;Ztt', '-c', 'TauEle']


    argsGGHUp = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightUp/hqtWeight', '-s', 'QCDscale_ggH1inUp', '-c', 'TauEle']
    argsGGHDown = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightDown/hqtWeight', '-s', 'QCDscale_ggH1inDown', '-c', 'TauEle']

    argsZLUp = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass*1.02', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_etau_8TeVUp', '-c', 'TauEle']
    argsZLDown = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass*0.98', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_etau_8TeVDown', '-c', 'TauEle']

    argsZLSmeared = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass.+l1_phi*1.9', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_etau_8TeVSmearedUp', '-c', 'TauEle']

    argsWJetsUp = ['python', 'plot_H2TauTauDataMC_TauEle_All.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-s', 'CMS_htt_WShape_etau_{cat}_8TeVUp'.format(cat=catsSM[cat])]
    argsWJetsDown = ['python', 'plot_H2TauTauDataMC_TauEle_All.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-s', 'CMS_htt_WShape_etau_{cat}_8TeVDown'.format(cat=catsSM[cat])]

    # allArgs += [args, argsUp, argsDown, argsGGHUp, argsGGHDown, argsZLUp, argsZLDown, argsWJetsUp, argsWJetsDown]
    allArgs += [argsZLSmeared]

for cat in catsMSSM:
    for mssmBinning in mssmBinnings:
            args = ['python', 'plot_H2TauTauDataMC_TauEle_All.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-k', mssmBinning, '-p', mssmBinning, '-g', '125']

            argsUp = ['python', 'plot_H2TauTauMC.py', dirUp, 'tauEle_2012_up_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'SUSY;Higgs;Ztt', '-k', mssmBinning, '-p', mssmBinning, '-c', 'TauEle', '-g', '125']
            argsDown = ['python', 'plot_H2TauTauMC.py', dirDown, 'tauEle_2012_down_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'SUSY;Higgs;Ztt', '-k', mssmBinning, '-p', mssmBinning, '-c', 'TauEle', '-g', '125']


            argsGGHUp = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightUp/hqtWeight', '-s', 'QCDscale_ggH1inUp', '-k', mssmBinning, '-p', mssmBinning, '-c', 'TauEle', '-g', '125']
            argsGGHDown = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightDown/hqtWeight', '-s', 'QCDscale_ggH1inDown', '-k', mssmBinning, '-p', mssmBinning, '-c', 'TauEle', '-g', '125']

            argsZLUp = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass*1.02', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_etau_8TeVUp', '-k', mssmBinning, '-p', mssmBinning, '-c', 'TauEle', '-g', '125']
            argsZLDown = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass*0.98', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_etau_8TeVDown', '-k', mssmBinning, '-p', mssmBinning, '-c', 'TauEle', '-g', '125']

            argsWJetsUp = ['python', 'plot_H2TauTauDataMC_TauEle_All.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-s', 'CMS_htt_WShape_etau_{cat}_8TeVUp'.format(cat=catsMSSM[cat]), '-k', mssmBinning, '-p', mssmBinning, '-g', '125']
            argsWJetsDown = ['python', 'plot_H2TauTauDataMC_TauEle_All.py', dirNom, 'tauEle_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-s', 'CMS_htt_WShape_etau_{cat}_8TeVDown'.format(cat=catsMSSM[cat]), '-k', mssmBinning, '-p', mssmBinning, '-g', '125']

            allArgs += [args, argsUp, argsDown, argsGGHUp, argsGGHDown, argsZLUp, argsZLDown, argsWJetsUp, argsWJetsDown]


for args in allArgs:
    processes.add(subprocess.Popen(args))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            p for p in processes if p.poll() is not None)

#Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()
        # os.wait()
