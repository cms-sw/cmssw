import subprocess
import os
import time

processes = set()
max_processes = 4

doSM = True
doMSSM = True

catsSM = []
catsMSSM = []

if doSM:
    catsSM = [
    'Xcat_IncX && mt<30 && Xcat_J0_lowX',
    'Xcat_IncX && mt<30 && Xcat_J0_mediumX',
    'Xcat_IncX && mt<30 && Xcat_J0_highX',
	'Xcat_IncX && mt<30 && Xcat_J1_mediumX',
	'Xcat_IncX && mt<30 && Xcat_J1_high_mediumhiggsX',
	'Xcat_IncX && mt<30 && Xcat_J1_high_lowhiggsX',
	'Xcat_IncX && mt<30 && Xcat_VBF_looseX',
	'Xcat_IncX && mt<30 && Xcat_VBF_tightX',
]

if doMSSM:
    catsMSSM = [
    'Xcat_IncX && mt<30 && Xcat_J1BX',
    'Xcat_IncX && mt<30 && Xcat_0BX'
]

dirUp = '/data/steggema/Aug07MuTau_Up/'
dirDown = '/data/steggema/Aug07MuTau_Down/'
dirNom = '/data/steggema/Aug07MuTau/'

dirGGH = '/data/steggema/Aug07MuTau/'

embedded = True

embOpt = ''
if embedded:
	embOpt = '-E'

mssmBinnings = ['2013', 'fine', 'default']

allArgs = []
for cat in catsSM:
    args = ['python', 'plot_H2TauTauDataMC_TauMu_All.py', dirNom, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt]


    argsUp = ['python', 'plot_H2TauTauMC.py', dirUp, 'tauMu_2012_tesup_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'Higgs;Ztt']
    argsDown = ['python', 'plot_H2TauTauMC.py', dirDown, 'tauMu_2012_tesdown_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'Higgs;Ztt']


    argsGGHUp = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightUp/hqtWeight', '-s', 'QCDscale_ggH1inUp']
    argsGGHDown = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightDown/hqtWeight', '-s', 'QCDscale_ggH1inDown']

    argsZLUp = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass*1.02', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_mutau_8TeVUp']
    argsZLDown = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass*0.98', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_mutau_8TeVDown']

    allArgs += [args, argsUp, argsDown, argsGGHUp, argsGGHDown, argsZLUp, argsZLDown]

for cat in catsMSSM:
    for mssmBinning in mssmBinnings:
            args = ['python', 'plot_H2TauTauDataMC_TauMu_All.py', dirNom, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-k', mssmBinning, '-p', mssmBinning, '-g', '125']

            argsUp = ['python', 'plot_H2TauTauMC.py', dirUp, 'tauMu_2012_tesup_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'SUSY;Higgs;Ztt', '-k', 
mssmBinning, '-p', mssmBinning, '-g', '125']
            argsDown = ['python', 'plot_H2TauTauMC.py', dirDown, 'tauMu_2012_tesdown_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', embOpt, '-f', 'SUSY;Higgs;Ztt', '-k', 
mssmBinning, '-p', mssmBinning, '-g', '125']


            argsGGHUp = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightUp/hqtWeight', '-s', 'QCDscale_ggH1inUp', '-k', mssmBinning, '-p', mssmBinning, '-g', '125']
            argsGGHDown = ['python', 'plot_H2TauTauMC.py', dirGGH, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass', '-b', '-f', 'HiggsGGH', '-w', 'weight*hqtWeightDown/hqtWeight', '-s', 'QCDscale_ggH1inDown', '-k', mssmBinning, '-p', mssmBinning, '-g', '125']
            #ZL_CMS_htt_ZLScale_mutau_8TeVUp
            argsZLUp = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass*1.02', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_mutau_8TeVUp', '-k', mssmBinning, '-p', mssmBinning, '-g', '125']
            argsZLDown = ['python', 'plot_H2TauTauMC.py', dirNom, 'tauMu_2012_cfg.py', '-C', cat, '-H', 'svfitMass*0.98', '-b', '-f', 'Ztt', '-s', 'CMS_htt_ZLScale_mutau_8TeVDown', '-k', mssmBinning, '-p', mssmBinning, '-g', '125']

            allArgs += [args, argsUp, argsDown, argsGGHUp, argsGGHDown, argsZLUp, argsZLDown]


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

