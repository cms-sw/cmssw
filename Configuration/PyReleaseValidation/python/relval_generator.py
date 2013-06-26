
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

workflows[501]=['',['MinBias_TuneZ2star_8TeV_pythia6','HARVGEN']]
workflows[502]=['',['QCD_Pt-30_TuneZ2star_8TeV_pythia6','HARVGEN']]
workflows[503]=['',['TT_TuneZ2star_8TeV_pythia6-evtgen','HARVGEN']]
workflows[504]=['',['DYToLL_M-50_TuneZ2star_8TeV_pythia6-tauola','HARVGEN']]
workflows[505]=['',['WToLNu_TuneZ2star_8TeV_pythia6-tauola','HARVGEN']]
workflows[506]=['',['MinBias_8TeV_pythia8','HARVGEN']]
workflows[507]=['',['QCD_Pt-30_8TeV_pythia8','HARVGEN']]
workflows[508]=['',['QCD_Pt-30_8TeV_herwig6','HARVGEN']]
workflows[509]=['',['QCD_Pt-30_8TeV_herwigpp','HARVGEN']]
workflows[510]=['',['GluGluTo2Jets_M-100_8TeV_exhume','HARVGEN']]
workflows[517]=['',['QCD_Ht-100To250_TuneZ2star_8TeV_madgraph-tauola','HARVGEN']]
workflows[518]=['',['QCD_Ht-250To500_TuneZ2star_8TeV_madgraph-tauola','HARVGEN']]
workflows[519]=['',['QCD_Ht-500To1000_TuneZ2star_8TeV_madgraph-tauola','HARVGEN']]
workflows[520]=['',['TTJets_TuneZ2star_8TeV_madgraph-tauola','HARVGEN']]
workflows[521]=['',['WJetsLNu_TuneZ2star_8TeV_madgraph-tauola','HARVGEN']]
workflows[522]=['',['ZJetsLNu_TuneZ2star_8TeV_madgraph-tauola','HARVGEN']]
workflows[539]=['',['ZJetsLNu_Tune4C_8TeV_madgraph-pythia8','HARVGEN']]
workflows[540]=['',['ReggeGribovPartonMC_EposLHC_5TeV_pPb','HARVGEN']]
