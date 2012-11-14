
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used


#PixPhase1
#workflows[3107] = ['', ['FourMuPt1_200_UPGphase1']]
#workflows[3123] = ['', ['MinBias_UPGphase1_14']]
#workflows[3135] = ['', ['TTbar_Tauola_UPGphase1_14']]

#Postls1
workflows[3207] = ['FourMuPt1_200', ['FourMuPt1_200_UPGpostls1','DIGIUP','RECOUP']]

workflows[3223] = ['MinBias', ['MinBias_UPGpostls1_14','DIGIUP','RECOUP']]
workflows[3235] = ['TTbar', ['TTbar_Tauola_UPGpostls1_14','DIGIUP','RECOUP']]

workflows[3201] = ['JpsiMM', ['JpsiMM_UPGpostls1_14','DIGIUP','RECOUP']]
workflows[3202] = ['WM', ['WM_UPGpostls1_14','DIGIUP','RECOUP']]
workflows[3203] = ['ZMM', ['ZMM_UPGpostls1_14','DIGIUP','RECOUP']]
workflows[3204] = ['SingleMuPt10', ['SingleMuPt10_UPGpostls1','DIGIUP','RECOUP']]
workflows[3205] = ['SingleMuPt100', ['SingleMuPt100_UPGpostls1','DIGIUP','RECOUP']]
workflows[3206] = ['SingleMuPt1000', ['SingleMuPt1000_UPGpostls1','DIGIUP','RECOUP']]

workflows[3208] = ['ZmumuJets_Pt20_300', ['ZmumuJets_Pt20_300_UPGpostls1_14','DIGIUP','RECOUP']]
