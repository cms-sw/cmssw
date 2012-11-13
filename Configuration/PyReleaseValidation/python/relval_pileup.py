
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

workflows[200]=['',['ZEE','DIGIPU1','RECOPU1']]
workflows[201]=['',['ZmumuJets_Pt_20_300','DIGIPU1','RECOPU1']]
workflows[202]=['',['TTbar','DIGIPU1','RECOPU1']]
workflows[203]=['',['H130GGgluonfusion','DIGIPU1','RECOPU1']]
workflows[204]=['',['QQH1352T_Tauola','DIGIPU1','RECOPU1']]
workflows[205]=['',['ZTT','DIGIPU1','RECOPU1']]

#heavy ions tests
workflows[300]=['Pyquen_GammaJet_pt20_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_GammaJet_pt20_2760GeV','DIGIHISt3','RECOHISt4']]
workflows[301]=['Pyquen_DiJet_pt80to120_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_DiJet_pt80to120_2760GeV','DIGIHISt3','RECOHISt4']]
workflows[302]=['Pyquen_ZeemumuJets_pt10_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_ZeemumuJets_pt10_2760GeV','DIGIHISt3','RECOHISt4']]

#fastsim
workflows[400]=['TTbar',['TTbarFSPU']]
workflows[401]=['TTbar',['TTbarFSPU2']]
