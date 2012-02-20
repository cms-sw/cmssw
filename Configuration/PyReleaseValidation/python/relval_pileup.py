
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

workflows[200]=['',['ZEE','DIGIPU1','RECOPU1']]
workflows[201]=['',['ZmumuJets_Pt_20_300','DIGIPU1','RECOPU1']]
workflows[202]=['',['TTbar','DIGIPU1','RECOPU1']]


#heavy ions tests
workflows[203]=['Pyquen_GammaJet_pt20_2760GeV',['HydjetQ_B0_2760GeVPUINPUT','Pyquen_GammaJet_pt20_2760GeV','DIGIHISt3','RECOHISt4']]
workflows[204]=['Pyquen_DiJet_pt80to120_2760GeV',['HydjetQ_B0_2760GeVPUINPUT','Pyquen_DiJet_pt80to120_2760GeV','DIGIHISt3','RECOHISt4']]
workflows[205]=['Pyquen_ZeemumuJets_pt10_2760GeV',['HydjetQ_B0_2760GeVPUINPUT','Pyquen_ZeemumuJets_pt10_2760GeV','DIGIHISt3','RECOHISt4']]

workflows[203.1]=['Pyquen_GammaJet_pt20_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_GammaJet_pt20_2760GeV','DIGIHISt3','RECOHISt4']]
workflows[204.1]=['Pyquen_DiJet_pt80to120_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_DiJet_pt80to120_2760GeV','DIGIHISt3','RECOHISt4']]
workflows[205.1]=['Pyquen_ZeemumuJets_pt10_2760GeV',['HydjetQ_MinBias_2760GeVINPUT','Pyquen_ZeemumuJets_pt10_2760GeV','DIGIHISt3','RECOHISt4']]

#fastsim
workflows[206]=['TTbar',['TTbarFSPU']]
