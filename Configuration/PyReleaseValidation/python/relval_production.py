
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

## data production test
workflows[1000] = [ '',['RunMinBias2011A','TIER0','SKIMD','HARVESTDfst2','ALCASPLIT']]
workflows[1001] = [ '',['RunMinBias2011A','TIER0EXP','ALCAEXP','ALCAHARVDSIPIXELCALRUN1','ALCAHARVD1','ALCAHARVD2','ALCAHARVD3','ALCAHARVD4','ALCAHARVD5']]
workflows[1001.2] = [ '',['RunZeroBias2017F','TIER0EXPRUN2','ALCAEXPRUN2','ALCAHARVDSIPIXELCAL']]

workflows[1002]=['RRD',['RunMinBias2011A','RECODR1','COPYPASTE']]
workflows[1003]=['', ['RunMinBias2012A','RECODDQM','HARVESTDDQM']]
workflows[1004] = [ '',['RunHI2011','TIER0EXPHI','ALCAEXPHI','ALCAHARVD1HI','ALCAHARVD2HI','ALCAHARVD3HI','ALCAHARVD5HI']]

workflows[1010] =  ['',['TestEnableEcalHCAL2017B','TIER0EXPTE', 'ALCAEXPTE', 'ALCAHARVDTE']]
workflows[1020] =  ['',['AlCaLumiPixels2016H','TIER0EXPLP','ALCAEXPLP','ALCAHARVLP', 'TIER0PROMPTLP']]
workflows[1030] =  ['',['RunHLTPhy2017B','TIER0EXPHPBS','ALCASPLITHPBS','ALCAHARVDHPBS', 'ALCAHARVDHPBSLOWPU']]

workflows[1040] =  ['',['RunZeroBias2017F','TIER0RAWSIPIXELCAL','ALCASPLITSIPIXELCAL','ALCAHARVDSIPIXELCAL']]
workflows[1040.1] =  ['',['RunExpressPhy2017F','TIER0EXPSIPIXELCAL','ALCASPLITSIPIXELCAL','ALCAHARVDSIPIXELCAL']]


## MC production test
#workflows[1100] = [ '',[]]

#workflows[1100]=['',['OldGenSimINPUT','REDIGIPU','RERECOPU1']]
workflows[1102]=['RR', ['TTbar','DIGI','RECO','RECOFROMRECO','COPYPASTE']]
#workflows[1103]=['RR', ['OldTTbarINPUT','RECOFROMRECOSt2']]

#workflows[1104]=['',['OldGenSimINPUT','RESIM','DIGIPU','RERECOPU']]

## special fastsim test
#workflows[1200]=['TTbar',['TTbarSFS','RECOFS','HARVESTFS']]
#workflows[1201]=['TTbar',['TTbarSFSA']]
