
import re

# ================================================================================

class WorkFlow(object):

    def __init__(self, num, nameID, inputInfo=None, commands=None, stepList=None):

        self.numId  = num
        self.nameId = nameID
        self.cmds = []

        if commands:
            for (i,c) in enumerate(commands):
                nToRun=10 + (i!=0)*90
                self.check(c,nToRun)
        self.stepList = stepList
        if commands and stepList:
            assert(len(commands)==len(stepList))

        # run on real data requested:
        self.input = inputInfo

        return

    def check(self, cmd=None, nEvtDefault=10):
        if not cmd : return None

        if (isinstance(cmd,str)) and ( ' -n ' not in cmd):
            cmd+=' -n '+str(nEvtDefault)+' '

        self.cmds.append(cmd)
        return cmd


class WorkFlowConnector(object):
    def __init__(self):
        self.moduleName=''
        self.tier=''
        self.fileName=''
    
class WorkFlowBlock(object):
    def __init__(self, name,cmdDict):
        self.nameId = name
        self.command = ''#made from the cmdDict

        ##I/O of the block
        self.ins=None
        self.outs=None

    def getProcess(self):
        #get ConfigBuilder to give a process back
        return None
    
