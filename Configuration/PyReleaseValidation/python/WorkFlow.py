
import re

# ================================================================================

class WorkFlow(object):

    def __init__(self, num, nameID, cmd1=None, cmd2=None, cmd3=None, cmd4=None, inputInfo=None, commands=None):

        self.numId  = num.strip()
        self.nameId = nameID
        self.cmds = []
        self.check(cmd1)
        self.check(cmd2, 100)
        self.check(cmd3, 100)
        self.check(cmd4, 100)
        if commands:
            for (i,c) in enumerate(commands):
                self.check(c,10 + (i==0)*90)
        

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
    
