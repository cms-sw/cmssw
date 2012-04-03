
import re

# ================================================================================

class WorkFlow(object):

    def __init__(self, num, nameID, defaultNEvents, inputInfo=[], commands=[]):

        self.numId  = num
        self.nameId = nameID
        self.cmds = []
        # We always run 10 events for step 0 otherwise get the defaults from
        # defaultNEvents
        nEvents = [10] + [defaultNEvents for x in xrange(len(commands)-1)]
        specs = zip(commands, nEvents)
        for (c, n) in specs:
            self.check(c, n)
        # run on real data requested
        self.input = inputInfo

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
    
