
import re

# ================================================================================

class WorkFlow(object):

    def __init__(self, num, nameID, cmd1, cmd2=None, cmd3=None, cmd4=None, inputInfo=None):

        self.numId  = num.strip()
        self.nameId = nameID
        self.cmdStep1 = self.check(cmd1)
        self.cmdStep2 = self.check(cmd2, 100)
        self.cmdStep3 = self.check(cmd3, 100)
        self.cmdStep4 = self.check(cmd4, 100)

        # run on real data requested:
        self.input = inputInfo

        return

    def check(self, cmd=None, nEvtDefault=10):
        if not cmd : return None

        # raw data are treated differently ...
        if 'DATAINPUT' in cmd: return cmd

        if ' -n ' not in cmd:
            cmd+=' -n '+str(nEvtDefault)+' '
        return cmd

