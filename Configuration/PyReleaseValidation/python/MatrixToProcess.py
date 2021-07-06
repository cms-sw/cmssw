#!/usr/bin/env python3


from __future__ import print_function
class MatrixToProcess:

    def __init__(self,what='standard',strict=True):
        from Configuration.PyReleaseValidation.MatrixReader import MatrixReader
        self.mrd = MatrixReader(what,noRun=True)
        self.mrd.prepare('all','',None)
        self.configBuilders={}
        self.processes={}
        self.strict=strict
    def getKey(self,wfNumber,step):
        return str(wfNumber)+':'+str(step)
    
    def getProcess(self,wfNumber,step):
        key=self.getKey(wfNumber,step)
        if not key in self.configBuilders:
            self.load(wfNumber,step)
        if not key in self.configBuilders:
            return None
        return self.configBuilders[key].process

    def load(self,wfNumber,step):
        from Configuration.Applications.ConfigBuilder import ConfigBuilder
        from Configuration.Applications.cmsDriverOptions import OptionsFromCommand
        import copy

        if len(self.configBuilders)!=0 and self.strict:
            raise Exception('one should never be loading more than one process at a time due to python loading/altering feature')
        key=self.getKey(wfNumber,step)
        if key in self.configBuilders:
            return True
        
        for wf in self.mrd.workFlows:
            if float(wf.numId)!=wfNumber: continue

            if not hasattr(wf,'cmdStep%d'%(step)): continue
            if not getattr(wf,'cmdStep%d'%(step)): continue
            
            command=getattr(wf,'cmdStep%d'%(step))
            opt=OptionsFromCommand(command)
            if opt:
                cb = ConfigBuilder(opt,with_input=True,with_output=True)
                cb.prepare()
                self.configBuilders[key]=copy.copy(cb)
                return True
        print("could not satisfy the request for step",step,"of workflow",wfNumber)
        return False
                             
    def getConfig(self,wfNumber,step):
        key=self.getKey(wfNumber,step)
        if not key in self.configBuilders:   self.getProcess(wfNumber,step)
        if not key in self.configBuilders: return None
        return self.configBuilders[key].pythonCfgCode

    def identityTest(self,wfNumber,step):
        self.getProcess(wfNumber,step)
        key=self.getKey(wfNumber,step)
        if not key in self.configBuilders: return None

        cb=self.configBuilders[key]
        #need to compare those two for identity
        cb.process
        cd.pythonCfgCode

                    
    def listAll(self):
        for wf in self.mrd.workFlows:
            step=1
            print('---------------------') 
            print('process workflow',wf.numId)
            print()
            while self.load(float(wf.numId),step):
                p=self.getProcess(float(wf.numId),step)
                print(', '.join(s.label() for s in p.schedule))
                #print p.outputModules()
                step+=1

