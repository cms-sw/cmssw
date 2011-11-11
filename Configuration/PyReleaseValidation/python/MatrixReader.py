
import sys

from Configuration.PyReleaseValidation.WorkFlow import WorkFlow

# ================================================================================

class MatrixException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
        
# ================================================================================

class MatrixReader(object):

    def __init__(self, what='all',noRun=False):

        self.reset(what)
        self.noRun = noRun
        return

    def reset(self, what='all'):

        self.what = what

        self.step1WorkFlows = {}
        self.step2WorkFlows = {}
        self.step3WorkFlows = {}
        self.step4WorkFlows = {}

        self.workFlows = []
        self.nameList  = {}
        
        self.filesPrefMap = {'relval_standard' : 'std-' ,
                             'relval_highstats': 'hi-'  ,
                             'relval_pileup': 'PU-'  ,
                             'relval_generator': 'gen-'  ,
                             'relval_production': 'prod-'  ,
                             }

        self.files = ['relval_standard' ,
                      'relval_highstats',
                      'relval_pileup',
                      'relval_generator',
                      'relval_production',
                      ]

        self.relvalModule = None
        
        return

    def makeCmd(self, step):

        cmd = ''
        cfg = None
        input = None
        #print step
        #print defaults
        for k,v in step.items():
            if 'no_exec' in k : continue  # we want to really run it ...
            if k.lower() == 'cfg':
                cfg = v
                continue # do not append to cmd, return separately
            if k.lower() == 'input':
                input = v
                continue # do not append to cmd, return separately
            #print k,v
            cmd += ' ' + k + ' ' + str(v)
        if self.noRun:
            cmd += ' --no_exec '
        return cfg, input, cmd
    
    def readMatrix(self, fileNameIn, useInput=None, refRel='CMSSW_4_2_0_pre2', fromScratch=None):
        
        prefix = self.filesPrefMap[fileNameIn]

        print "processing ", fileNameIn

        try:
            _tmpMod = __import__( 'Configuration.PyReleaseValidation.'+fileNameIn )
            self.relvalModule = sys.modules['Configuration.PyReleaseValidation.'+fileNameIn]
        except Exception, e:
            print "ERROR importing file ", fileNameIn, str(e)
            return

        print "request for INPUT for ", useInput

        for num, wfInfo in self.relvalModule.workflows.items():
            wfName = wfInfo[0]
            stepList = wfInfo[1]
            addTo=None
            addCom=None
            if len(wfInfo)>=3:
                addCom=wfInfo[2]
                if not type(addCom)==list:
                    addCom=[addCom]
                #print 'added dict',addCom
                if len(wfInfo)>=4:
                    addTo=wfInfo[3]
                    #pad with 0
                    while len(addTo)!=len(stepList):
                        addTo.append(0)
            # if no explicit name given for the workflow, use the name of step1
            if wfName.strip() == '': wfName = stepList[0] 
            stepCmds = ['','','','']
            stepIndex = 0
            name  = wfName
            inputInfo = None
            for step in stepList:
                if len(name) > 0 : name += '+'
                stepName = step
                #use input check, only for step0
                if stepIndex==0:
                    if useInput and (str(num) in useInput or "all" in useInput):
                        if step+'INPUT' in self.relvalModule.step1.keys():
                            stepName = step+"INPUT"
                        if fromScratch and (str(num) in fromScratch or "all" in fromScratch):
                            msg = "FATAL ERROR: request for both fromScratch and input for workflow "+str(num)
                            raise MatrixException(msg)

                name += stepName
                if addCom and (not addTo or addTo[stepIndex]==1):
                    from Configuration.PyReleaseValidation.relval_steps import merge
                    copyStep=merge(addCom+[self.relvalModule.stepList[stepIndex][stepName]])
                    cfg, input, opts = self.makeCmd(copyStep)
                else:                        
                    cfg, input, opts = self.makeCmd(self.relvalModule.stepList[stepIndex][stepName])

                if input and cfg :
                    msg = "FATAL ERROR: found both cfg and input for workflow "+str(num)+' step '+stepName
                    raise MatrixException(msg)

                if (not input) and (stepIndex!=0) and (not '--filein' in opts):
                    if 'HARVESTING' in opts:
                        opts+=' --filein file:step%d_inDQM.root '%(stepIndex,)
                    else:
                        opts+=' --filein file:step%d.root '%(stepIndex,)
                if (not input) and (not 'fileout' in opts):
                    opts+=' --fileout file:step%d.root '%(stepIndex+1,)
                
                if cfg:
                    cmd  = 'cmsDriver.py '+cfg+' '+opts
                if stepIndex==0 and not inputInfo and input: # only if we didn't already set the input
                    inputInfo = input
                    # map input dataset to the one from the reference release:
                    inputInfo.dataSet = inputInfo.dataSet.replace('CMSSW_4_2_0_pre4', refRel)
                    cmd = 'DATAINPUT from '+inputInfo.dataSet+' on '+inputInfo.location
                    if input.run:
                        cmd+=' run %d'%(input.run)
                    from Configuration.PyReleaseValidation.relval_steps import InputInfoNDefault
                    if input.events!=InputInfoNDefault:
                        cmd+=' N %d'%(input.events)
                        
                if stepIndex > 0 and not 'cfg' in self.relvalModule.stepList[stepIndex][stepName]:
                    cmd  = 'cmsDriver.py step'+str(stepIndex+1)+' '+opts
    
                stepCmds[stepIndex] = cmd
                stepIndex += 1

            self.step1WorkFlows[(float(num),prefix)] = (str(float(num)), name, stepCmds[0], stepCmds[1], stepCmds[2], stepCmds[3], inputInfo)

        return

    def showRaw(self, useInput, refRel='CMSSW_4_2_0_pre2', fromScratch=None, what='all',step1Only=False):

        for matrixFile in self.files:

            self.reset(what)

            if self.what != 'all' and self.what not in matrixFile:
                print "ignoring non-requested file",matrixFile
                continue

            try:
                self.readMatrix(matrixFile, useInput, refRel, fromScratch)
            except Exception, e:
                print "ERROR reading file:", matrixFile, str(e)
                raise

            if not self.step1WorkFlows: continue

            dataFileName = matrixFile.replace('relval_', 'cmsDriver_')+'_hlt.txt'
            outFile = open(dataFileName,'w')

            print "found ", len(self.step1WorkFlows.keys()), ' workflows for ', dataFileName
            ids = self.step1WorkFlows.keys()
            ids.sort()
            stepCmds = ['','','','']
            for key in ids:
                #trick to skip the HImix IB test
                if key[0]==203.1 or key[0]==204.1 or key[0]==205.1: continue;
                num, name, stepCmds[0], stepCmds[1], stepCmds[2], stepCmds[3], inputInfo = self.step1WorkFlows[key]
                wfName,stepNames= name.split('+',1)
                stepNames=stepNames.replace('+RECODFROMRAWRECO','')
                stepNames=stepNames.replace('+SKIMCOSD','')
                stepNames=stepNames.replace('+SKIMD','')
                stepNames=stepNames.replace('+HARVESTD','')
                stepNames=stepNames.replace('+HARVEST','')
                otherSteps = None
                if '+' in stepNames:
                    step1,otherSteps = stepNames.split('+',1)
                line = num + ' ++ '+ wfName 
                if otherSteps and not step1Only:
                    line += ' ++ ' +otherSteps.replace('+',',')
                else:
                    line += ' ++ none' 
                if inputInfo :
                    #skip the samples from INPUT when step1Only is on
                    if step1Only: continue
                    line += ' ++ REALDATA: '+inputInfo.dataSet
                    if inputInfo.run!=0: line += ', RUN:'+str(inputInfo.run)
                    line += ', FILES: ' +str(inputInfo.files)
                    line += ', EVENTS: '+str(inputInfo.events)
                    if inputInfo.label!='':
                        line += ', LABEL: ' +inputInfo.label
                    line += ', LOCATION:'+inputInfo.location
                    line += ' @@@'
                else:
                    line += ' @@@ '+stepCmds[0]
                line=line.replace('DQMROOT','DQM')
                print line
                outFile.write(line+'\n')

            outFile.write('\n'+'\n')
            if step1Only: continue
            
            for stepName in self.relvalModule.step2.keys():
                cfg,input,cmd = self.makeCmd(self.relvalModule.step2[stepName])
                if 'dbsquery.log' in cmd: continue
                line = 'STEP2 ++ ' +stepName + ' @@@ cmsDriver.py step2 ' +cmd
                line=line.replace('DQMROOT','DQM')
                print line
                outFile.write(line+'\n')
                
            outFile.write('\n'+'\n')
            for stepName in self.relvalModule.step3.keys():
                cfg,input,cmd = self.makeCmd(self.relvalModule.step3[stepName])
                if 'dbsquery.log' in cmd: continue
                line ='STEP3 ++ ' +stepName + ' @@@ cmsDriver.py step3 ' +cmd
                line=line.replace('DQMROOT','DQM')
                print line
                outFile.write(line+'\n')
                
            outFile.write('\n'+'\n')
            for stepName in self.relvalModule.step4.keys():
                cfg,input,cmd = self.makeCmd(self.relvalModule.step4[stepName])
                if 'dbsquery.log' in cmd: continue
                line = 'STEP4 ++ ' +stepName + ' @@@ cmsDriver.py step4 ' +cmd
                line=line.replace('DQMROOT','DQM')
                print line
                outFile.write(line+'\n')
                
            outFile.close()

        
        return

    def showWorkFlows(self, selected=None):

        maxLen = 100 # for summary, limit width of output
        fmt1   = "%-6s %-35s [1]: %s ..."
        fmt2   = "       %35s [%d]: %s ..."
        print "\nfound a total of ", len(self.workFlows), ' workflows:'
        if selected:
            print "      of which the following", len(selected), 'were selected:'
        #-ap for now:
        maxLen = -1  # for individual listing, no limit on width
        fmt1   = "%-6s %-35s [1]: %s " 
        fmt2   = "       %35s [%d]: %s"

        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        for wf in self.workFlows:
            if selected and float(wf.numId) not in selected: continue
            print ''
            n1+=1
            wfName, stepNames = wf.nameId.split('+',1)
            print fmt1 % (wf.numId, stepNames, (wf.cmdStep1+' ')[:maxLen])
            if wf.cmdStep2:
                n2+=1
                print fmt2 % ( ' ', 2, (wf.cmdStep2+' ')[:maxLen])
                if wf.cmdStep3:
                    n3+=1
                    print fmt2 % ( ' ', 3, (wf.cmdStep3+' ')[:maxLen])
                    if wf.cmdStep4:
                        n4+=1
                        print fmt2 % ( ' ', 4, (wf.cmdStep4+' ')[:maxLen])

        print n1, 'workflows for step1,'
        print n2, 'workflows for step1 + step2,'
        print n3, 'workflows for step1 + step2 + step3'
        print n4, 'workflows for step1 + step2 + step3 + step4'

        return
    
    def createWorkFlows(self, fileNameIn):

        prefixIn = self.filesPrefMap[fileNameIn]

        # get through the list of items and update the requested workflows only
        keyList = self.step1WorkFlows.keys()
        ids = []
        for item in keyList:
            id, pref = item
            if pref != prefixIn : continue
            ids.append( float(id) )
        ids.sort()
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        for key in ids:
            val = self.step1WorkFlows[(key,prefixIn)]
            num, name, cmd, step2, step3, step4, inputInfo = val
            nameId = num+'_'+name
            if nameId in self.nameList.keys():
                print "==> duplicate name found for ", nameId
                print '    keeping  : ', self.nameList[nameId]
                print '    ignoring : ', val
            else:
                self.nameList[nameId] = val

            cmd2 = None
            cmd3 = None
            cmd4 = None
            
            n1 += 1

            if step2.lower() != '':
                n2 += 1
                cmd2 = step2
                if step3.lower() != '':
                    n3 += 1
                    cmd3 = step3
                    if step4.lower() != '':
                        n4 += 1
                        cmd4 = step4
                    #print '\tstep3 : ', self.step3WorkFlows[step3]
            self.workFlows.append( WorkFlow(num, name, cmd, cmd2, cmd3, cmd4, inputInfo) )
        return

    def prepare(self, useInput=None, refRel='', fromScratch=None):
        
        for matrixFile in self.files:
            if self.what != 'all' and self.what not in matrixFile:
                print "ignoring non-requested file",matrixFile
                continue

            try:
                self.readMatrix(matrixFile, useInput, refRel, fromScratch)
            except Exception, e:
                print "ERROR reading file:", matrixFile, str(e)
                raise
            
            try:
                self.createWorkFlows(matrixFile)
            except Exception, e:
                print "ERROR creating workflows :", str(e)
                raise
            
                
    def show(self, selected=None):    
        # self.showRaw()
        self.showWorkFlows(selected)
        print '\n','-'*80,'\n'


    def updateDB(self):

        import pickle
        pickle.dump(self.workFlows, open('theMatrix.pkl', 'w') )

        return

