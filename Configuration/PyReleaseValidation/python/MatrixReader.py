
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

        #a bunch of information, but not yet the WorkFlow object
        self.workFlowSteps = {}
        #the actual WorkFlow objects
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
                input = v #of type InputInfo
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
            commands=[]
            wfName = wfInfo[0]
            stepList = wfInfo[1]
            # if no explicit name given for the workflow, use the name of step1
            if wfName.strip() == '': wfName = stepList[0]
            # option to specialize the wf as the third item in the WF list
            addTo=None
            addCom=None
            if len(wfInfo)>=3:
                addCom=wfInfo[2]
                if not type(addCom)==list:   addCom=[addCom]
                #print 'added dict',addCom
                if len(wfInfo)>=4:
                    addTo=wfInfo[3]
                    #pad with 0
                    while len(addTo)!=len(stepList):
                        addTo.append(0)

            name=wfName
            for (stepIndex,step) in enumerate(stepList):
                stepName=step
                if len(name) > 0 : name += '+'
                #any step can be mirrored with INPUT
                ## maybe we want too level deep input
                if useInput and (str(num) in useInput or "all" in useInput):
                    if step+'INPUT' in self.relvalModule.steps.keys():
                        stepName = step+"INPUT"
                    if fromScratch and (str(num) in fromScratch or "all" in fromScratch):
                        msg = "FATAL ERROR: request for both fromScratch and input for workflow "+str(num)
                        raise MatrixException(msg)
                name += stepName
                if addCom and (not addTo or addTo[stepIndex]==1):
                    from Configuration.PyReleaseValidation.relval_steps import merge
                    copyStep=merge(addCom+[self.relvalModule.steps[stepName]])
                    cfg, input, opts = self.makeCmd(copyStep)
                else:
                    cfg, input, opts = self.makeCmd(self.relvalModule.steps[stepName])

                if input and cfg :
                    msg = "FATAL ERROR: found both cfg and input for workflow "+str(num)+' step '+stepName
                    raise MatrixException(msg)

                if input:
                    cmd = input
                else:
                    if cfg:
                        cmd  = 'cmsDriver.py '+cfg+' '+opts
                    else:
                        cmd  = 'cmsDriver.py step'+str(stepIndex+1)+' '+opts

                commands.append(cmd)

            self.workFlowSteps[(float(num),prefix)] = (str(float(num)), name, commands)
        
        return


    def showRaw(self, useInput, refRel='CMSSW_4_2_0_pre2', fromScratch=None, what='all',step1Only=False,selected=None):

        if selected:
            selected=map(float,selected)
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

            if not self.workFlowSteps: continue

            dataFileName = matrixFile.replace('relval_', 'cmsDriver_')+'_hlt.txt'
            outFile = open(dataFileName,'w')

            print "found ", len(self.workFlowSteps.keys()), ' workflows for ', dataFileName
            ids = self.workFlowSteps.keys()
            ids.sort()
            indexAndSteps=[]

            writtenWF=0
            for key in ids:
                if selected and not (float(key[0]) in selected):
                    continue
                #trick to skip the HImix IB test
                if key[0]==203.1 or key[0]==204.1 or key[0]==205.1 or key[0]==4.51 or key[0]==4.52: continue
                num, name, commands = self.workFlowSteps[key]
                wfName,stepNames= name.split('+',1)
                stepNames=stepNames.replace('+RECODFROMRAWRECO','')
                stepNames=stepNames.replace('+SKIMCOSD','')
                #stepNames=stepNames.replace('+SKIMD','')
                #stepNames=stepNames.replace('+HARVESTD','')
                #stepNames=stepNames.replace('+HARVEST','')
                otherSteps = None
                if '+' in stepNames:
                    step1,otherSteps = stepNames.split('+',1)
                
                line = num + ' ++ '+ wfName 
                if otherSteps and not step1Only:
                    line += ' ++ ' +otherSteps.replace('+',',')
                else:
                    line += ' ++ none'
                inputInfo=None
                if not isinstance(commands[0],str):
                    inputInfo=commands[0]
                if otherSteps:
                    for (i,c) in enumerate(otherSteps.split('+')):
                        #pad with set
                        for p in range(len(indexAndSteps),i+2):
                            indexAndSteps.append(set())
                        indexAndSteps[i+1].add(c)

                if inputInfo :
                    #skip the samples from INPUT when step1Only is on
                    if step1Only: continue
                    line += ' ++ REALDATA: '+inputInfo.dataSet
                    if inputInfo.run!=[]: line += ', RUN:'+','.join(map(str,inputInfo.run))
                    line += ', FILES: ' +str(inputInfo.files)
                    line += ', EVENTS: '+str(inputInfo.events)
                    if inputInfo.label!='':
                        line += ', LABEL: ' +inputInfo.label
                    line += ', LOCATION:'+inputInfo.location
                    line += ' @@@'
                else:
                    line += ' @@@ '+commands[0]
                line=line.replace('DQMROOT','DQM')
                writtenWF+=1
                outFile.write(line+'\n')


            outFile.write('\n'+'\n')
            if step1Only: continue

            for (index,s) in enumerate(indexAndSteps):
                for stepName in s:
                    stepIndex=index+1
                    cfg,input,cmd = self.makeCmd(self.relvalModule.steps[stepName])
                    if 'dbsquery.log' in cmd: continue
                    line = 'STEP%d ++ '%(stepIndex,) +stepName + ' @@@ cmsDriver.py step%d '%(stepIndex,) +cmd
                    line=line.replace('DQMROOT','DQM')
                    outFile.write(line+'\n')
                outFile.write('\n'+'\n')
            outFile.close()
            print "wrote ",writtenWF, ' workflow'+('s' if (writtenWF!=1) else ''),' to ', outFile.name
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

        N=[]
        for wf in self.workFlows:
            if selected and float(wf.numId) not in selected: continue
            print ''
            #pad with zeros
            for i in range(len(N),len(wf.cmds)):                N.append(0)
            N[len(wf.cmds)-1]+=1
            wfName, stepNames = wf.nameId.split('+',1)
            for i,s in enumerate(wf.cmds):
                if i==0:
                    print fmt1 % (wf.numId, stepNames, (str(s)+' ')[:maxLen])
                else:
                    print fmt2 % ( ' ', 2, (str(s)+' ')[:maxLen])

        for i,n in enumerate(N):
            if n:            print n,'workflows with',i+1,'steps'

        return
    
    def createWorkFlows(self, fileNameIn):

        prefixIn = self.filesPrefMap[fileNameIn]

        # get through the list of items and update the requested workflows only
        #keyList = self.step1WorkFlows.keys()
        keyList = self.workFlowSteps.keys()
        ids = []
        for item in keyList:
            id, pref = item
            if pref != prefixIn : continue
            ids.append( float(id) )
        ids.sort()
        for key in ids:
            val = self.workFlowSteps[(key,prefixIn)]
            num, name, commands = val
            nameId = num+'_'+name
            if nameId in self.nameList.keys():
                print "==> duplicate name found for ", nameId
                print '    keeping  : ', self.nameList[nameId]
                print '    ignoring : ', val
            else:
                self.nameList[nameId] = val

            self.workFlows.append(WorkFlow(num, name, commands=commands))

        return
    
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

        self.showWorkFlows(selected)
        print '\n','-'*80,'\n'


    def updateDB(self):

        import pickle
        pickle.dump(self.workFlows, open('theMatrix.pkl', 'w') )

        return

