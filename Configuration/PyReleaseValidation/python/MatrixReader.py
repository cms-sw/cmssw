
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

    def __init__(self, opt):

        self.reset(opt.what)

        self.wm=opt.wmcontrol
        self.addCommand=opt.command
        self.apply=opt.apply
        self.commandLineWf=opt.workflow
        self.overWrite=opt.overWrite

        self.noRun = opt.noRun
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
                             'relval_ged': 'ged-',
                             'relval_upgrade':'upg-',
                             'relval_identity':'id-'
                             }

        self.files = ['relval_standard' ,
                      'relval_highstats',
                      'relval_pileup',
                      'relval_generator',
                      'relval_production',
                      'relval_ged',
                      'relval_upgrade',
                      'relval_identity'                      
                      ]

        self.relvalModule = None
        
        return

    def makeCmd(self, step):

        cmd = ''
        cfg = None
        input = None
        for k,v in step.items():
            if 'no_exec' in k : continue  # we want to really run it ...
            if k.lower() == 'cfg':
                cfg = v
                continue # do not append to cmd, return separately
            if k.lower() == 'input':
                input = v 
                continue # do not append to cmd, return separately
            
            #chain the configs
            #if k.lower() == '--python':
            #    v = 'step%d_%s'%(index,v)
            cmd += ' ' + k + ' ' + str(v)
        return cfg, input, cmd
    
    def readMatrix(self, fileNameIn, useInput=None, refRel=None, fromScratch=None):
        
        prefix = self.filesPrefMap[fileNameIn]
        
        print "processing ", fileNameIn
        
        try:
            _tmpMod = __import__( 'Configuration.PyReleaseValidation.'+fileNameIn )
            self.relvalModule = sys.modules['Configuration.PyReleaseValidation.'+fileNameIn]
        except Exception, e:
            print "ERROR importing file ", fileNameIn, str(e)
            return

        print "request for INPUT for ", useInput

        
        fromInput={}
        
        if useInput:
            for i in useInput:
                if ':' in i:
                    (ik,il)=i.split(':')
                    if ik=='all':
                        for k in self.relvalModule.workflows.keys():
                            fromInput[float(k)]=int(il)
                    else:
                        fromInput[float(ik)]=int(il)
                else:
                    if i=='all':
                        for k in self.relvalModule.workflows.keys():
                            fromInput[float(k)]=0
                    else:
                        fromInput[float(i)]=0
                
        if fromScratch:
            fromScratch=map(float,fromScratch)
            for num in fromScratch:
                if num in fromInput:
                    fromInput.pop(num)
        #overwrite steps
        if self.overWrite:
            for p in self.overWrite:
                self.relvalModule.steps.overwrite(p)
        
        #change the origin of dataset on the fly
        if refRel:
            if ',' in refRel:
                refRels=refRel.split(',')
                if len(refRels)!=len(self.relvalModule.baseDataSetRelease):
                    return
                self.relvalModule.changeRefRelease(
                    self.relvalModule.steps,
                    zip(self.relvalModule.baseDataSetRelease,refRels)
                    )
            else:
                self.relvalModule.changeRefRelease(
                    self.relvalModule.steps,
                    [(x,refRel) for x in self.relvalModule.baseDataSetRelease]
                    )
            

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
            stepIndex=0
            ranStepList=[]

            #first resolve INPUT possibilities
            if num in fromInput:
                ilevel=fromInput[num]
                #print num,ilevel
                for (stepIr,step) in enumerate(reversed(stepList)):
                    stepName=step
                    stepI=(len(stepList)-stepIr)-1
                    #print stepIr,step,stepI,ilevel                    
                    if stepI>ilevel:
                        #print "ignoring"
                        continue
                    if stepI!=0:
                        testName='__'.join(stepList[0:stepI+1])+'INPUT'
                    else:
                        testName=step+'INPUT'
                    #print "JR",stepI,stepIr,testName,stepList
                    if testName in self.relvalModule.steps.keys():
                        #print "JR",stepI,stepIr
                        stepList[stepI]=testName
                        #pop the rest in the list
                        #print "\tmod prepop",stepList
                        for p in range(stepI):
                            stepList.pop(0)
                        #print "\t\tmod",stepList
                        break
                                                        
                                                    
            for (stepI,step) in enumerate(stepList):
                stepName=step
                if self.wm:
                    #cannot put a certain number of things in wm
                    if stepName in [
                        #'HARVEST','HARVESTD','HARVESTDreHLT',
                        'RECODFROMRAWRECO','SKIMD','SKIMCOSD','SKIMDreHLT'
                        ]:
                        continue
                    
                #replace stepName is needed
                #if stepName in self.replaceStep
                if len(name) > 0 : name += '+'
                #any step can be mirrored with INPUT
                ## maybe we want too level deep input
                """
                if num in fromInput:
                    if step+'INPUT' in self.relvalModule.steps.keys():
                        stepName = step+"INPUT"
                        stepList.remove(step)
                        stepList.insert(stepIndex,stepName)
                """    
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
                    if self.noRun:
                        cmd.run=[]
                else:
                    if cfg:
                        cmd  = 'cmsDriver.py '+cfg+' '+opts
                    else:
                        cmd  = 'cmsDriver.py step'+str(stepIndex+1)+' '+opts
                    if self.wm:
                        cmd+=' --io %s.io --python %s.py'%(stepName,stepName)
                    if self.addCommand:
                        if self.apply:
                            if stepIndex in self.apply or stepName in self.apply:
                                cmd +=' '+self.addCommand
                        else:
                            cmd +=' '+self.addCommand
                    if self.wm:
                        cmd=cmd.replace('DQMROOT','DQM')
                        cmd=cmd.replace('--filetype DQM','')
                commands.append(cmd)
                ranStepList.append(stepName)
                stepIndex+=1
                
            self.workFlowSteps[(num,prefix)] = (num, name, commands, ranStepList)
        
        return


    def showRaw(self, useInput, refRel=None, fromScratch=None, what='all',step1Only=False,selected=None):

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
                if selected and not (key[0] in selected):
                    continue
                #trick to skip the HImix IB test
                if key[0]==203.1 or key[0]==204.1 or key[0]==205.1 or key[0]==4.51 or key[0]==4.52: continue
                num, name, commands, stepList = self.workFlowSteps[key]
                
                wfName,stepNames= name.split('+',1)
                
                stepNames=stepNames.replace('+RECODFROMRAWRECO','')
                stepNames=stepNames.replace('+SKIMCOSD','')
                stepNames=stepNames.replace('+SKIMD','')
                if 'HARVEST' in stepNames:
                    #find out automatically what to remove
                    exactb=stepNames.index('+HARVEST')
                    exacte=stepNames.index('+',exactb+1) if ('+' in stepNames[exactb+1:]) else (len(stepNames))
                    stepNames=stepNames.replace(stepNames[exactb:exacte],'')
                otherSteps = None
                if '+' in stepNames:
                    step1,otherSteps = stepNames.split('+',1)
                
                line = str(num) + ' ++ '+ wfName 
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
                        indexAndSteps[i+1].add((c,commands[i+1]))

                if inputInfo :
                    #skip the samples from INPUT when step1Only is on
                    if step1Only: continue
                    line += ' ++ REALDATA: '+inputInfo.dataSet
                    if inputInfo.run!=[]: line += ', RUN:'+'|'.join(map(str,inputInfo.run))
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
                for (stepName,cmd) in s:
                    stepIndex=index+1
                    if 'dasquery.log' in cmd: continue
                    line = 'STEP%d ++ '%(stepIndex,) +stepName + ' @@@ '+cmd
                    line=line.replace('DQMROOT','DQM')
                    outFile.write(line+'\n')
                outFile.write('\n'+'\n')
            outFile.close()
            print "wrote ",writtenWF, ' workflow'+('s' if (writtenWF!=1) else ''),' to ', outFile.name
        return 
                    

    def showWorkFlows(self, selected=None, extended=True):
        if selected: selected = map(float,selected)
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
            if extended: print ''
            #pad with zeros
            for i in range(len(N),len(wf.cmds)):                N.append(0)
            N[len(wf.cmds)-1]+=1
            wfName, stepNames = wf.nameId.split('+',1)
            for i,s in enumerate(wf.cmds):
                if extended:
                    if i==0:
                        print fmt1 % (wf.numId, stepNames, (str(s)+' ')[:maxLen])
                    else:
                        print fmt2 % ( ' ', i+1, (str(s)+' ')[:maxLen])
                else:
                    print "%-6s %-35s "% (wf.numId, stepNames)
                    break
        print ''
        for i,n in enumerate(N):
            if n:            print n,'workflows with',i+1,'steps'

        return
    
    def createWorkFlows(self, fileNameIn):

        prefixIn = self.filesPrefMap[fileNameIn]

        # get through the list of items and update the requested workflows only
        keyList = self.workFlowSteps.keys()
        ids = []
        for item in keyList:
            id, pref = item
            if pref != prefixIn : continue
            ids.append(id)
        ids.sort()
        for key in ids:
            val = self.workFlowSteps[(key,prefixIn)]
            num, name, commands, stepList = val
            nameId = str(num)+'_'+name
            if nameId in self.nameList:
                print "==> duplicate name found for ", nameId
                print '    keeping  : ', self.nameList[nameId]
                print '    ignoring : ', val
            else:
                self.nameList[nameId] = val

            self.workFlows.append(WorkFlow(num, name, commands=commands))

        return

    def prepare(self, useInput=None, refRel='', fromScratch=None):
        
        for matrixFile in self.files:
            if self.what != 'all' and self.what not in matrixFile:
                print "ignoring non-requested file",matrixFile
                continue
            if self.what == 'all' and ('upgrade' in matrixFile):
                print "ignoring",matrixFile,"from default matrix"
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
            
                
    def show(self, selected=None, extended=True):    

        self.showWorkFlows(selected,extended)
        print '\n','-'*80,'\n'


    def updateDB(self):

        import pickle
        pickle.dump(self.workFlows, open('theMatrix.pkl', 'w') )

        return

