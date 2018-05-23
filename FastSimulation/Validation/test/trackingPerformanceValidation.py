#! /usr/bin/env python

import os
import sys
import fileinput
import string

##########################################################################
##########################################################################
######### User variables

#Reference release
RefRelease='CMSSW_3_3_0_pre5'

#Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_3_3_0_pre6'

# FastSim flags:
NewFastSim=True
RefFastSim=False

# startup and ideal sample list

#This are the standard relvals (startup)
#startupsamples= ['RelValTTbar', 'RelValMinBias', 'RelValQCD_Pt_3000_3500']
#startupsamples= ['RelValTTbar']  # FASTSIM

#This is pileup sample
#startupsamples= ['RelValTTbar_Tauola']

#to skip startup samples:
startupsamples= []

#This are the standard relvals (ideal)
#idealsamples= ['RelValSingleMuPt1', 'RelValSingleMuPt10', 'RelValSingleMuPt100', 'RelValSinglePiPt1', 'RelValSinglePiPt10', 'RelValSinglePiPt100', 'RelValSingleElectronPt35', 'RelValTTbar', 'RelValQCD_Pt_3000_3500','RelValMinBias']
#idealsamples= ['RelValSinglePiPt1']  # FASTSIM
#idealsamples= ['RelValSinglePiPt1','RelValSingleMuPt10']
idealsamples= ['RelValSingleMuPt10']

#This is pileup sample
#idealsamples= ['RelValZmumuJets_Pt_20_300_GEN']

#summer09 preproduction (the character '-' must be avoided)
#idealsamples= ['InclusiveMu5_Pt250__Summer09', 'InclusiveMu5_Pt50__Summer09', 'MinBias_herwig__Summer09', 'TTbar__Summer09']

#to skip ideal samples:
#idealsamples= []



# track algorithm name and quality. Can be a list.
#Algos= ['ootb', 'initialStep', 'lowPtTripletStep']
Algos= ['ootb', 'initialStep', 'lowPtTripletStep','pixelPairStep','detachedTripletStep','mixedTripletStep','pixelLessStep']
#Qualities=['']
Qualities=['', 'highPurity']

#Leave unchanged unless the track collection name changes
Tracksname=''

# Sequence. Possible values:
#   -only_validation
#   -re_tracking
#   -digi2track
#   -only_validation_and_TP
#   -re_tracking_and_TP
#   -digi2track_and_TP
#   -harvesting
#   -preproduction
#   -comparison_only

#Sequence='preproduction'
Sequence='comparison_only'


# Ideal and Statup tags
IdealTag='MC_31X_V9'
StartupTag='STARTUP31X_V8'
if (NewFastSim):
    IdealTag+='_FastSim'
    StartupTag+='FastSim'
IdealTagRef='MC_31X_V8'
StartupTagRef='STARTUP31X_V7'
if (RefFastSim):
    IdealTagRef+='_FastSim'
    StartupTagRef+='FastSim'

# PileUp: PU . No PileUp: noPU
PileUp='_noPU'

ReferenceSelection=IdealTagRef+PileUp
StartupReferenceSelection=StartupTagRef+PileUp

if (NewFastSim):
    NewFormat='GEN-SIM-DIGI-RECO'
else:
    NewFormat='GEN-SIM-RECO'

# Default label is GlobalTag_noPU__Quality_Algo. Change this variable if you want to append an additional string.
NewSelectionLabel=''
#NewSelectionLabel='test2_logpt'


#Reference and new repository
RefRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance'
NewRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance'

castorHarvestedFilesDirectory='/castor/cern.ch/user/n/nuno/relval/harvest/'
#for preproduction samples:
#RefRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance/preproduction'
#NewRepository = '/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance/preproduction'

#Default Nevents
defaultNevents ='-1'

#Put here the number of event to be processed for specific samples (numbers must be strings) if not specified is defaultNevents:
Events={}
#Events={'RelValTTbar':'100'}

# template file names. Usually should not be changed.
cfg='trackingPerformanceValidation_cfg.py'
macro='macro/TrackValHistoPublisher.C'



#########################################################################
#########################################################################
############ Functions

def replace(map, filein, fileout):
    replace_items = map.items()
    while True:
        line = filein.readline()
        if not line: break
        for old, new in replace_items:
            line = string.replace(line, old, new)
        fileout.write(line)
    fileout.close()
    filein.close()
    
############################################

    
def do_validation(samples, GlobalTag, trackquality, trackalgorithm):
    global Sequence, RefSelection, RefRepository, NewSelection, NewRepository, defaultNevents, Events, castorHarvestedFilesDirectory
    global cfg, macro, Tracksname
    print 'Tag: ' + GlobalTag
    tracks_map = { 'ootb':'general_AssociatorByHitsRecoDenom','initialStep':'cutsRecoZero_AssociatorByHitsRecoDenom','lowPtTripletStep':'cutsRecoFirst_AssociatorByHitsRecoDenom','pixelPairStep':'cutsRecoSecond_AssociatorByHitsRecoDenom','detachedTripletStep':'cutsRecoThird_AssociatorByHitsRecoDenom','mixedTripletStep':'cutsRecoFourth_AssociatorByHitsRecoDenom','pixelLessStep':'cutsRecoFifth_AssociatorByHitsRecoDenom'}
    tracks_map_hp = { 'ootb':'cutsRecoHp_AssociatorByHitsRecoDenom','initialStep':'cutsRecoZeroHp_AssociatorByHitsRecoDenom','lowPtTripletStep':'cutsRecoFirstHp_AssociatorByHitsRecoDenom','pixelPairStep':'cutsRecoSecondHp_AssociatorByHitsRecoDenom','detachedTripletStep':'cutsRecoThirdHp_AssociatorByHitsRecoDenom','mixedTripletStep':'cutsRecoFourthHp_AssociatorByHitsRecoDenom','pixelLessStep':'cutsRecoFifthHp_AssociatorByHitsRecoDenom'}
    if(trackalgorithm=='initialStep' or trackalgorithm=='ootb'):
        mineff='0.80'
        maxeff='1.01'
        maxfake='0.7'
    else:
        mineff='0.0'
        maxeff='1.0'
        maxfake='0.8'
    #build the New Selection name
    NewSelection=GlobalTag + '_' + PileUp
    if (NewFastSim):
        NewSelection+='_FastSim'
    if( trackquality !=''):
        NewSelection+='_'+trackquality
    if(trackalgorithm!=''and not(trackalgorithm=='ootb' and trackquality !='')):
        NewSelection+='_'+trackalgorithm
    if(trackquality =='') and (trackalgorithm==''):
        if(Tracksname==''):
            NewSelection+='_ootb'
            Tracks='generalTracks'
        else:
           NewSelection+= Tracks
    if(Tracksname==''):
        Tracks='cutsRecoTracks'
    else:
        Tracks=Tracksname
    NewSelection+=NewSelectionLabel
    listofdatasets = open('listofdataset.txt' , 'w' )
    #loop on all the requested samples
    for sample in samples :
        templatecfgFile = open(cfg, 'r')
        templatemacroFile = open(macro, 'r')
#        newdir=NewRepository+'/'+NewRelease+'/'+NewSelection+'/'+sample 
        newdir=NewRelease+'/'+NewSelection+'/'+sample 
	cfgFileName=sample+GlobalTag
        #check if the sample is already done
        if(os.path.isfile(newdir+'/building.pdf' )!=True):    

            if( Sequence=="harvesting"):
            	harvestedfile='./DQM_V0001_R000000001__' + GlobalTag+ '__' + sample + '__Validation.root'
            elif( Sequence=="preproduction"):
                harvestedfile='./DQM_V0001_R000000001__' + sample+ '-' + GlobalTag + '_preproduction_312-v1__'+NewFormat+'_1.root'
            elif( Sequence=="comparison_only"):
                harvestedfile='./DQM_V0001_R000000001__' + sample+ '__' + NewRelease+ '-' +GlobalTag + '-v1__'+NewFormat+'.root'
                if(os.path.exists(harvestedfile)==False):
                    cpcmd='rfcp '+ castorHarvestedFilesDirectory+ NewRelease +'/' + harvestedfile + ' .'
                    returncode=os.system(cpcmd)
                    if (returncode!=0):
                        print 'copy of harvested file from castor for sample ' + sample + ' failed'
                        print 'try backup repository...'
                        NewCondition='MC'
                        if (NewFastSim):
                            NewCondition=NewCondition+'_FSIM'
                        cpcmd='rfcp /castor/cern.ch/user/a/aperrott/ValidationRecoMuon/'+NewRelease+'_'+NewCondition+'_'+sample+'_val.'+sample+'.root ./'+harvestedfile
                        returncode=os.system(cpcmd)
                        if (returncode!=0):
                            continue

            print ' Harvested file : '+harvestedfile
            
            #search the primary dataset
            if( Sequence!="comparison_only"):
                print 'Get information from DBS for sample', sample
                cmd='dbsql "find  dataset where dataset like *'
                #            cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*GEN-SIM-DIGI-RAW-HLTDEBUG-RECO* "'
                cmd+=sample+'/'+NewRelease+'_'+GlobalTag+'*'+NewFormat+'* order by dataset.createdate "'
                cmd+='|grep '+sample+'|grep -v test|tail -1'
                print cmd
                dataset= os.popen(cmd).readline().strip()
                print 'DataSet:  ', dataset, '\n'
                
                #Check if a dataset is found
                if dataset!="":
                    listofdatasets.write(dataset)
                    #Find and format the list of files
                    cmd2='dbsql "find file where dataset like '+ dataset +'"|grep ' + sample
                    filenames='import FWCore.ParameterSet.Config as cms\n'
                    filenames+='readFiles = cms.untracked.vstring()\n'
                    filenames+='secFiles = cms.untracked.vstring()\n'
                    filenames+='source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)\n'
                    filenames+='readFiles.extend( [\n'
                    first=True
                    print cmd2
                    for line in os.popen(cmd2).readlines():
                        filename=line.strip()
                        if first==True:
                            filenames+="'"
                            filenames+=filename
                            filenames+="'"
                            first=False
                        else :
                            filenames+=",\n'"
                            filenames+=filename
                            filenames+="'"
                    filenames+=']);\n'

                    # if not harvesting find secondary file names
                    if(Sequence!="preproduction"):
                            cmd3='dbsql  "find dataset.parent where dataset like '+ dataset +'"|grep ' + sample
                            parentdataset=os.popen(cmd3).readline()
                            print 'Parent DataSet:  ', parentdataset, '\n'

                    #Check if a dataset is found
                            if parentdataset!="":
                                    cmd4='dbsql  "find file where dataset like '+ parentdataset +'"|grep ' + sample 
                                    filenames+='secFiles.extend( [\n'
                                    first=True

                                    for line in os.popen(cmd4).readlines():
                                        secfilename=line.strip()
                                        if first==True:
                                            filenames+="'"
                                            filenames+=secfilename
                                            filenames+="'"
                                            first=False
                                        else :
                                            filenames+=",\n'"
                                            filenames+=secfilename
                                            filenames+="'"
                                    filenames+='\n ]);\n'
                            else :
                                    print "No primary dataset found skipping sample: ", sample
                                    continue
                    else :
                            filenames+='secFiles.extend( (               ) )'

                    cfgFile = open(cfgFileName+'.py' , 'w' )
                    cfgFile.write(filenames)

                    if ((sample in Events)!=True):
                            Nevents=defaultNevents
                    else:
                            Nevents=Events[sample]
                    thealgo=trackalgorithm
                    thequality=trackquality
                    if(trackalgorithm=='ootb'):
                        thealgo=''
                    if(thealgo!=''):
                        thealgo='\''+thealgo+'\''
                    if(trackquality!=''):
                        thequality='\''+trackquality+'\''
                    symbol_map = { 'NEVENT':Nevents, 'GLOBALTAG':GlobalTag, 'SEQUENCE':Sequence, 'SAMPLE': sample, 'ALGORITHM':thealgo, 'QUALITY':thequality, 'TRACKS':Tracks}


                    cfgFile = open(cfgFileName+'.py' , 'a' )
                    replace(symbol_map, templatecfgFile, cfgFile)
                    if(( (Sequence=="harvesting" or Sequence=="preproduction" or Sequence=="comparison_only") and os.path.isfile(harvestedfile) )==False):
                        # if the file is already harvested do not run the job again
                        cmdrun='cmsRun ' +cfgFileName+ '.py >&  ' + cfgFileName + '.log < /dev/zero '
                        retcode=os.system(cmdrun)
                    else:
                        retcode=0

                else:      
                    print 'No dataset found skipping sample: '+ sample, '\n'  
                    continue
            else:
                print ' Used sequence : '+Sequence
                retcode = 0
                
            if (retcode!=0):
                print 'Job for sample '+ sample + ' failed. \n'
            else:
                if (Sequence=="harvesting" or Sequence=="preproduction" or Sequence=="comparison_only"):
                    #copy only the needed histograms
                    if(trackquality==""):
                        rootcommand='root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map[trackalgorithm]+ '\\\"\\) >& /dev/null'
                        os.system(rootcommand)
                    elif(trackquality=="highPurity"):
                        rootcommand='root -b -q -l CopySubdir.C\\('+ '\\\"'+harvestedfile+'\\\",\\\"val.' +sample+'.root\\\",\\\"'+ tracks_map_hp[trackalgorithm]+ '\\\"\\) >& /dev/null'
                        os.system(rootcommand)


                referenceSample=RefRepository+'/'+RefRelease+'/'+RefSelection+'/'+sample+'/'+'val.'+sample+'.root'
                if os.path.isfile(referenceSample ):
                    replace_map = { 'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':RefRelease+'/'+RefSelection+'/val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':RefRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':RefSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': cfgFileName, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}

                    if(os.path.exists(RefRelease+'/'+RefSelection)==False):
                        os.makedirs(RefRelease+'/'+RefSelection)
                        os.system('cp ' + referenceSample+ ' '+RefRelease+'/'+RefSelection)  
                else:
#                    print "No reference file found at: ", RefRelease+'/'+RefSelection
                    print "No reference file found at: ", referenceSample
                    replace_map = { 'NEW_FILE':'val.'+sample+'.root', 'REF_FILE':'val.'+sample+'.root', 'REF_LABEL':sample, 'NEW_LABEL': sample, 'REF_RELEASE':NewRelease, 'NEW_RELEASE':NewRelease, 'REFSELECTION':NewSelection, 'NEWSELECTION':NewSelection, 'TrackValHistoPublisher': cfgFileName, 'MINEFF':mineff, 'MAXEFF':maxeff, 'MAXFAKE':maxfake}


                macroFile = open(cfgFileName+'.C' , 'w' )
                replace(replace_map, templatemacroFile, macroFile)


                os.system('root -b -q -l '+ cfgFileName+'.C'+ '>  macro.'+cfgFileName+'.log')
                

                if(os.path.exists(newdir)==False):
                    os.makedirs(newdir)

                print "moving pdf files for sample: " , sample
                os.system('mv  *.pdf ' + newdir)

                print "moving root file for sample: " , sample
                os.system('mv val.'+ sample+ '.root ' + newdir)

                print "copy py file for sample: " , sample
                os.system('cp '+cfgFileName+'.py ' + newdir)
	
	
        else:
            print 'Validation for sample ' + sample + ' already done. Skipping this sample. \n'




##########################################################################
#########################################################################
######## This is the main program
if(NewRelease==''): 
    try:
        #Get some environment variables to use
        NewRelease     = os.environ["CMSSW_VERSION"]
        
    except KeyError:
        print >>sys.stderr, 'Error: The environment variable CMSSW_VERSION is not available.'
        print >>sys.stderr, '       Please run cmsenv'
        sys.exit()
else:
    try:
        #Get some environment variables to use
        os.environ["CMSSW_VERSION"]
        
    except KeyError:
        print >>sys.stderr, 'Error: CMSSW environment variables are not available.'
        print >>sys.stderr, '       Please run cmsenv'
        sys.exit()



NewSelection=''

for algo in Algos:
    for quality in Qualities:
        RefSelection=ReferenceSelection
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(idealsamples, IdealTag, quality , algo)
        RefSelection=StartupReferenceSelection
        if( quality !=''):
            RefSelection+='_'+quality
        if(algo!=''and not(algo=='ootb' and quality !='')):
            RefSelection+='_'+algo
        if(quality =='') and (algo==''):
            RefSelection+='_ootb'
        do_validation(startupsamples, StartupTag, quality , algo)
        
