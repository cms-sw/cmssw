#!/usr/bin/env python

#########################################################################
#
# submit_OHltRateEff.py
#
#
# Usage: ./submit_OHltRateEff.py --help
#
#  Submit to condor:
#     ./submit_RateEff.py -d filename.cfg 
#
# Author: Michael Luk, Oct 2011
#
#########################################################################

import os, sys, glob
from optparse import OptionParser
_legend= '[OHltRateEff:]'

add_help_option = './submit_OHltRateEff -d <openhlt>.cfg'
parser = OptionParser(add_help_option)

parser.add_option("-d","--file",dest="cfgfile",default="file",type="string",
                  help="config file to parse to OHltTree", metavar="INFILE")

parser.add_option("-s","--submit",dest="submitjobs",default=False,
                  help="submit to condor?")

parser.add_option("-r","--run-locally",dest="runlocally",default=False,
                  help="run the whole thing locally... takes time")

parser.add_option("-m","--merge-results",dest="mergeresults",default="directory",type="string",
                  help="merge lumiscalefactor results into one text file")

print _legend,'parsing command line options...',
(options, args) = parser.parse_args()
print 'done'

#############################################################################
#
# Lumi Settings to Consider (in *e33 format) --
#
#####

lumiScaleFactor   = [3,4,5]    #[3,4,5] can just list more , this is used in merging too!   # currently 3 is the most that fits on a page... 
doPreScales       = True       # will we do the different prescale loop as well - in the #* [ PS1, PS2 ,...] on the same line as original 


# important for lsf batch submission only
currentRelease = "CMSSW_4_2_9_HLT1_hltpatch2"
dirRelease     = "/afs/cern.ch/user/m/mmhl/scratch0/trigger/"
submitqueue    = "8nh"     # if run in batch, then which q? 8nh is good  or 8nm if testing

## merge prescales? If false, will merge the lumiScaleFactors
mergePreScale   = False
doPreScaleQ     = 2          # how many prescales to merge | has to be <= the number of prescales run over... 
PreScaleLumiI   = 0          # which lumi? i.e. in order [0,1,2...], 0 will select the first lumiScaleFactor originally specified

#############################################################################
# can set the following manually here or at terminal line
#####

simulate          = True      # generates test jobs with 1000ev  per LS NB -- leave true for running locally  (using submit options will override this eventually...)

# one of these should be set to true to run the code
runlocally        = False     # run all events in series on local computer (makes simulate redundent) ./submit_OHltRateEff.py -d <OHlt_cfgfile>.cfg -r True/False terminal override
submitjobs        = True      # if both true, it will submit - overriden by -s True 


#############################################################################
#
# Setup -- Shouldn't need to touch anything else below here
#
####################

# don't run or merge yet
runcode   = False
mergelogs = False
nEv        = -1
nPrint     = 10000

#this is the directory and filename of the cfg to run or merge
if options.mergeresults == "directory" and options.cfgfile=="file":
    print _legend+" a file or folder needs to be inputted"
    print _legend+" submit_OHltRateEff.py -h for details"
    sys.exit()

if options.mergeresults != "directory":
    print "No cfg File Inputted - You should only see this if merging result files"
    directomerge = options.mergeresults
    runcode = False
    mergelogs    = True
    
if "file" != options.cfgfile and mergelogs == False :
    config = open(options.cfgfile)
    runcode= True

if (options.submitjobs or submitjobs) and mergelogs==False:
    runcode   = True
    submitjobs= True
    runlocally= False

#if run locally - save the output
if (runlocally or options.runlocally) and mergelogs==False:
    runcode    = True
    submitjobs = False

#simulate, to check procedure
if simulate:
    nEv    = 1000
    nPrint = 100
    
    
#################################################
#
# Main Code
#
#################################################
if runcode:
    os.system("echo "+_legend+" LumiScaleFactors to run : "+str(lumiScaleFactor))
    os.system("echo "+_legend+" Going to run ...        : "+str(nEv)+" ev per ScaleFactor")
    from datetime import datetime
    d = datetime.now()
    dt = d.strftime("%d-%b-%Y_%H-%M-%S")
    _dir = str(options.cfgfile).replace("/","_")
    _dir = _dir.replace(".cfg","")
    _dir = _dir+dt
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    os.system('mkdir results/'+_dir)
    os.system("echo "+_legend+" the contents are in ...   : ./results/"+_dir)
    
    os.system("cp "+options.cfgfile+" ./results/"+_dir+"/temporaryOHlt_template.cfg")
    filecounter=-1

    for i in lumiScaleFactor:      
        psCounter   = 0 # will store the max number of different prescale factors
        filecounter = filecounter + 1
        tempOhltout = open('./results/'+_dir+'/tempOHlt_cfg_'+str(filecounter)+'_sf_0_psf.cfg','w')
        tempOhltin  = open('./results/'+_dir+'/temporaryOHlt_template.cfg')
        
        for line in tempOhltin:
            line=line.replace('nEntries','nEntries                  = '+str(nEv)+'; ##')
            line=line.replace('nPrintStatusEvery','nPrintStatusEvery= '+str(nPrint)+'; ##')
            line=line.replace('lumiScaleFactor','lumiScaleFactor    = '+str(i/3.)+'; ##')
            #need counter of how many prescales
            if "#*" in line and "[" in line and "]" in line:
                psBraceB = line.find('[')
                psBraceE = line.find(']')
                if line[psBraceB:psBraceE].count(',') > psCounter:
                    psCounter = line[psBraceB:psBraceE].count(',')

            tempOhltout.write(line)
        tempOhltout.close()
        tempOhltin.close()
        #in loop over the lumiscalefactors - but we want to do multiple prescale factors for each
        psCounter += 1 #i.e. one more prescale factor than commas
        for psf in range(0,psCounter): # note the one extra since there are 1 more sf than commas
            tempOhltoutPS = open('./results/'+_dir+'/tempOHlt_cfg_'+str(filecounter)+'_sf_'+str(psf+1)+'_psf.cfg','w')
            tempOhltinPS  = open('./results/'+_dir+'/tempOHlt_cfg_'+str(filecounter)+'_sf_0_psf.cfg')

            for line in tempOhltinPS:                
                if "#*" in line and "[" in line and "]" in line:
                    psBraceB = line.find('[')
                    psBraceE = line.find(']')

                    #these next few lines give the start and end point to replace bw CB is beginning and CE is end. Format is ("HLT Name ","L1 Name", int rate <-to change, ... 
                    psBraceCE = line.find(',')
                    psBraceCE+= 1
                    psBraceCB = line[psBraceCE:].find(',')
                    psBraceCB+= psBraceCE +1
                    #psBraceCE = line[psBraceCB:].find(',')
                    if line[:].find(')') < psBraceCE+line[psBraceCE:].find(','):
                        psBraceCB = psBraceCE
                        psBraceCE = line[:].find(')')

                    else:
                        psBraceCE = line[psBraceCB:].find(',')
                        psBraceCE+= psBraceCB

                    #need to fi this down to simulate
                    if line[psBraceB:psBraceE].count(',')+1 > psf:
                        if psf==0:
                            psBraceM = line[psBraceB+1:psBraceE].find(',') #this only finds the first one
                            if psBraceM > 0:
                                psBraceM+= psBraceB + 1
                            else:
                                psBraceM = psBraceE  
                      
                            line = line[:psBraceCB]+line[psBraceB+1:psBraceM]+line[psBraceCE:]

                        else:
                            for l in range(1,psf+1): #psf is the number of entries to changex
                                psBraceM = line[psBraceB+1:psBraceE].find(',')
                                psBraceM += psBraceB+1 # this is of first comma
                                psBracetemp = line[psBraceM+1:psBraceE].find(',') #of 2nd comma
                                if psBracetemp > 0:
                                    psBraceB     = psBraceM
                                    psBracetempB = psBracetemp+psBraceM+1
                                    
                            if psBracetemp > 0:
                                psBraceE = psBraceM+1+psBracetemp

                            line = line[:psBraceCB]+line[psBraceM+1:psBraceE]+line[psBraceCE:]
                tempOhltoutPS.write(line)

        if runlocally or options.runlocally:
            print _legend," running "+str(len(lumiScaleFactor))+" lumiScaleFactors - "+str(filecounter) +"/"+str(len(lumiScaleFactor))
            os.system("./OHltRateEff ./results/"+_dir+"/tempOHlt_cfg_"+str(filecounter)+"_sf_0_psf.cfg > results/"+_dir+"/tempOHlt_log_"+str(filecounter)+"_sf_0_psf.log")
             
            if doPreScales:
                print _legend+" will run n different prescale settings n = "+str(psCounter)
                for psf in range(1,psCounter):
                    print _legend," running "+str(len(lumiScaleFactor))+" lumiScaleFactors && "+str(psCounter)+" prescaleFactors - "+str(filecounter*psCounter+psf+1) +"/"+str(len(lumiScaleFactor)*psCounter)
                    
                    os.system("./OHltRateEff ./results/"+_dir+"/tempOHlt_cfg_"+str(filecounter)+"_sf_"+str(psf)+"_psf.cfg > results/"+_dir+"/tempOHlt_log_"+str(filecounter)+"_sf_"+str(psf)+"_psf.log")

    #########################
    #Sumbits to condor/lsf  -- THIS IS NOT YET TESTED FULLY
    #########################
        else:
            if doPreScales == False:
                psCounter = 0
                
            tempsubmitfileP = {}

            print _legend+" will run n different prescale settings n = "+str(psCounter)

            os.system('cp OHltRateEff ./results/'+_dir)
            os.chdir ('./results/'+_dir)

            for psf in range(0,psCounter):
                tempsubmitfileP[psf] = open('tempSubmit'+str(filecounter)+'_'+str(psf)+'.job','w')
                os.system('chmod 755 tempSubmit'+str(filecounter)+'_'+str(psf)+'.job')
                tempsubmitfileP[psf].write("#!/bin/csh -f \n")
                tempsubmitfileP[psf].write("set nonomatch  \n")
                tempsubmitfileP[psf].write("setenv CMSSW_RELEASE "+currentRelease+"\nsetenv CODE_SRC "+dirRelease+"/" +currentRelease+" \n")
                tempsubmitfileP[psf].write("cd ${WORKDIR} \n scramv1 project CMSSW ${CMSSW_RELEASE} \n cd ${CMSSW_RELEASE}/src \n eval `scramv1 runtime -csh` \n ")
                tempsubmitfileP[psf].write("cp -rf ${CODE_SRC}/src/HLTrigger . \n  cd HLTrigger \n scram b \n cd HLTanalyzers/test/RateEff/  \n source setup.csh \n cd ./results/"+_dir+" \n")
                tempsubmitfileP[psf].write("./OHltRateEff tempOHlt_cfg_"+str(filecounter)+"_sf_"+str(psf)+"_psf.cfg > tempOHlt_log_"+str(filecounter)+"_sf_"+str(psf)+"_psf.log \n")
                tempsubmitfileP[psf].write("cp tempOHlt_log_"+str(filecounter)+"_sf_"+str(psf)+"_psf.log  ${CODE_SRC}/src/HLTrigger/HLTanalyzers/test/RateEff/results/"+_dir+"/tempOHlt_log_"+str(filecounter)+"_sf_"+str(psf)+"_psf.log \n")
                os.system("bsub -q "+submitqueue+" tempSubmit"+str(filecounter)+"_"+str(psf)+".job")
            os.chdir('../../')
                
                
#######################
#
# Merge the log files 
#
#######################
 
if mergelogs:
    filecounter = -1
    mergedlog = open('./merged_OHlt_log.tex','w')
    os.system("echo "+_legend+" creating merged_OHlt_log.tex in ./ folder ...")    
    mergedlog.write('\\documentclass[prb]{revtex4} \n \\usepackage[UKenglish]{babel}\n \\usepackage{rotating} \n  \\setlength{\\textheight}{270mm} \n    \\begin{document} \n')
    mergedlog.write('\\begin{sidewaystable}[ht] \n \\begin{tabular}{|c|')

    if mergePreScale:
        psChecker = len(glob.glob(directomerge+'/tempOHlt_log_'+str(PreScaleLumiI)+'_sf_*_psf.log'))
        if psChecker < doPreScaleQ:
            print _legend,"MERGE WARNING - Number of prescales to merge exceeds number available, setting to maximum..."
            doPreScaleQ = psChecker
        
        mergedlog.write('|')
        for i in range(0,doPreScaleQ):
            for j in range(0,3): #3 indicates how many columns each scaling needs,  Prescale, Indiv, Cumulative
                mergedlog.write("c|")
            mergedlog.write("|")
    else:
        sfChecker = len(glob.glob(directomerge+'/tempOHlt_log_*_sf_0_psf.log'))
        if sfChecker < len(lumiScaleFactor):
            print _legend,"MERGE ERROR - Number of scale factors to merge exceeds total avaiable, exiting..."
            sys.exit()
        
        mergedlog.write('c||')
        for i in lumiScaleFactor:
            for j in range(0,2): #2 indicates how many columns each scaling needs,  Indiv, Cumulative
                mergedlog.write("c|")
            mergedlog.write("|")

    mergedlog.write('} \n \\hline \n')
   
    mergedlog.write("Name ")
    if mergePreScale:
        for i in range(0,doPreScaleQ):
            mergedlog.write("& ("+str(i)+") Prescale (HLT*L1) & ("+str(i)+") Indiv. ["+str(lumiScaleFactor[PreScaleLumiI])+"e33] & ("+str(i)+") Cumul. ["+str(lumiScaleFactor[PreScaleLumiI])+"e33]")
        
    else:
        mergedlog.write(" & Prescale (HLT*L1)")
        fi=-1
        for i in lumiScaleFactor:
            fi+=1
            mergedlog.write(" & ("+str(fi)+") Indiv. ["+str(i)+"e33] & ("+str(fi)+") Cumul. ["+str(i)+"e33]")

    mergedlog.write("\n \\\\ \n \\hline \n")

    if mergePreScale:
        lineStore = [""]*3*doPreScaleQ
        savedline   =[""]*doPreScaleQ
        savetotal   =[""]*doPreScaleQ
        savedlineno =[-1]*doPreScaleQ
    else:
        lineStore = [""]*3*len(lumiScaleFactor)
        savedline   =[""]*len(lumiScaleFactor)
        savetotal   =[""]*len(lumiScaleFactor)
        savedlineno =[-1]*len(lumiScaleFactor)
    filecounter =-1
    linecounter =-1
    reachEnd = False
  
    for n in range(0,5000): #number of triggers...a little rough but ok, will automatically break at end
        filecounter = -1
        linecounter = 0        
        
        if mergePreScale:
            numfiles = doPreScaleQ
        else:
            numfiles = len(lumiScaleFactor)

        for i in range(0,numfiles):
            boLog        = False
            boMenu       = False
            linecounter =-1
            filecounter+= 1
            
            if mergePreScale:
                mergedinput = open("./"+directomerge+"/tempOHlt_log_"+str(PreScaleLumiI)+"_sf_"+str(filecounter)+"_psf.log") 
            else:
                mergedinput = open("./"+directomerge+"/tempOHlt_log_"+str(filecounter)+"_sf_0_psf.log") 
            
            for line in mergedinput:
                linecounter +=1
                if "TOTAL RATE" in line:
                    reachEnd = True
                    savetotal[filecounter]=line
                    break
            
                if boMenu:
                    if linecounter > savedlineno[filecounter]:
                        savedline[filecounter]= line
                        savedlineno[filecounter] = linecounter
                        linecounter = -1 
                        boMenu= False
                        boLog = False
                        break
                    
                if "Trigger Rates" in line:
                    boLog = True
                if boLog:
                    if "-----------------------------" in line:
                        boMenu = True
        if reachEnd:
            break

        filecounter=-1
        for l in savedline:
            filecounter+=1
            sname = line.find("(")
            rscale = line.find(")")
            if filecounter==0: 
                triggerName = line[:sname]
                triggerName = triggerName.replace("_","\_")
                mergedlog.write(triggerName) # the name of the trigger only once
                if not mergePreScale:
                    mergedlog.write(" & "+l[sname+1:rscale]) #then the prescale
            
            if mergePreScale:
                mergedlog.write(" & "+l[sname+1:rscale]) #then the prescale
                                
            rindiv = l.find("|")
            mergedlog.write(" & "+l[rscale+1:rindiv])   #pure rate
            rpur   = line[rindiv+1:].find("|")
            rpur   = rpur + rindiv +1
            mergedlog.write(" & "+l[rindiv+1:rpur])     #store the cumulative rate

        for le in lineStore:
            le = le.replace("+-","$\\pm$")
            mergedlog.write(le)
        mergedlog.write("  \\\\   \n ")

    mergedlog.write(" \hline  \n")
    mergedlog.write("\\end{tabular} \n")
    mergedlog.write('\\caption{Trigger Rates [Hz] :  ')

    fi=0
    for li in savetotal:
        li = li.replace("+-","$\\pm$")
        fi+=1
        if mergePreScale:
            mergedlog.write("("+str(fi-1)+")   "+str(lumiScaleFactor[PreScaleLumiI])+"e33  "+li+"     ")
        else:
            mergedlog.write("("+str(fi-1)+")   "+str(lumiScaleFactor[fi-1])+"e33  "+li+"     ")
    mergedlog.write('}  \n  \\end{sidewaystable}')
    mergedlog.write('\n \\end{document}')
    os.system("echo "+_legend +" file created in current dir... tex it!")
