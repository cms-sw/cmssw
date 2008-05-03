#!/usr/bin/env python

#################################################
#
# DQMfromDBSgui.py
#
# v1.0
#
# by Jeff Temple (jtemple@fnal.gov)
#
# 2 May 2008
#
# GUI to automatically grab new runs from DBS
# and run HCAL DQM on any newly-found runs
#
#################################################

import sys
try:
    from Tkinter import *
except:
    print "Could not import Tkinter module!"
    print "Cannot display GUI!"
    print "(If you are running from FNAL, this is a known problem --\n"
    print " Tkinter isn't available in the SL4 release of python for some reason."
    print "This is being investigated.)"
    sys.exit()

import tkMessageBox # for displaying warning messages

import thread # used for automated checking; may not be necessary
import os
import time # use to determine when to check for files
import cPickle # stores file information
import python_dbs # use for "sendmessage" function to get info from DBS
import string
import helpfunctions

class DBSRun:
    '''
    Stores information about a given run
    (Run number, files in run, whether DQM has been performed on run.)
    '''

    def __init__(self,filelist=None):
        self.runnum=-1
        self.dataset=None
        self.files=[] # Stores files associated with run
        if (filelist<>None):
            self.files=filelist
        self.ignoreRun=0 # if true, DQM won't be performed on run
        self.startedDQM=0 # set to true once DQM started 
        self.finishedDQM=0 # set to true once DQM finished (DQM directory for run exists)
        self.maxEvents=50 # number of events to run over with DQM
        return


    def Print(self):
        x= "%10s     %55s     %10s%12s%15s%15s\n"%(self.runnum, self.dataset,
                                             len(self.files),self.ignoreRun,
                                             self.startedDQM,self.finishedDQM)
        
        #print x
        return x

def dbsSort(x,y):
    return x.runnum>y.runnum
        
########################################################################

class dbsAccessor:
    '''
    Class that stores values which are used when accessing DBS web page.
    Values stored as IntVars, StringVars, etc. so that they can be
    easily utilized by the main DQMDBSgui.
    '''

    def __init__(self):
        self.host=StringVar()
        self.port=IntVar()
        self.dbsInst=StringVar()
        self.input=StringVar() #specify input string
        self.page=IntVar()
        self.limit=IntVar()
        self.xml=BooleanVar()
        self.case=StringVar()
        self.details=BooleanVar()
        self.debug=BooleanVar()
        self.beginRun=IntVar()
        self.endRun=IntVar()
        
        self.searchResult=None

        self.host.set("cmsweb.cern.ch/dbs_discovery/")
        self.port.set(443)
        self.dbsInst.set("cms_dbs_prod_global")
        # Assume file takes the form */Global*/A/*/RAW/*.root
        self.input.set("find file where file=*/Global*/A/*/RAW/*.root and run=30625")
        self.page.set(0)
        self.limit.set(10000)
        self.xml.set(False)
        self.case.set("on")
        self.details.set(False)
        self.debug.set(False)
        self.beginRun.set(30625)
        self.endRun.set(30625)
        return


    def searchDBS(self,beginrun=-1, endrun=-1,mytext=None):

        # If beginrun, endrun specified, use their values in the search
        if (beginrun>-1):
            self.beginRun.set(beginrun)
        if (endrun>-1):
            self.endRun.set(endrun)
        if (mytext==None):
            # Assume dataset takes the form */Global*-A/*
            mytext="find file where file=*/Global*/A/*/RAW/*.root and run between %i-%i"%(self.beginRun.get(),self.endRun.get())

        self.input.set(mytext)
        self.searchResult=python_dbs.sendMessage(self.host.get(),
                                      self.port.get(),
                                      self.dbsInst.get(),
                                      self.input.get(),
                                      self.page.get(),
                                      self.limit.get(),
                                      self.xml.get(),
                                      self.case.get(),
                                      self.details.get(),
                                      self.debug.get())
        #print "Search = ",self.searchResult
        #print "Finished!"
        return
    

###################################################################

class DQMDBSgui:
    '''
    Main GUI Class
    '''
    
    def __init__(self, parent=None, debug=False):

        #self.basedir=os.path.abspath(os.curdir) # set this directory to a permanent location later

        checkCMSSW=os.popen2("echo $CMSSW_BASE")
        self.basedir=checkCMSSW[1].read()
        if len(self.basedir)<2:
            print "No $CMSSW_BASE directory can be found."
            print "Are you sure you've set up your CMSSW release area?"
            sys.exit()

        self.basedir=self.basedir.strip("\n")
        self.basedir=os.path.join(self.basedir,"src/DQM/HcalMonitorModule/python/dqmdbsGUI")
        
        self.finaldir=self.basedir # set this to some other location later!
       
        # Create GUI window
        if (parent==None):
            self.root=Tk()
            self.root.title("DQMfromDBS GUI")
        else:
            self.root=parent

        # Set initial position of GUI
        self.root.geometry('+25+25')

        if (debug):
            print "Created main GUI window"

        # Set up bg, fg colors for later use
        self.bg="#ffff73cb7"  # basic background color -- peach-ish
        self.bg_alt="#b001d0180" # alternate bg - dark red-ish
        self.fg="#180580410" # basic fg color -- green/grey-ish
        self.alt_active="gold3" # active bg for buttons

        self.root.configure(bg=self.bg)
        rootrow=0
        
        # Make menubar
        self.menubar=Frame(self.root,borderwidth=1,
                           bg=self.bg,
                           relief='raised')
        self.root.columnconfigure(0,weight=1)
        self.root.rowconfigure(1,weight=1)
        
        self.menubar.grid(row=rootrow,column=0,sticky=EW)

        rootrow=rootrow+1
        self.searchFrame=Frame(self.root,
                               bg=self.bg)
                               
        self.searchFrame.grid(row=rootrow,
                              sticky=EW,
                              column=0)

        rootrow=rootrow+1
        
        self.mainFrame=Frame(self.root,
                             bg=self.bg)
        self.mainFrame.grid(row=rootrow,column=0,sticky=EW)

        self.statusFrame=Frame(self.root,
                               bg=self.bg
                               )
        rootrow=rootrow+1
        self.statusFrame.grid(row=rootrow,column=0,sticky=EW)


        ########################################################
        #
        #  Fill the menu bar
        #

        # make File button on menubar
        self.BFile=Menubutton(self.menubar,
                              text = "File",
                              font = ('Times',12,'bold italic'),
                              activebackground=self.bg_alt,
                              activeforeground=self.bg,
                              bg=self.bg,
                              fg=self.fg,
                              padx=10, pady=8)
        self.BFile.pack(side="left")

        self.Bdbs=Menubutton(self.menubar,
                             text="DBS options",
                             font= ('Times',12,'bold italic'),
                             activebackground=self.bg_alt,
                             activeforeground=self.bg,
                             bg=self.bg,
                             fg=self.fg,
                             padx=10, pady=8)
        self.Bdbs.pack(side="left")


        self.Bprogress=Menubutton(self.menubar,
                                  text="Status",
                                  font=('Times',12,'bold italic'),
                                  activebackground=self.bg_alt,
                                  activeforeground=self.bg,
                                  bg=self.bg,
                                  fg=self.fg,
                                  padx=10, pady=8)
        self.Bprogress.pack(side="left")

        self.BAbout=Menubutton(self.menubar,
                               text="About",
                               font= ('Times',12,'bold italic'),
                               activebackground=self.bg_alt,
                               activeforeground=self.bg,
                               bg=self.bg,
                               fg=self.fg,
                               padx=10, pady=8)
        self.BAbout.pack(side="right")
        
        self.quitmenu=Menu(self.BFile, tearoff=0,
                           bg="white")
        self.quitmenu.add_separator()
        self.quitmenu.add_command(label="Quit",
                                  command = lambda x=self: x.goodQuit())
        self.BFile['menu']=self.quitmenu

        self.statusmenu=Menu(self.Bprogress,
                             bg="white")
        self.statusmenu.add_command(label="Show run status",
                                    command = lambda x=self:x.displayFiles())
        self.statusmenu.add_separator()
        self.statusmenu.add_command(label="Clear run file",
                                    command = lambda x=self:x.clearPickle())
        self.statusmenu.add_command(label="Restore from backup run file",
                                    command = lambda x=self:x.restoreFromBackupPickle())
        self.Bprogress['menu']=self.statusmenu


        # TO DO:  Add menu buttons for controlling DBS, DQM run ranges

        

        ########################################################
        #
        #  Fill the searchFrame
        #
        
        # Not yet sure whether to have BeginRun,EndRun variables,
        # or to determine them from LastFound, Range
        self.dbsRange=IntVar()
        self.lastFoundDBS=IntVar()

        searchrow=0
        Label(self.searchFrame,text = "Search over ",
              bg=self.bg,
              fg=self.bg_alt).grid(row=searchrow,column=0)
        self.dbsRangeEntry=Entry(self.searchFrame,
                                 bg="white",
                                 fg=self.bg_alt,
                                 width=8,
                                 textvar=self.dbsRange)
        self.dbsRangeEntry.grid(row=searchrow,column=1)
        self.lastFoundDBSEntry=Entry(self.searchFrame,
                                     bg="white",
                                     fg=self.bg_alt,
                                     width=8,
                                     textvar=self.lastFoundDBS)
        self.lastFoundDBSEntry.grid(row=searchrow,column=3)


        
        Label(self.searchFrame,text="runs, starting with run #",
              bg=self.bg,
              fg=self.bg_alt).grid(row=searchrow,column=2)


        #########################################################
        #
        # Fill main window frame
        #
        
        mainrow=0
        Label(self.mainFrame,text="",
              font = ('Times',2,'bold italic'),
              bg=self.fg).grid(row=mainrow,column=0,
                               columnspan=4,sticky=EW)

        
        mainrow=mainrow+1
        Label(self.mainFrame,text="Current Status",
              bg=self.bg,
              fg=self.bg_alt).grid(row=mainrow,column=1)
        Label(self.mainFrame,text="Last Update",
              bg=self.bg,
              fg=self.bg_alt).grid(row=mainrow,column=2)


        

        mainrow=mainrow+1
        self.dbsLabel=Label(self.mainFrame,
                            text="DBS:",
                            fg=self.fg, bg=self.bg,
                            width=10)
        self.dbsProgress=Label(self.mainFrame,
                               text="Nothing yet...",
                               bg="black",
                               width=40,
                               fg=self.bg)
        self.dbsStatus=Label(self.mainFrame,
                             text="Nothing yet...",
                             width=40,
                             bg=self.bg,
                             fg=self.fg)
        self.dbsButton=Button(self.mainFrame,
                              text="Check DBS\nfor runs",
                              height=2,
                              width=15,
                              fg=self.bg,
                              bg=self.bg_alt,
                              activebackground=self.alt_active,
                              command = lambda x=self:x.checkDBS()
                              )
        # Grid the DBS stuff
        self.dbsLabel.grid(row=mainrow,column=0)
        self.dbsProgress.grid(row=mainrow,column=1)
        self.dbsStatus.grid(row=mainrow,column=2)
        self.dbsButton.grid(row=mainrow,column=3)
        
        # Make labels/entries/buttons dealing with DQM searches
        self.dqmLabel=Label(self.mainFrame,
                            text="DQM:",
                            fg=self.fg, bg=self.bg,
                            width=10)
        self.dqmProgress=Label(self.mainFrame,
                               text="Nothing yet...",
                               bg="black",
                               width=40,
                               fg=self.bg)
        self.dqmStatus=Label(self.mainFrame,
                             text="Nothing yet...",
                             width=40,
                             bg=self.bg,
                             fg=self.fg)
        self.dqmButton=Button(self.mainFrame,
                              text="Run the \nHCAL DQM",
                              height=2,
                              width=15,
                              fg=self.bg,
                              bg=self.bg_alt,
                              activebackground=self.alt_active,
                              command = lambda x=self:x.runDQM_thread())

        # Grid the DQM stuff
        mainrow=mainrow+1
        self.dqmLabel.grid(row=mainrow,column=0)
        self.dqmProgress.grid(row=mainrow,column=1)
        self.dqmStatus.grid(row=mainrow,column=2)
        self.dqmButton.grid(row=mainrow,column=3)

        

        ######################################################
        #
        #  Fill the statusFrame
        #
        
        self.statusFrame.columnconfigure(0,weight=1)
        self.commentLabel=Label(self.statusFrame,
                                bg=self.bg,
                                fg=self.bg_alt,
                                height=2,
                                text="Welcome to the HCAL DQM/DBS GUI")
        statusrow=0
        self.commentLabel.grid(row=statusrow,column=0,sticky=EW)

        self.setup()

        return

    
    def setup(self):
        ''' Setup creates variables, sets values, etc. once drawing of
            main GUI is complete.'''

        
        self.dbsRange.set(10) # specify range of runs over which to search, starting at the LastDBS value

        self.lastFoundDBS.set(30625) # specify last run # found in DBS

        self.inittime=time.time()
        # call thread with time.sleep option
        self.foundfiles=0 # number of files found in the latest DBS search

        self.myDBS = dbsAccessor()

        self.dbsSearchInProgress=False
        self.pickleFileOpen=False
        self.runningDQM=False

        self.readPickle()
        # Set lastFoundDBS to most recent run in filesInDBS 
        if len(self.filesInDBS.keys()):
            x=self.filesInDBS.keys()
            x.sort()
            x.reverse()
            self.lastFoundDBS.set(x[0])

        #self.runDQM_thread()
        return

    def readPickle(self):
        
        if (self.pickleFileOpen):
            self.commentLabel.configure(text="Sorry, .filesInDBS.cPickle is already open")
            return
        self.pickleFileOpen=True

        if os.path.isfile(os.path.join(self.basedir,".filesInDBS.cPickle")):
            try:
                temp=open(os.path.join(self.basedir,".filesInDBS.cPickle"),'rb')
                self.filesInDBS=cPickle.load(temp)
                self.commentLabel.configure(text = "Loaded previously-read DBS entries from cPickle file")
                self.root.update()
            except SyntaxError:
                self.commentLabel.configure(text="WARNING!  Could not read .filesInDBS.cPickle file!\n-- Starting DBS list from scratch")
                self.filesInDBS={}
        else:
            self.filesInDBS={}
            self.commentLabel.configure(text = "Could not find file .filesInDBS.cPickle\n-- Starting DBS list from scratch")
            self.root.update()

        self.pickleFileOpen=False
        return


    def writePickle(self):
        if (self.pickleFileOpen):
            self.commentLabel.configure(text="Sorry, could not write information to .filesInDBS.cPickle.\ncPickle file is currently in use.")
            self.root.update()
            return
        self.pickleFileOpen=True
        if len(self.filesInDBS)>0:
            myfile=open(os.path.join(self.basedir,".filesInDBS.cPickle"),'wb')
            cPickle.dump(self.filesInDBS,myfile)
        self.pickleFileOpen=False
        return


    def clearPickle(self):
        if not (os.path.isfile(os.path.join(self.basedir,".filesInDBS.cPickle"))):
            self.commentLabel.configure(text="No run list .filesInDBS.cPickle exists!\nThere is nothing yet to clear!")
            self.root.update()
            return
                
        if tkMessageBox.askyesno("Remove .filesInDBS.cPickle?",
                                 "Clearing the list of runs is a major change!\nAre you sure you wish to proceed?"):
            os.system("mv %s %s"%(os.path.join(self.basedir,".filesInDBS.cPickle"),
                                  os.path.join(self.basedir,".backup_filesInDBS.cPickle")))

            self.filesInDBS={} # cleared files in memory
            self.commentLabel.configure(text="Run list cleared (saved as .backup_filesInDBS.cPickle)")
            self.root.update()

        return

    def restoreFromBackupPickle(self):
        if not (os.path.isfile(os.path.join(self.basedir,".backup_filesInDBS.cPickle"))):
            self.commentLabel.configure("Sorry, backup file does not exist!")
            self.root.update()
            return
        if tkMessageBox.askyesno("Restore from .backup_filesInDBS.cPickle",
                                 "Are you sure you want to restore files\nfrom backup?"):
            os.system("mv %s %s"%(os.path.join(self.basedir,".backup_filesInDBS.cPickle"),
                                  os.path.join(self.basedir,".filesInDBS.cPickle")))
            self.readPickle()
            self.commentLabel.configure(text="Run list restored from .backup_filesInDBS.cPickle")
            self.root.update()
        return
    
            

    def goodQuit(self):
        ''' Performs clean exit (checks that files are not currently running
            or being written, etc.) '''
        self.root.destroy()
        return



    def runDQM_thread(self):
        thread.start_new(self.runDQM,(1,2))
        return

    def runDQM(self,dummy1=None,dummy2=None):
        mytime=time.time()
        
        if self.runningDQM:
            self.commentLabel.configure(text="Sorry, DQM is already running")
            self.root.update()
            return

        self.dqmProgress.configure(text="Running DQM on available runs",
                                   bg=self.bg_alt)
        self.dqmStatus.configure(text="%s"%time.strftime("%d %b %Y at %H:%M:%S",time.localtime()))
        
        # Get list of runs -- whenever we change info, we write to pickle file
        # Therefore, read from the file to get the latest & greatest
        self.readPickle() 
        if len(self.filesInDBS.keys())==0:
            self.commentLabel.configure(text = "Sorry, no file info available.\nTry the 'Check DBS for Runs' button first.")
            self.dqmProgress.configure(text="No Run Info available",
                                       bg="black")
            self.root.update()
            return
        
        if len(self.filesInDBS.keys()):
            x=self.filesInDBS.keys()
            x.sort()
            x.reverse()
        else:
            self.commentLabel.configure(text="No unprocessed runs found")
            self.root.update()
            return
        
        self.runningDQM=True
        self.dqmButton.configure(state=DISABLED)

        goodcount=0
        for i in x:
            self.commentLabel.configure(text="Running DQM on run #%i"%i)
            self.dqmProgress.configure(text="Running DQM on run #%i"%i,
                                       bg=self.bg_alt)
            self.root.update()
            # Allow user to break loop via setting the runningDQM variable
            # (change to BooleanVar?)
            if (self.runningDQM==False):
                self.dqmButton.configure(state=ACTIVE)
                break
            # ignore files if necessary
            if self.filesInDBS[i].ignoreRun:
                goodcount=goodcount+1
                continue
            # if DQM started, check to see if DQM has finished
            if self.filesInDBS[i].startedDQM:
                # DQM finished; no problem
                if self.filesInDBS[i].finishedDQM:
                    goodcount=goodcount+1
                    continue
                # DQM not finished; look to see if directory made for it
                # (can later check for files within directory?)
                # Check to see if the output exists
                if (i<100000):
                    x=os.path.isdir("output/DQM_Hcal_R0000%i"%i)
                else:
                    x=os.path.isdir("output/DQM_Hcal_R000%i"%i)
                if (not x):
                    print "Problem with Run # %i -- DQM started but did not finish!"%i
                    self.commentLabel.configure(text="Problem with Run # %i -- DQM started but did not finish!"%i)
                    self.root.update()

                else:
                    # files have finished; need to update status
                    self.filesInDBS[i].finishedDQM=True
                    goodcount=goodcount+1
                    continue
            else:
                # nothing started yet; begin DQM

                # First check that cmsRun is available
                checkcmsRun=os.popen3("which cmsRun")
                # popen3 returns 3 streams -- in, out, and stderr
                # check that stderr is empty
                if len(checkcmsRun[2].readlines())>0:
                    self.commentLabel.configure(text="Could not find 'cmsRun'\nHave you set up your CMSSW environment?")
                    self.root.update()
                    return

                self.runningDQM=True
                self.filesInDBS[i].startedDQM=True
                if (self.callDQMscript(i)):
                    self.filesInDBS[i].finishedDQM=True
                    goodcount=goodcount+1



            # Every 20 minutes or so, check for updates to DBS files
            
            if (time.time()-mytime)>20*60:
                mytime=time.time()
                self.checkDBS()
                newfiles=False
                if len(self.filesInDBS.keys())<>x:
                    self.commentLabel.configure(text="DBS files have been added since last call to DQM.\n  Restarting DQM.")
                    self.root.update()
                    newfiles=True
                    break
                if (newFiles):
                    # Save current progress
                    self.writePickle()
                    self.dqmButton.configure(state=ACTIVE)
                    self.runDQM(self)
                else:
                    self.runningDQM=True

        self.runningDQM=False
        self.writePickle()

        if (goodcount==len(x)):
            self.dqmProgress.configure(text="Successfully finished running DQM",
                                       bg="black")
        else:
            self.dqmProgress.configure(text="Ran DQM on %i/%i runs"%(goodcount,len(x)))
        self.dqmStatus.configure(text="%s"%time.strftime("%d %b %Y at %H:%M:%S",time.localtime()))
        self.commentLabel.configure(text="Finished running DQM:\n%i out of %i runs successfully processed"%(goodcount,len(x)))
        self.dqmButton.configure(state=ACTIVE)
        self.root.update()
        return
                

    def callDQMscript(self,i):
        time.sleep(5)
        # Get rid of old file
        if os.path.isfile(os.path.join(self.basedir,".runOptions.cfi")):
            os.system("rm %s"%(os.path.join(self.basedir,".runOptions.cfi")))
        time.sleep(1)

        print "Opening file"
        print os.path.join(self.basedir,".runOptions.cfi")
        temp=open(os.path.join(self.basedir,".runOptions.cfi"),'w')
        temp.write("replace maxEvents.input=%i\n"%self.filesInDBS[i].maxEvents)
        filelength=len(self.filesInDBS[i].files)
        temp.write("replace PoolSource.fileNames={\n")
        for f in range(0,filelength):
            temp.write("'%s'"%string.strip(self.filesInDBS[i].files[f]))
            if (f==filelength-1):
                temp.write("\n")
            else:
                temp.write(",\n")
        temp.write("}\n")
        temp.close()
        
        # Now run cmsRun!
        if not (os.path.isfile(os.path.join(self.basedir,".runOptions.cfi"))):
            self.commentLabel.configure(text="Could not find .runOptions.cfi file\nFor run #%i"%i)
            self.root.update()
            return

        os.system("cmsRun hcal_dqm_dbsgui.cfg")
        
        if (i<100000):
            x="DQM_Hcal_R0000%i"%i
        else:
            x="DQM_Hcal_R000%i"%i
        success=False
        time.sleep(5)


        # Enable move commands once final_dir is determined
        #os.system("mv %s %s"%(x,self.final_dir))
        #os.system("mv %s %s"%(x,self.final_dir))

        
        # make fancier success requirement later
        if os.path.isdir(x) and os.path.isfile("%s.root"%x):
            success=True
        return success



    def checkDBS(self):
        if (self.dbsSearchInProgress):
            self.commentLabel.configure(text="Sorry, a DBS search is already in progress at the moment")
            return

        self.dbsButton.configure(state=DISABLED)
        self.dbsSearchInProgress=True
        begin=self.lastFoundDBS.get()
        end=begin+self.dbsRange.get()

        self.dbsProgress.configure(text="Checking DBS for runs %i-%i..."%(begin,end),
                                   bg=self.bg_alt)
        self.dbsStatus.configure(text="%s"%time.strftime("%d %b %Y at %H:%M:%S",time.localtime()))

        self.commentLabel.configure(text="Checking DBS for runs in range %i-%i..."%(begin,end))
        self.root.update()
        self.myDBS.searchDBS(begin,end)

        if (self.parseDBSInfo()):
            self.commentLabel.configure(text="Finished checking DBS runs (%i-%i)\nFound a total of %i runs"%(begin,end,self.foundfiles))

        self.dbsSearchInProgress=False
        self.dbsButton.configure(state=ACTIVE)
        return


    def parseDBSInfo(self):
        ''' Once we've checked DBS, let's parse the output for runs!'''

        files=string.split(self.myDBS.searchResult,"\n")
        tempdict={}

        self.foundfiles=0
        badcount=0
        for i in files:
            # Assume files are of the form .../A/XXX/XXX/XXX/RAW/....root,
            # where XXXXXXXX is the run number
            if len(i)==0:
                continue
            # Don't try to split the response describing found files;
            # simply store that info
            # (Send warning if found files is greater than # allowed files?)
            if (i.startswith("Found")):
                self.foundfiles=string.atoi(string.split(i)[1])
                if (self.foundfiles==0):
                    self.commentLabel.configure(text="WARNING!  No files found in the run range %i-%i"%(self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()))
                    self.dbsProgress.configure(text="No files found in range %i-%i"%(self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()),
                                               bg="black")
                    self.root.update()
                    return False
                continue

            try:
                temp=string.split(i,"/A/")[1]
                temp=string.split(temp,"/RAW/")[0]
                temp=temp.replace("/","")
                temp=string.atoi(temp)

                ignore=False
                # Protection when run number is wrong
                if (temp < self.lastFoundDBS.get() or temp > self.lastFoundDBS.get()+self.dbsRange.get()):
                    self.commentLabel.configure(text="WARNING!  File %s appears to be outside run range \n(%i-%i)"%(i,self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()))
                    self.root.update()
                    print "Run number disagreement for file %s !"%i
                    ignore=True # ignore these files for now; later try to correct with another call to dbs?
                
                # New run number found
                if temp not in tempdict.keys():
                    tempdict[temp]=DBSRun([i])
                    tempdict[temp].runnum=temp
                    tempdict[temp].ignoreRun=ignore
                    self.dbsProgress.configure(text="Found run %i in range (%i-%i)..."%(temp,self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()))
                    self.root.update()
                    

                    # Create separate DBS accessor to get dataset name
                    # (Do we need data
                    mytext="find dataset where file=%s"%i
                    tempDBS=dbsAccessor()
                    tempDBS.searchDBS(mytext=mytext)
                    dataset=string.split(tempDBS.searchResult,"\n")
                    for i in dataset:
                        if i.startswith("/"):
                            tempdict[temp].dataset=string.strip(i,"\n")
                            #print "DATASET = ",tempdict[temp].dataset
                            break
                # Another file found for old run number
                else:
                    tempdict[temp].files.append(i)

            except:
                self.commentLabel.configure(text="Could not parse DBS entry:\n'%s'"%i)
                badcount=badcount+1
                print "Could not parse DBS entry: %s"%i
                self.root.update()

        # more files were found that DBS is allowed to grab

        # Dump new events to pickle file
        for i in tempdict.keys():
            if i not in self.filesInDBS.keys():
                self.filesInDBS[i]=tempdict[i]

        # Set lastFoundDBS to most recent run in filesInDBS 
        if len(self.filesInDBS.keys()):
            x=self.filesInDBS.keys()
            x.sort()
            x.reverse()
            self.lastFoundDBS.set(x[0]) # add a +1?
        self.writePickle()
        
        if (self.foundfiles>self.myDBS.limit.get()):
            self.commentLabel.configure(text="WARNING! A total of %i files were found in DBS, but the current DBS limit is set to %i.  \nConsider increasing your DBS limit, or running on a smaller range of runs."%(foundfiles,self.myDBS.limit.get()))
            self.dbsProgress.configure(text="%i files found; only %i stored!"%(foundfiles,self.myDBS.limit.get()),
                                       bg="black")
            self.root.update()
            return False

        if badcount:
            self.dbsProgress.configure(text="%i lines from DBS could not be parsed!"%badcount)
            self.root.update()
            return False

        self.dbsProgress.configure(text="Successfully grabbed runs %i-%i"%(self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()),
                                   bg="black")
        self.root.update()
        return True



    def displayFiles(self):
        x=self.filesInDBS.keys()
        x.sort()
        x.reverse()
        temp = "%10s     %45s%10s     %10s%12s%15s%15s\n"%(" Run #", "Dataset"," ",
                                                 "# of files","IgnoreRun?",
                                                 "Started DQM?","Finished DQM?")
        for i in x:
            temp=temp+self.filesInDBS[i].Print()
        
        helpfunctions.Helpwin(temp,usetext=1)
        return


    


############################################

if __name__=="__main__":
    mygui=DQMDBSgui()
    mygui.root.mainloop()
