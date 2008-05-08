#!/usr/bin/env python

#################################################
#
# DQMfromDBSgui.py
#
# v1.2 Beta
#
# by Jeff Temple (jtemple@fnal.gov)
#
# 7 May 2008
#
# GUI to automatically grab new runs from DBS
# and run HCAL DQM on any newly-found runs
#
#################################################

import sys
import os # used for sending user commands
base=os.popen2("echo $CMSSW_BASE")
base=base[1].read()
if len(base)<2:
    print "No $CMSSW_BASE directory can be found."
    print "Are you sure you've set up your CMSSW release area?"
    sys.exit()
                                                 

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
import time # use to determine when to check for files
import cPickle # stores file information
import python_dbs # use for "sendmessage" function to get info from DBS
import string # DBS output parsing
import helpfunctions  # displays text in new window


###############################################################################
class DBSRun:
    '''
    Stores information about a given run
    (Run number, files in run, whether DQM has been performed on run.)
    '''

    def __init__(self,filelist=None):

        '''
        Class stores all files associated with a given run number, as
        given by DBS.  Also stores local DQM status (checking whether
        DQM has run on the set of files, and whether it has successfully
        completed.
        '''
        
        self.runnum=-1
        self.dataset=None
        self.files=[] # Stores files associated with run
        if (filelist<>None):
            self.files=filelist
        self.ignoreRun=0 # if true, DQM won't be performed on run
        self.startedDQM=0 # set to true once DQM started 
        self.finishedDQM=0 # set to true once DQM finished (DQM directory for run exists)
        self.previouslyFinishedDQM=0 # set to true if file is checked, and self.finishedDQM is already true
        self.maxEvents=1000 # number of events to run over with DQM
        return


    def Print(self):
        '''
        Returns DBSRun class variable values as a string.
        '''
        x= "%10s     %55s     %10s%12s%15s%15s\n"%(self.runnum,
                                                       self.dataset,
                                                       len(self.files),
                                                       self.ignoreRun,
                                                       self.startedDQM,
                                                       self.finishedDQM
                                                       #self.previouslyFinishedDQM
                                                       )
        
        #print x
        return x

def dbsSort(x,y):
    '''
    Sorts DBSRun objects by run number.
    '''
    return x.runnum>y.runnum
        
#############################################################################

class dbsAccessor:
    '''
    dbsAccessor:  Class that stores values which are used when
    accessing DBS web page.
    Values stored as IntVars, StringVars, etc. so that they can be
    easily utilized by the main DQMDBSgui.
    '''

    def __init__(self,basepath=os.curdir):
        '''
        dbsAccessor tries to read its values from a cPickle file.
        If the file does not exist, values will be initialized as defaults.
        IMPORTANT -- this class uses Tkinter StringVar, etc. variables,
        so this class won't properly initialize unless a Tk instance has
        been created first.
        '''
        
        self.basepath=basepath 
        try:
            self.host=StringVar()
            self.port=IntVar()
            self.dbsInst=StringVar()
            self.searchString=StringVar()
            self.page=IntVar()
            self.limit=IntVar()
            self.xml=BooleanVar()
            self.case=StringVar()
            self.details=BooleanVar()
            self.debug=BooleanVar()
            self.beginRun=IntVar()
            self.endRun=IntVar()
        except:
            print "Cannot define StringVar, IntVar, BooleanVar variables."
            print "Are you trying to use dbsAccessor outside of Tkinter?"
            sys.exit()
        self.getDefaultsFromPickle() # do we want to call this for every instance?
        return

    def setDefaults(self):
        '''
        Sets defaults for dbsAccessor.  This is used when
        defaults cannot be read from cPickle file.
        '''
        self.searchResult=None
        self.host.set("cmsweb.cern.ch/dbs_discovery/")
        self.port.set(443)
        self.dbsInst.set("cms_dbs_prod_global")
        self.searchString.set("*/Global*/A/*RAW/*.root")
        # Assume file takes the form */Global*/A/*/RAW/*.root
        self.page.set(0)
        self.limit.set(10000)
        self.xml.set(False)
        self.case.set("on")
        self.details.set(False)
        self.debug.set(False)
        self.beginRun.set(42100)
        self.endRun.set(42200)
        return

    def getDefaultsFromPickle(self):
        '''
        Try to read default values of dbsAccessor from .dbsDefaults.cPickle.
        If unsuccessful, defaults will be initialized from "setDefaults"
        function.
        '''
        
        if os.path.isfile(os.path.join(self.basepath,".dbsDefaults.cPickle")):
            try:
                pcl=open(os.path.join(self.basepath,".dbsDefaults.cPickle"),'rb')
                self.host.set(cPickle.load(pcl))
                self.port.set(cPickle.load(pcl))
                self.dbsInst.set(cPickle.load(pcl))
                self.searchString.set(cPickle.load(pcl))
                self.page.set(cPickle.load(pcl))
                self.limit.set(cPickle.load(pcl))
                self.xml.set(cPickle.load(pcl))
                self.case.set(cPickle.load(pcl))
                self.details.set(cPickle.load(pcl))
                self.debug.set(cPickle.load(pcl))
                self.beginRun.set(cPickle.load(pcl))
                self.endRun.set(cPickle.load(pcl))
                pcl.close()
                          
            except:
                self.setDefaults()
                print "Could not read file '.dbsDefaults.cPickle'"
        else:
            self.setDefaults()
        return

    def writeDefaultsToPickle(self):
        '''
        Writes default dbsAccessor values to "dbsDefaults.cPickle"
        '''
        
        try:
            pcl=open(os.path.join(self.basepath,".dbsDefaults.cPickle"),'wb')
            cPickle.dump(self.host.get(),pcl)
            cPickle.dump(self.port.get(),pcl)
            cPickle.dump(self.dbsInst.get(),pcl)
            cPickle.dump(self.searchString.get(),pcl)
            cPickle.dump(self.page.get(),pcl)
            cPickle.dump(self.limit.get(),pcl)
            cPickle.dump(self.xml.get(),pcl)
            cPickle.dump(self.case.get(),pcl)
            cPickle.dump(self.details.get(),pcl)
            cPickle.dump(self.debug.get(),pcl)
            cPickle.dump(self.beginRun.get(),pcl)
            cPickle.dump(self.endRun.get(),pcl)
            pcl.close()
            os.system("chmod a+rw %s"%(os.path.join(self.basedir,".dbsDefaults.cPickle")))
                      
        except:
            print "Could not write file '.dbsDefaults.cPickle'"
        return

    def searchDBS(self,beginrun=-1, endrun=-1,mytext=None):
        '''
        Searches DBS for files matching specified criteria.
        Criteria is given by user-supplied "mytext" string.
        If no such string is provided, default is "find run
        where file = <dbsAccessor default searchString> and
        run between <default begin run>-<default end run>"
        '''

        # If beginrun, endrun specified, use their values in the search
        if (beginrun>-1):
            self.beginRun.set(beginrun)
        if (endrun>-1):
            self.endRun.set(endrun)
        if (mytext==None):
            # Assume dataset takes the form */Global*-A/*
            mytext="find run where file=%s and run between %i-%i"%(self.searchString.get(),self.beginRun.get(),self.endRun.get())
        #print "mytext = ",mytext

        # Send search string to DBS, and store result as "searchResult"
        self.searchResult=python_dbs.sendMessage(self.host.get(),
                                                 self.port.get(),
                                                 self.dbsInst.get(),
                                                 mytext,
                                                 self.page.get(),
                                                 self.limit.get(),
                                                 self.xml.get(),
                                                 self.case.get(),
                                                 self.details.get(),
                                                 self.debug.get())
        return
    

###################################################################
class DQMDBSgui:
    '''
    DQMDBSgui:  Main GUI Class
    '''
    
    def __init__(self, parent=None, debug=False):

        '''
        DQMDBSgui.__init__  creates the graphic interface for the
        program, and initializes a few needed variables.  Remaining
        variables are created through the setup() method, called at
        the end of __init__.
        '''

        # Check that CMSSW environment has been set;
        # Set basedir to CMSSW release area
        checkCMSSW=os.popen2("echo $CMSSW_BASE")
        self.basedir=checkCMSSW[1].read()
        if len(self.basedir)<2:
            print "No $CMSSW_BASE directory can be found."
            print "Are you sure you've set up your CMSSW release area?"
            sys.exit()


        # Now set base directory to area of release containg GUI
        self.basedir=self.basedir.strip("\n")
        self.basedir=os.path.join(self.basedir,"src/DQM/HcalMonitorModule/python/dqmdbsGUI")
        if not os.path.exists(self.basedir):
            print "Unable to find directory '%s'"%self.basedir
            print "Have you checked out the appropriate package in your release area?"
            sys.exit()

        os.chdir(self.basedir)  # put all output into basedir

        self.debug=debug
        
        # Create GUI window
        if (parent==None):
            self.root=Tk()
            self.root.title("HCAL DQM from DBS GUI")
            self.root.geometry('+25+25') # set initial position of GUI
        else:
            self.root=parent # could conceivably put GUI within another window


        if (debug):
            print "Created main GUI window"

        # Set up bg, fg colors for use by GUI
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
        self.root.columnconfigure(0,weight=1) # allows column 0 to expand
        self.root.rowconfigure(1,weight=1)
        
        self.menubar.grid(row=rootrow,column=0,sticky=EW)

        rootrow=rootrow+1
        # Create frame that holds search values (i.e., run range to search in DBS)
        self.searchFrame=Frame(self.root,
                               bg=self.bg)
                               
        self.searchFrame.grid(row=rootrow,
                              sticky=EW,
                              column=0)

        rootrow=rootrow+1

        # Create main Frame (holds "Check DBS" and "Check DQM" buttons and status values)
        self.mainFrame=Frame(self.root,
                             bg=self.bg)
        self.mainFrame.grid(row=rootrow,column=0,sticky=EW)

        # Frame that will display overall status messages
        self.statusFrame=Frame(self.root,
                               bg=self.bg
                               )
        rootrow=rootrow+1
        self.statusFrame.grid(row=rootrow,column=0,sticky=EW)


        ########################################################
        #                                                      #
        #  Fill the menu bar                                   #
        #                                                      #
        ########################################################
        
        mycol=0
        # make File button on menubar
        self.BFile=Menubutton(self.menubar,
                              text = "File",
                              font = ('Times',12,'bold italic'),
                              activebackground=self.bg_alt,
                              activeforeground=self.bg,
                              bg=self.bg,
                              fg=self.fg,
                              padx=10, pady=8)
        self.BFile.grid(row=0,column=mycol,sticky=W)

        # Make DBS option menu on menubar
        mycol=mycol+1
        self.Bdbs=Menubutton(self.menubar,
                             text="DBS options",
                             font= ('Times',12,'bold italic'),
                             activebackground=self.bg_alt,
                             activeforeground=self.bg,
                             bg=self.bg,
                             fg=self.fg,
                             padx=10, pady=8)
        self.Bdbs.grid(row=0,column=mycol,sticky=W)


        # Make DQM option menu on menubar
        mycol=mycol+1
        self.Bdqm=Menubutton(self.menubar,
                             text="DQM options",
                             font= ('Times',12,'bold italic'),
                             activebackground=self.bg_alt,
                             activeforeground=self.bg,
                             bg=self.bg,
                             fg=self.fg,
                             padx=10, pady=8)
        self.Bdqm.grid(row=0,column=mycol,sticky=W)

        # Make status button on menubar
        mycol=mycol+1
        self.Bprogress=Menubutton(self.menubar,
                                  text="Status",
                                  font=('Times',12,'bold italic'),
                                  activebackground=self.bg_alt,
                                  activeforeground=self.bg,
                                  bg=self.bg,
                                  fg=self.fg,
                                  padx=10, pady=8)
        self.Bprogress.grid(row=0,column=mycol)

        # Eventually will allow copying of files to destinations
        # outside local areas
        mycol=mycol+1
        self.menubar.columnconfigure(mycol,weight=1)
        self.enableSCP=BooleanVar()
        self.enableSCP.set(True)
        self.scpAutoButton=Checkbutton(self.menubar,
                                       bg=self.bg,
                                       fg=self.fg,
                                       text="scp copying enabled",
                                       activebackground=self.alt_active,
                                       variable=self.enableSCP,
                                       padx=10,
                                       command=self.toggleSCP)
        self.scpAutoButton.grid(row=0,column=mycol,sticky=E)
        #mycol=mycol+1
        #self.menubar.columnconfigure(mycol,weight=1)
        #self.scpLabel=Label(self.menubar,
        #                    #text="Copy DQM to:",
        #                    text="scp copying enabled ",
        #                    fg=self.fg,bg=self.bg).grid(row=0,
        #                                                column=mycol,
        #                                                sticky=E)
        mycol=mycol+1
        # Add in variable copy abilities later
        #self.copyLocVar=StringVar()
        #self.copyLocVar.set("Local area")
        # List possible copy destinations
        # (not yet implemented, until we can figure out how to
        #  auto scp)
        #self.copyLoc=OptionMenu(self.menubar,self.copyLocVar,
        #                        "Local area"
        #                        #"cmshcal01"
        #                        )
        #self.copyLoc.configure(background=self.bg,
        #                       foreground=self.fg,
        #                       activebackground=self.alt_active)

        # for now, make button to copy files with scp
        self.copyLoc=Button(self.menubar,
                            text="Copy Output!",
                            command=lambda x=self:x.tempSCP())
        self.copyLoc.configure(background=self.bg_alt,
                               foreground=self.bg,
                               activebackground=self.alt_active)
        self.copyLoc.grid(row=0,column=mycol,sticky=E)
                

        # Make 'heartbeat' label that shows when auto-checking is on
        mycol=mycol+1
        self.HeartBeat=Label(self.menubar,
                             text="Auto",
                             bg=self.bg,
                             fg=self.bg,
                             padx=10,pady=8)
        self.HeartBeat.grid(row=0,column=mycol,sticky=W)

        # Make 'About' menu to store help information 
        self.BAbout=Menubutton(self.menubar,
                               text="About",
                               font= ('Times',12,'bold italic'),
                               activebackground=self.bg_alt,
                               activeforeground=self.bg,
                               bg=self.bg,
                               fg=self.fg,
                               padx=10, pady=8)
        mycol=mycol+1
        self.BAbout.grid(row=0,column=mycol,sticky=W)


        # Fill 'File' Menu
        self.quitmenu=Menu(self.BFile, tearoff=0,
                           bg="white")

        # Clear out default values
        self.quitmenu.add_command(label="Clear all default files",
                                  command=lambda x=self:x.removeFiles(removeAll=False))
        # Clear out hidden files
        self.quitmenu.add_command(label="Clear ALL hidden files",
                                  command=lambda x=self:x.removeFiles(removeAll=True))
        self.quitmenu.add_separator()
        # Call Quit coomand
        self.quitmenu.add_command(label="Quit",
                                  command = lambda x=self: x.goodQuit())

        self.BFile['menu']=self.quitmenu


        # Fill 'Status' Menu
        self.statusmenu=Menu(self.Bprogress,
                             bg="white")
        self.statusmenu.add_command(label="Show run status",
                                    command = lambda x=self:x.displayFiles())
        self.statusmenu.add_separator()
        self.statusmenu.add_command(label="Change run status",
                                    command = lambda x=self:x.changeFileSettings())
        self.statusmenu.add_separator()
        self.statusmenu.add_command(label="Clear run file",
                                    command = lambda x=self:x.clearPickle())
        self.statusmenu.add_command(label="Restore from backup run file",
                                    command = lambda x=self:x.restoreFromBackupPickle())
        self.Bprogress['menu']=self.statusmenu

        # Fill 'About' menu"
        self.aboutmenu=Menu(self.BAbout,
                            bg="white")
        temptext="DQMfromDBS GUI\n\nv1.2 Beta\nby Jeff Temple\n7 May 2008\n\n"
        temptext=temptext+"GUI allows users to query DBS for files in a specified\nrun range, and then run HCAL DQM over those files.\n\nQuestions or comments?\nSend to:  jtemple@fnal.gov\n"
        self.aboutmenu.add_command(label="Info",
                                   command = lambda x=helpfunctions:
                                   x.Helpwin(temptext,usetext=1,title="About this program..."))
        self.aboutmenu.add_command(label="Help",
                                   command = lambda x=helpfunctions:
                                   x.Helpwin("dqmdbs_instructions.txt",title="Basic instructions for the user"))
        self.BAbout['menu']=self.aboutmenu


        # Fill 'DBS Options' Menu
        self.dbsmenu=Menu(self.Bdbs,
                          bg="white",
                          tearoff=0)
        self.dbsmenu.add_command(label="Change DBS settings",
                                 command = lambda x=self:x.printDBS())
        # update with save DBS
        self.dbsmenu.add_separator()

        # Implement save command later
        #self.dbsmenu.add_command(label="Save DBS settings",
        #                         command = lambda x=self:x.printDBS())
        
        self.Bdbs['menu']=self.dbsmenu

        # Fill 'DQM Options' Menu
        self.dqmmenu=Menu(self.Bdqm,
                          bg="white",
                          tearoff=0)
        self.dqmmenu.add_command(label="Change DQM settings",
                                 command = lambda x=self:x.printDQM())
        self.dqmmenu.add_separator()
        self.Bdqm['menu']=self.dqmmenu
        

        ########################################################
        #                                                      #
        #  Fill the searchFrame                                #
        #                                                      #
        ########################################################
        

        # Declare variables for range of runs to be searched in DBS,
        # as well as the starting point
        # ("lastFoundDBS" is a bit of a misnomer -- this value will
        # actually be 1 greater than the last found run under normal
        # circumstances.)

        self.dbsRange=IntVar()
        self.lastFoundDBS=IntVar()

        searchrow=0
        Label(self.searchFrame,text = "Search over ",
              bg=self.bg,
              fg=self.bg_alt).grid(row=searchrow,column=0)

        # Entry boxes hold IntVars
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

        
        Label(self.searchFrame,
              text="runs, starting with run #",
              bg=self.bg,
              fg=self.bg_alt).grid(row=searchrow,column=2)


        #########################################################
        #                                                       #
        # Fill main window frame                                #
        #                                                       #
        #########################################################
        
        mainrow=0
        # This is a blank label that provides a green stripe across the GUI
        Label(self.mainFrame,text="",
              font = ('Times',2,'bold italic'),
              bg=self.fg).grid(row=mainrow,column=0,
                               columnspan=10,sticky=EW)

        
        mainrow=mainrow+1
        Label(self.mainFrame,text="Current Status",
              bg=self.bg,
              fg=self.bg_alt).grid(row=mainrow,column=1)
        Label(self.mainFrame,text="Last Update",
              bg=self.bg,
              fg=self.bg_alt).grid(row=mainrow,column=2)

        # Create boolean for determining whether or not Auto-running
        # is enabled
        self.Automated=BooleanVar()
        self.Automated.set(False)
        self.autoButton=Button(self.mainFrame,
                               text="Auto-Update\nDisabled!!",
                               bg="black",
                               fg="white",
                               command = lambda x=self:x.checkAutoUpdate())
        self.autoButton.grid(row=mainrow,column=4,padx=10,pady=6)

        

        mainrow=mainrow+1
        # Make labels/entries/buttons dealing with DBS
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
        self.dbsAutoVar=BooleanVar()
        self.dbsAutoVar.set(False)
        self.dbsAutoButton=Checkbutton(self.mainFrame,
                                       text="Auto DBS",
                                       state=DISABLED,
                                       bg=self.bg,
                                       fg=self.fg,
                                       activebackground=self.alt_active,
                                       variable=self.dbsAutoVar)


        # Grid the DBS stuff
        self.dbsLabel.grid(row=mainrow,column=0)
        self.dbsProgress.grid(row=mainrow,column=1)
        self.dbsStatus.grid(row=mainrow,column=2)
        self.dbsButton.grid(row=mainrow,column=3)
        self.dbsAutoButton.grid(row=mainrow,column=4)
        
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

        self.dqmAutoVar=BooleanVar()
        self.dqmAutoVar.set(False)
        self.dqmAutoButton=Checkbutton(self.mainFrame,
                                       text="Auto DQM",
                                       state=DISABLED,
                                       bg=self.bg,
                                       fg=self.fg,
                                       activebackground=self.alt_active,
                                       variable=self.dqmAutoVar)


        # Grid the DQM stuff
        mainrow=mainrow+1
        self.dqmLabel.grid(row=mainrow,column=0)
        self.dqmProgress.grid(row=mainrow,column=1)
        self.dqmStatus.grid(row=mainrow,column=2)
        self.dqmButton.grid(row=mainrow,column=3)
        self.dqmAutoButton.grid(row=mainrow,column=4)
        

        ######################################################
        #                                                    #
        #  Fill the statusFrame                              #
        #                                                    #
        ######################################################
        
        self.statusFrame.columnconfigure(0,weight=1)
        # commentLabel will display messages to user
        self.commentLabel=Label(self.statusFrame,
                                bg=self.bg,
                                fg=self.bg_alt,
                                height=2,
                                text="Welcome to the HCAL DQM/DBS GUI")
        statusrow=0
        self.commentLabel.grid(row=statusrow,column=0,sticky=EW)

        # Call setup (initializes remaining needed variables)
        self.setup()
        return



    ##########################################################################
    def setup(self):
        ''' Setup creates variables, sets values, etc. once drawing of
            main GUI is complete.'''


        # DQM output is initially stored locally;
        #self.finalDir determines where
        # it will be sent once the DQM has finished running.
        self.finalDir=StringVar()
        self.finalDir.set(self.basedir) # set this to some other location later!

        # Store maximum # of events to be run for each DQM job 
        self.maxDQMEvents=IntVar()
        self.maxDQMEvents.set(1000)


        # TO DO:  Make this default value changeable by user?  Save in cPickle?
        self.dbsRange.set(100) # specify range of runs over which to search, starting at the LastDBS value

        self.lastFoundDBS.set(42100) # specify last run # found in DBS

        self.inittime=time.time()
        # call thread with time.sleep option
        self.foundfiles=0 # number of files found in the latest DBS search

        self.myDBS = dbsAccessor(self.basedir) # Will access runs from DBS
        self.myDBS.getDefaultsFromPickle()
        self.dbsSearchInProgress=False
        self.pickleFileOpen=False
        self.runningDQM=False

        self.readPickle() # Read defaults from cPickle file
        
        # Set lastFoundDBS to most recent run in filesInDBS 
        if len(self.filesInDBS.keys()):
            x=self.filesInDBS.keys()
            x.sort()
            x.reverse()
            self.lastFoundDBS.set(x[0])


        # TO DO:  Make Auto Update Times adjustable by user
        self.dbsAutoUpdateTime=20 # dbs update time in minutes
        self.dbsAutoCounter=0
        self.dqmAutoUpdateTime=20 # dqm update time in minutes
        self.dqmAutoCounter=0
        self.autoRunning=False
        self.hbcolor=self.bg

        self.cfgFileName=StringVar()
        self.cfgFileName.set(os.path.join(self.basedir,"hcal_dqm_dbsgui.cfg"))
        self.getDefaultDQMFromPickle()

        self.autoRunShift=True # automatically updates run entry when new run found
        # Hidden trick to freeze starting run value!
        self.lastFoundDBSEntry.bind("<Shift-Up>",self.toggleAutoRunShift)
        self.lastFoundDBSEntry.bind("<Shift-Down>",self.toggleAutoRunShift)
        return


    ############################################################
    def checkAutoUpdate(self):
        ''' This is the function associated with the "Auto Update" button.
            It toggles the self.Automated variable.
            If self.Automated is true, then DBS searches and DQM running
            are performed automatically.
            '''

        #self.dqmAutoButton.flash()
        self.Automated.set(1-self.Automated.get())  # toggle boolean
        if (self.Automated.get()==True):
            self.autoButton.configure(text="Auto Update\nEnabled",
                                      bg=self.bg_alt,
                                      fg=self.bg)
            # enable DQM, DBS buttons
            self.dqmAutoButton.configure(state=NORMAL,bg=self.bg,fg=self.fg)
            self.dbsAutoButton.configure(state=NORMAL,bg=self.bg,fg=self.fg)
            self.dbsAutoVar.set(True)
            self.dqmAutoVar.set(True)
            # Start autoUpdater thread
            thread.start_new(self.autoUpdater,())

        else:
            # Boolean false; turn off auto updater
            self.autoButton.configure(text="Auto Update\nDisabled!!",
                                      bg="black",
                                      fg="white")
            self.dqmAutoButton.configure(state=DISABLED)
            self.dbsAutoButton.configure(state=DISABLED)
            self.dbsAutoVar.set(False)
            self.dqmAutoVar.set(False)

        self.root.update()

        return

    #########################################################
    def heartbeat(self,interval=1):
        '''
        Make heartbeat label flash once per second.
        '''
        while (self.Automated.get()):
            if (self.hbcolor==self.bg):
                self.hbcolor=self.bg_alt
            else:
                self.hbcolor=self.bg
            self.HeartBeat.configure(bg=self.hbcolor)
            self.root.update()
            time.sleep(interval)

        self.HeartBeat.configure(bg=self.bg)
        return

    #########################################################
    def autoUpdater(self):
        ''' DQM/DBS Auto updater. '''

        if self.autoRunning==True:
            # Don't allow more than one autoUpdater to run at one time
            # (I don't think this is possible anyway)
            self.commentLabel.configure(text="Auto Updater is already running!")
            self.root.update()
            return
        if self.Automated.get()==False:
            self.commentLabel.configure(text="Auto Updater is disabled")
            self.root.update()
            return

        thread.start_new(self.heartbeat,()) # create heartbeat to show auto-update is running
        self.checkDBS() # perform initial check of files
        self.runDQM_thread() # perform initial check of DQM
        
        while (self.Automated.get()):
            self.autoRunning=True
            for xx in range(0,60):
                time.sleep(1)
                if not self.Automated.get():
                    break

            # print self.dbsAutoVar.get(), self.dqmAutoVar.get()
            # Increment counters once per minute
            self.dbsAutoCounter=self.dbsAutoCounter+1
            self.dqmAutoCounter=self.dqmAutoCounter+1
            #print self.dbsAutoCounter
            
            # if DBS counter > update time ,check DBS for new files
            if (self.dbsAutoCounter >= self.dbsAutoUpdateTime):
                # Reset counter if auto dbs disabled
                if (self.dbsAutoVar.get()==False):
                    self.dbsAutoCounter=0
                else:
                    #print "Checking DBS!"
                    if (self.checkDBS()): # search was successful; reset counter
                        self.dbsAutoCounter=0
                        #print "DBS Check succeeded!"
                    else: # search unsuccessful; try again in 1 minute
                        self.dbsAutoCounter=(self.dbsAutoUpdateTime-1)*60
                        #print "DBS Check unsuccessful"

            # repeat for DQM checking
            if (self.dqmAutoCounter >= self.dqmAutoUpdateTime):
                # Remind user to scp completed files
                self.tempSCP()
                # If dqmAutoVar is off, reset counter
                if (self.dqmAutoVar.get()==False):
                    self.dqmAutoCounter=0
                # Otherwise, run DQM
                else:
                    self.runDQM_thread()


        # Auto updating deactivated; reset counters and turn off heartbeat
        self.dbsAutoCounter=0
        self.dqmAutoCounter=0
        self.autoRunning=False
        return
        

    def printDBS(self):
        '''
        Create new window showing DBS values; allow user to change them.
        '''
        try:
            self.dbsvaluewin.destroy()
            self.dbsvaluewin=Toplevel()
        except:
            self.dbsvaluewin=Toplevel()

        self.dbsvaluewin.title('Change DBS values')
        self.dbsvaluewin.geometry('+600+300')
        myrow=0

        # Variables to be shown in window
        # Add spaces in front of some keys so that they appear
        # first when keys are sorted.
        myvars={"  DBS File Search String = ":self.myDBS.searchString,
                "  DBS Files to Return = ":self.myDBS.limit,
                " DBS Host = ":self.myDBS.host,
                " DBS Port = ":self.myDBS.port,
                " DBS Instance = ":self.myDBS.dbsInst,
                "Output in XML Format? ":self.myDBS.xml,
                "Show detailed output? ":self.myDBS.details,
                "Case-sensitive matching? ":self.myDBS.case,
                "Output page = ":self.myDBS.page,
                "Debugging = ":self.myDBS.debug}

        temp=myvars.keys()
        temp.sort()

        # Grid variable labels, entreis
        for i in temp:
            Label(self.dbsvaluewin,
                  width=40,
                  text="%s"%i).grid(row=myrow,column=0)
            Entry(self.dbsvaluewin,
                  width=40,
                  textvar=myvars[i]).grid(row=myrow,column=1)
            myrow=myrow+1

        # Grid buttons for saving, restoring values
        Button(self.dbsvaluewin,text="Save as new default values",
               command = lambda x=self.myDBS:x.writeDefaultsToPickle()).grid(row=myrow,column=0)
        Button(self.dbsvaluewin,text="Restore default values",
               command = lambda x=self.myDBS:x.getDefaultsFromPickle()).grid(row=myrow,
                                                                             column=1)
        return


    def printDQM(self):
        '''
        Create window for editing DQM values.
        '''
        try:
            self.dqmvaluewin.destroy()
            self.dqmvaluewin=Toplevel()
        except:
            self.dqmvaluewin=Toplevel()

        self.dqmvaluewin.geometry('+400+300')
        self.dqmvaluewin.title("Change DQM Values")
        myrow=0

        # List of variables to be shown in window.
        # Add spaces in front of some keys so that they
        # appear first when sorted.
        
        myvars={"  Final DQM Save Directory = ":self.finalDir,
                "  # of events to run for each DQM = ":self.maxDQMEvents,
                "  .cfg file to run for each DQM = ":self.cfgFileName}
        temp=myvars.keys()
        temp.sort()

        # Create variable labels, entries
        for i in temp:
            Label(self.dqmvaluewin,
                  width=40,
                  text="%s"%i).grid(row=myrow,column=0)
            tempEnt=Entry(self.dqmvaluewin,
                          width=80,
                          textvar=myvars[i])
            tempEnt.grid(row=myrow,column=1)
            if i=="  Final DQM Save Directory = " or i=="  .cfg file to run for each DQM = ":
                tempEnt.bind("<Return>",(lambda event:self.checkExistence(myvars[i])))
            myrow=myrow+1
        newFrame=Frame(self.dqmvaluewin)
        newFrame.grid(row=myrow,column=0,columnspan=2,sticky=EW)
        newFrame.columnconfigure(0,weight=1)
        newFrame.columnconfigure(1,weight=1)
        newFrame.columnconfigure(2,weight=1)
        Button(newFrame,text="Save as new default\n DQM values",
               command = lambda x=self:x.writeDefaultDQMToPickle()).grid(row=0,column=0)
        Button(newFrame,text="Restore default DQM values",
               command = lambda x=self:x.getDefaultDQMFromPickle()).grid(row=0,
                                                                         column=1)
        Button(newFrame,text="Exit",
               command = lambda x=self.dqmvaluewin:x.destroy()).grid(row=0,column=2)
        return


    def getDefaultDQMFromPickle(self):
        '''
        Get DQM default values from .dqmDefaults.cPickle.
        '''

        if os.path.isfile(os.path.join(self.basedir,".dqmDefaults.cPickle")):
            try:
                pcl=open(os.path.join(self.basedir,".dqmDefaults.cPickle"),'rb')
                self.finalDir.set(cPickle.load(pcl))
                self.maxDQMEvents.set(cPickle.load(pcl))
                self.cfgFileName.set(cPickle.load(pcl))
                pcl.close()
            except:
                self.commentLabel.configure(text="Could not read file '.dqmDefaults.cPickle' ")
                self.root.update()
        return

    def writeDefaultDQMToPickle(self):
        '''
        Write DQM default values to .dqmDefaults.cPickle.
        '''
        try:
            pcl=open(os.path.join(self.basedir,".dqmDefaults.cPickle"),'wb')
            cPickle.dump(self.finalDir.get(),pcl)
            cPickle.dump(self.maxDQMEvents.get(),pcl)
            cPickle.dump(self.cfgFileName.get(),pcl)
            pcl.close()
            os.system("chmod a+rw %s"%os.path.join(self.basedir,".dqmDefaults.cPickle"))
                      
        except SyntaxError:
            self.commentLabel.configure(text="Could not write file '.dqmDefaults.cPickle' ")
            self.root.update()
        return


    def readPickle(self):
        '''
        Read list of found runs from .filesInDBS.cPickle.
        '''
        
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
            except:
                self.commentLabel.configure(text="WARNING!  Could not read .filesInDBS.cPickle file!\n-- Starting DBS list from scratch")
                self.filesInDBS={}
        else:
            self.filesInDBS={}
            self.commentLabel.configure(text = "Could not find file .filesInDBS.cPickle\n-- Starting DBS list from scratch")
            self.root.update()

        self.pickleFileOpen=False
        return


    def writePickle(self):
        '''
        Write list of found runs to .filesInDBS.cPickle.
        '''
        if (self.pickleFileOpen):
            self.commentLabel.configure(text="Sorry, could not write information to .filesInDBS.cPickle.\ncPickle file is currently in use.")
            self.root.update()
            return
        self.pickleFileOpen=True
        if len(self.filesInDBS)>0:
            try:
                myfile=open(os.path.join(self.basedir,".filesInDBS.cPickle"),'wb')
                cPickle.dump(self.filesInDBS,myfile)
                myfile.close()
                os.system("chmod a+rw %s"%os.path.join(self.basedir,".filesInDBS.cPickle"))
            except:
                self.commentLabel.configure(text="ERROR!  Could not write to file .filesInDBS.cPickle!\n  This bug will be investigated!")
                self.root.update()
        self.pickleFileOpen=False
        return


    def clearPickle(self):
        '''
        Clear list of found runs, copying .cPickle info to backup file.
        '''
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
        '''
        Restore list of found runs from .backup_filesInDBS.cPickle file
        '''
        
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
    


    def removeFiles(self,removeAll=False):
        '''
        Removes hidden files (files starting with "."), such as default option settings, etc.
        If removeAll is set true, then the .filesInDBS.cPickle file that is used to store run history is also removed.
        One exception:  .backup_filesInDBS.cPickle can never be removed via the GUI
        '''
        if (removeAll):
            text="This will remove ALL hidden files\n used by the GUI, and will clear\nthe list of runs processed by the\n program.\n\nAre you ABSOLUTELY SURE \nthis is what you want?\n"
        else:
            text="This will remove hidden files used to\nstore certain user-set default values.\n\nAre you SURE this is what you want?\n"
        if tkMessageBox.askyesno("Remove default files???  Really???",
                                 text):
            temp=[".dqmDefaults.cPickle",".dbsDefaults.cPickle",".runOptions.cfi"]
            if (removeAll):
                temp.append(".filesInDBS.cPickle")
                self.filesInDBS={}
            for i in temp:
                x=os.path.join(self.basedir,i)
                if os.path.isfile(x):
                  self.commentLabel.configure(text="Removing file '%s'"%i)
                  self.root.update()
                  os.system("rm -f %s"%x)
                  time.sleep(0.5)
        return

    def goodQuit(self):
        ''' Eventually will perform clean exit
        (checks that files are not currently running
        or being written, etc.)
        '''

        if (self.dbsSearchInProgress or self.runningDQM):
            text="The following jobs have not yet finished:"
            if (self.dbsSearchInProgress):
                text=text+"\nA DBS Search is still in progress"
            if (self.runningDQM):
                text=text+"\nDQM is currently running"
            text=text+"\nDo you want to exit anyway?"
            if not tkMessageBox.askyesno("Jobs not yet completed",text):
                return

        # TO DO:  KILL EXISTING cmsRun jobs if they are running when the user decides to quit.
        self.Automated.set("False")
        self.root.destroy()
        return



    def runDQM_thread(self):
        '''
        Start new thread for running DQM
        '''
        thread.start_new(self.runDQM,(1,2))
        return

    def runDQM(self,dummy1=None,dummy2=None):
        '''
        Runs DQM over all found files.
        Can we get rid of dummy1, dummy2 variables?
        '''
        
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

        if (self.debug): print "<runDQM>  Read pickle file"
        if len(self.filesInDBS.keys())==0:
            self.commentLabel.configure(text = "Sorry, no file info available.\nTry the 'Check DBS for Runs' button first.")
            self.dqmProgress.configure(text="No Run Info available",
                                       bg="black")
            self.root.update()
            return

        # If runs found, sort by run number (largest number first)
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

        mycount=0
        goodcount=0
        for i in x:
            if self.debug:  print "<runDQM> Checking run #%i"%i
            self.commentLabel.configure(text="Running DQM on run #%i"%i)
            self.dqmProgress.configure(text="Running DQM on run #%i"%i,
                                       bg=self.bg_alt)
            self.root.update()
            # Allow user to break loop via setting the runningDQM variable
            # (change to BooleanVar?)
            if (self.runningDQM==False):
                if self.debug:  print "<runDQM> runningDQM bool = False"
                self.dqmButton.configure(state=ACTIVE)
                break
            # ignore files if necessary
            if self.filesInDBS[i].ignoreRun:
                goodcount=goodcount+1
                continue
            # if DQM started, check to see if DQM has finished
            if self.filesInDBS[i].startedDQM:
                if self.filesInDBS[i].previouslyFinishedDQM:
                    # File was finished previously; don't count it here
                    continue
                # DQM finished; no problem
                mycount=mycount+1
                if self.filesInDBS[i].finishedDQM:
                    self.filesInDBS[i].previouslyFinishedDQM=True
                    goodcount=goodcount+1
                    continue
                # DQM not finished; look to see if directory made for it
                # (can later check for files within directory?)
                # Check to see if the output exists
                if (i<100000):
                    mydirname=os.path.isdir("DQM_Hcal_R0000%i"%i)
                else:
                    mydirname=os.path.isdir("DQM_Hcal_R000%i"%i)
                if not mydirname:
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
                if (self.debug): print "<runDQM> looking for cmsRun"
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
                
            if (self.debug):
                print "<runDQM> made it through callDQMscript"

            # Every 20 minutes or so, check for updates to DBS files
            
            if (time.time()-mytime)>20*60:
                if (self.debug): print "<runDQM> getting time info"
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

        if (self.debug):
            print "<runDQM> Hi there!"
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
        ''' Here is where we actually perform the cmsRun call.'''
        
        time.sleep(1)
        # Get rid of old file
        if os.path.isfile(os.path.join(self.basedir,".runOptions.cfi")):
            os.system("rm %s"%(os.path.join(self.basedir,".runOptions.cfi")))
        time.sleep(1)

        #print os.path.join(self.basedir,".runOptions.cfi")
        try:
            temp=open(os.path.join(self.basedir,".runOptions.cfi"),'w')
        except:
            self.commentLabel.configure(text="MAJOR ERROR!  Could not write to .runOptions.cfi!  \nCheck file/directory permissions!")
            self.dqmProgress.configure(text="FAILED!  Couldn't write .runOptions.cfi")
            self.root.update()
            return False
        
        # Allow a different # for each file?
        #temp.write("replace maxEvents.input=%i\n"%self.filesInDBS[i].maxEvents)

        temp.write("replace maxEvents.input=%i\n"%self.maxDQMEvents.get())
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
        os.system("chmod a+rw %s"%os.path.join(self.basedir,".runOptions.cfi"))
        
        # Now run cmsRun!
        if not (os.path.isfile(os.path.join(self.basedir,".runOptions.cfi"))):
            self.commentLabel.configure(text="Could not find .runOptions.cfi file\nFor run #%i"%i)
            self.root.update()
            return False

        os.system("cmsRun %s"%self.cfgFileName.get())
        
        if (i<100000):
            x="DQM_Hcal_R0000%i"%i
        else:
            x="DQM_Hcal_R000%i"%i
        success=False
        time.sleep(2)

        #print "x = %s"%x
        # make fancier success requirement later -- for now, just check that directory exists
        if (self.debug):
            print "%s exists? %i"%(os.path.join(self.basedir,x),os.path.isdir(os.path.join(self.basedir,x)))

        if os.path.isdir(os.path.join(self.basedir,x)):
            success=True
            if (self.debug):
                print "<callDQMScript> success=True!"

            #print "Directory exists!"
            # If final destination is in local area, and
            # if final dir differs from base dir, move to that directory
            if (
                #self.copyLocVar.get()=="Local area" and # copyLocVar doesn't exist at the moment!
                 self.finalDir.get()<>self.basedir):
                if self.debug:
                    print "Checking for root file"
                # move .root file, if it's been created
                temproot="%s.root"%(os.path.join(self.basedir,x))
                #print "temproot = ",temproot
                if os.path.isfile(temproot):
                    junk="mv %s %s.root"%(temproot,os.path.join(self.finalDir.get(),x))
                    #print "junk = ",junk
                    os.system("mv %s %s.root"%(temproot,
                                               os.path.join(self.finalDir.get(),
                                                            x)))
                    self.commentLabel.configure(text = "moved file %s\n to directory %s"%(temproot,
                                                                                        self.finalDir.get()))
                    self.root.update()
                    time.sleep(1)
                else:
                    self.commentLabel("ERROR -- Can't find file %s\nDo you know where your output is?"%temproot)
                    self.root.update()
                # move directory
                #print "Now moving directory"
                tempdir=os.path.join(self.finalDir.get(),x)
                # Get rid of old version of directory, if it exists
                if os.path.isdir(tempdir):
                    os.system("rm -rf %s"%tempdir)
                # Now move results to final directory
                os.system("mv %s %s"%(x,self.finalDir.get())) 
                self.commentLabel.configure(text = "moved folder %s\n to directory %s"%(x,self.finalDir.get()))
                self.root.update()
                time.sleep(1)

            if (self.debug):
                print "<CallDQMscript> What's going on?"
            # This needs to be updated once we figure out how to auto scp
            #elif (self.copyLocVar.get()=="cmshcal01"):
                #os.system("scp %s ..."%x)  # update with end location name!
                #print "cmshcal01 copying not yet implemented!"
                
        if self.debug:
            print "<CallDQMScript> Success = %s"%success
        return success


    def checkDBS(self):
        if (self.dbsSearchInProgress):
            self.commentLabel.configure(text="Sorry, a DBS search is already in progress at the moment")
            return False

        self.dbsButton.configure(state=DISABLED)
        self.dbsSearchInProgress=True
        begin=self.lastFoundDBS.get()
        end=begin+self.dbsRange.get()

        self.dbsProgress.configure(text="Checking DBS for runs %i-%i..."%(begin,end),
                                   bg=self.bg_alt)
        self.dbsStatus.configure(text="%s"%time.strftime("%d %b %Y at %H:%M:%S",time.localtime()))

        self.commentLabel.configure(text="Checking DBS for runs in range %i-%i..."%(begin,end))
        self.root.update()
        self.myDBS.searchDBS(begin,end) # Search, getting run numbers

        if (self.parseDBSInfo()):
            self.commentLabel.configure(text="Finished checking DBS runs (%i-%i)\nFound a total of %i files"%(begin,end,self.foundfiles))

        self.dbsSearchInProgress=False
        self.dbsButton.configure(state=ACTIVE)
        return True


    def parseDBSInfo(self):
        '''
        Once we've checked DBS, let's parse the output for runs!
        Updated on 5 May 2008 -- apparently, run number in file name cannot be trusted.
        Instead, we'll get info by checking DBS for all run numbers within range.
        Then, for each found run number, we'll grab all the files for that run number
        '''

        runlist=[]
        begin=self.lastFoundDBS.get()
        end=begin+self.dbsRange.get()
        runs=string.split(self.myDBS.searchResult,"\n")
        for r in runs:
            if (len(r)==0):
                continue # skip blank output lines
            # Use "Found ... runs" output line to determine # of runs found
            if (r.startswith("Found")):
                self.foundfiles=string.atoi(string.split(r)[1])
                if (self.foundfiles==0):
                    self.commentLabel.configure(text="WARNING!  No runs found in the run range %i-%i"%(self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()))
                    self.dbsProgress.configure(text="No runs found in range %i-%i"%(self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()),
                                               bg="black")
                    self.root.update()
                    return False

            try:
                r.strip("\n")
                r=string.atoi(r)
                if r not in runlist:
                    runlist.append(r)
            except:
                continue
            
        if len(runlist)==0:
            self.commentLabel.configure(text="ODD BEHAVIOR!  Runs apparently found, but cannot be parsed!\nDBS output being redirected to screen")
            print "DBS Run search result: ",self.myDBS.searchResult
            self.dbsProgress.configure(text="No runs in (%i-%i) could be parsed!"%(self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()),
                                       bg="black")
            self.root.update()
            return False


        # Now loop over each run to get final search result

        self.foundfiles=0
        badcount=0
        for r in runlist:
                        
            self.dbsProgress.configure(text="Found run %i in range (%i-%i)..."%(r,self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()))
            self.root.update()

            tempfiles=[]

            # For each run, create new accessor that will find files, datasets associated with the run
            x=dbsAccessor()
            text="find file,dataset where file=%s and run=%i"%(self.myDBS.searchString.get(),r)
            x.searchDBS(mytext=text)
            
            files=string.split(x.searchResult,"\n")

            dataset=None
            # Parse output looking over files for a given run number
            for file in files:

                # ignore blank lines
                if len(file)==0:
                    continue

                # Don't try to split the response describing found files;
                # simply store that info
                # (Send warning if found files is greater than # allowed files?)
                if (file.startswith("Found")):
                    self.foundfiles=self.foundfiles+string.atoi(string.split(file)[1])
                    if (self.foundfiles==0):
                        self.commentLabel.configure(text="WARNING!  No files found for run # %i"%r)
                        self.dbsProgress.configure(text="No files found for run # %i"%r,
                                                   bg="black")
                        self.root.update()
                        return False
                else:
                    #print "i = ",file
                    try:
                        i=string.split(file,",")
                        dataset=i[1] # dataset
                        i=i[0] # file name
                        if (i.endswith(".root")):  # file must be .root file
                            tempfiles.append(i) 
                        else:
                             self.commentLabel.configure(text="Could not recognize DBS entry:\n'%s'"%i)
                             badcount=badcount+1
                             print "Could not parse DBS entry: %s"%i
                             self.root.update()

                    except:
                        self.commentLabel.configure(text="Could not parse DBS entry:\n'%s'"%i)
                        badcount=badcount+1
                        print "Could not parse DBS entry: %s"%i
                        self.root.update()

            tempDBS=DBSRun(tempfiles)
            tempDBS.runnum=r
            tempDBS.maxEvents=self.maxDQMEvents.get()
            tempDBS.dataset=string.strip(dataset,"\n")

            if r not in self.filesInDBS.keys():
                self.filesInDBS[r]=tempDBS
            else:
                for file in tempDBS.files:
                    if file not in self.filesInDBS[r].files:
                        self.filesInDBS[r].files.append(file)
                
            
        # Set lastFoundDBS to most recent run in filesInDBS 
        
        if len(self.filesInDBS.keys()):
            x=self.filesInDBS.keys()
            x.sort()
            x.reverse()
            if (self.autoRunShift):
                self.lastFoundDBS.set(x[0]+1) # move to next run after last-found
            #for zz in x:
            #    print self.filesInDBS[zz].Print()
            #    for ff in self.filesInDBS[zz].files:
            #        print ff

            #self.lastFoundDBS.set(self.lastFoundDBS.get()+self.dbsRange.get())
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
        
        self.dbsProgress.configure(text="Successfully grabbed runs %i-%i"%(begin,end),
                                   bg="black")
        self.root.update()
        
        return True
    


    def displayFiles(self):
        '''
        Show all run numbers that have been found from DBS, along with the
        DQM status of each run.
        '''

        # Get runs, sort highest-to-lowest
        x=self.filesInDBS.keys()
        x.sort()
        x.reverse()
        temp ="%10s     %45s%10s     %10s%12s%15s%15s\n"%(" Run #", "Dataset"," ",
                                                          "# of files","IgnoreRun?",
                                                          "Started DQM?","Finished DQM?")
        for i in x:
            temp=temp+self.filesInDBS[i].Print()

        # Make window displaying run info
        if (len(x)<>1):
            title="A total of %i runs have been found in DBS"%len(x)
        else:
            title="A total of 1 run has been found in DBS"
        helpfunctions.Helpwin(temp,usetext=1,title=title )
        return


    def changeFileSettings(self):
        '''
        Allows user to change the DQM status of the runs found from DBS.
        (Mark runs as having already completed DQM, set ignoreRun true, etc.)
        '''

        # If window exists already, destroy it and recreate 
        try:
            self.changevaluewin.destroy()
            self.changevaluewin=Toplevel()
        except:
            self.changevaluewin=Toplevel()
        self.changevaluewin.geometry('+800+20')
        self.changevaluewin.title("Change status of files")
        
        # Add list of runs as a list box with attached scrollbar
        scrollwin=Frame(self.changevaluewin)
        scrollwin.grid(row=0,column=0)
        myrow=0
        Label(scrollwin,
              text="Be sure to check Status menu after your changes are made").grid(row=myrow,column=0)
        myrow=myrow+1
        lb=Listbox(scrollwin,
                   selectmode = MULTIPLE)
        # Get list of runs
        self.listboxruns=self.filesInDBS.keys()
        self.listboxruns.sort()
        self.listboxruns.reverse()

        for i in self.listboxruns:
            lb.insert(END,i)
            
        scroll=Scrollbar(scrollwin,command=lb.yview)
        lb.configure(yscrollcommand=scroll.set)
        lb.grid(row=myrow,column=0,sticky=NSEW)
        scroll.grid(row=myrow,column=1,sticky=NS)

        # Add buttons for changing DQM values
        myrow=myrow+1
        self.changevaluewin.rowconfigure(myrow,weight=1)
        bFrame=Frame(self.changevaluewin)
        bFrame.grid(row=1,column=0)
        igY=Button(bFrame,
                   text="Set\n'Ignore Run'\nTrue",
                   command=lambda x=self:x.commandChangeFileSettings(lb.curselection(),
                                                                     "ignoreRun",True),
                   width=14,height=3)
        igN=Button(bFrame,
                   text="Set\n'Ignore Run'\nFalse",
                   command=lambda x=self:x.commandChangeFileSettings(lb.curselection(),
                                                                     "ignoreRun",False),
                   width=14,height=3)
        stY=Button(bFrame,
                   text="Set\n'Started DQM'\nTrue",
                    command=lambda x=self:x.commandChangeFileSettings(lb.curselection(),
                                                                      "startedDQM",True),
                   width=14,height=3)
        stN=Button(bFrame,
                   text="Set\n'Started DQM'\nFalse",
                   command=lambda x=self:x.commandChangeFileSettings(lb.curselection(),
                                                                     "startedDQM",False),
                   width=14,height=3)
        fiY=Button(bFrame,
                   text="Set\n'Finished DQM'\nTrue",
                   command=lambda x=self:x.commandChangeFileSettings(lb.curselection(),
                                                                     "finishedDQM",True),
                   width=14,height=3)
        fiN=Button(bFrame,
                   text="Set\n'Finished DQM'\nFalse",
                   command=lambda x=self:x.commandChangeFileSettings(lb.curselection(),
                                                                     "finishedDQM",False),
                   width=14,height=3)

        # Grid buttons
        igY.grid(row=0,column=0)
        stY.grid(row=0,column=1)
        fiY.grid(row=0,column=2)
        igN.grid(row=1,column=0)
        stN.grid(row=1,column=1)
        fiN.grid(row=1,column=2)

        return
        
    def commandChangeFileSettings(self,selected,var,value=True):
        '''
        Commands for changing DQM settings.
        "selected" is the set of listbox indices that have been
        highlighted by the user.
        (self.listboxruns[int(i)] returns the associated run #, for
         all i in selected.)
        Allowed values for var:
        "ignoreRun", "startedDQM", "finishedDQM"
        '''
        
        for i in selected:

            run=self.listboxruns[int(i)] # get run number from index

            if (var=="ignoreRun"):
                self.filesInDBS[run].ignoreRun=value
            elif (var=="startedDQM"):
                self.filesInDBS[run].startedDQM=value
            elif (var=="finishedDQM"):
                self.filesInDBS[run].finishedDQM=value
        self.writePickle() # save to pickle file?  I think this is the sensible option (user can always change back)
        return


    def toggleSCP(self):
        ''' swaps SCP variable.  If SCP variable is off, no scp copying will take place.'''

        if (self.enableSCP.get()==0):
            self.scpAutoButton.configure(text="scp copying disabled")
            self.copyLoc.configure(state=DISABLED)
        else:
            self.scpAutoButton.configure(text="scp copying enabled")
            self.copyLoc.configure(state=NORMAL)
        return

    def toggleAutoRunShift(self,event):
        '''
        This toggles the autoRunShift variable.
        If autoRunShift is true, then the run entry
        value will increment whenever a new run is found.
        If not, the run entry value will remain the same.
        '''

        
        self.autoRunShift=1-self.autoRunShift # toggle value
        # Change entry box color if auto shifting is not enabled
        if (self.autoRunShift==False):
              self.lastFoundDBSEntry.configure(bg="yellow")
        else:
            self.lastFoundDBSEntry.configure(bg="white")
        return


    def checkExistence(self,obj):
        #print obj.get()
        exists=True
        if not os.path.exists(obj.get()):
            self.commentLabel.configure(text="ERROR!\n Object '%s' does not exist!"%obj.get())
            self.root.update()
            obj.set("ERROR -- FILE/DIR DOES NOT EXIST")
            exists=False
        return exists

    def tempSCP(self):
        '''
        Temporary method for running scp from local final directory
        to hcalusc55@cmshcal01:hcaldqm/global_auto/
        Jeff
        '''

        if not (self.enableSCP.get()):
            self.commentLabel.configure("scp copying is not currently enabled.\n(Check button in the middle of the menu bar")")
            self.root.update()
            return
        
        if not (os.path.exists(self.finalDir.get())):
            self.commentLabel.configure(text="ERROR -- directory '%s' DOES NOT EXIST!!\nEdit the Final DQM Save Directory in DQM options!"%self.finalDir.get())
            return

        # make directory for files/dirs that have already been copied.
        if not os.path.isdir(os.path.join(self.finalDir.get(),"copied_to_hcaldqm")):
            os.mkdir(os.path.join(self.finalDir.get(),"copied_to_hcaldqm"))

        movelist=os.listdir(self.finalDir.get())
        movelist.remove("copied_to_hcaldqm")
        if len(movelist)==0:
            self.commentLabel.configure(text="There are no files in %s\n to be copied to cmshcal01!"%self.finalDir.get())
            self.root.update()
            return
        text1="scp -r "
        text="scp -r "
        for i in movelist:
            text=text+"%s "%os.path.join(self.finalDir.get(),"copied_to_hcaldqm",i)
            text1=text1+"%s "%os.path.join(self.finalDir.get(),i)
        text=text+" hcalusc55@cmshcal01:/hcaldqm/global_auto\n\n"
        text1=text1+" hcalusc55@cmshcal01:/hcaldqm/global_auto\n\n"
        
        #if at cms:
        #if os.getenv("USER")=="cchcal":
        compname=os.uname()[1]
        if string.find(compname,"lxplus")>-1 and string.find(compname,".cern.ch")>-1:
            zzz=os.system(text1)
            print text1
            #print zzz
            return
        
        else:
            for i in movelist:
                cmd="mv %s %s\n"%(os.path.join(self.finalDir.get(),i),
                                  os.path.join(self.finalDir.get(),"copied_to_hcaldqm",i))
            
                #os.system(cmd)
            helpfunctions.Helpwin(text,usetext=1,title="Cut and paste this command into your lxplus window now!" )
        return

############################################

if __name__=="__main__":

    mygui=DQMDBSgui(debug=0)  # set up gui
    mygui.root.mainloop() # run main loop
