#!/usr/bin/env python

#################################################
#
# DQMfromDBSgui.py
#
# v1.3 Beta
#
# by Jeff Temple (jtemple@fnal.gov)
#
# 10 May 2008
#
# v1.3 updates -- separate code into subpackages
#      introduce separate file, dataset substrings
#      reposition daughter windows relative to parent

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


from pyDBSRunClass import DBSRun  # Gets class that holds information as to whether a given run has been processed through DQM

from pydbsAccessor import dbsAccessor  # sends queries to DBS

    

###################################################################
class dbsBaseGui:
    '''
    dbsBaseGui:  Base Class for finding files in DBS, running analyses on them

    '''
    
    def __init__(self, parent=None, debug=False):

        '''
        **self.__init__(parent=None, debug=False)**
        dbsBaseGui.__init__  sets up all class variables used by the GUI.
        '''

        # Check that CMSSW environment has been set;
        # Set basedir to CMSSW release area
        checkCMSSW=os.popen2("echo $CMSSW_BASE")
        self.basedir=checkCMSSW[1].read()
        if len(self.basedir)<2:
            print "No $CMSSW_BASE directory can be found."
            print "Are you sure you've set up your CMSSW release area?"
            sys.exit()


        # Now set base directory to area of release containing GUI
        self.basedir=self.basedir.strip("\n")
        self.basedir=os.path.join(self.basedir,"src/DQM/HcalMonitorModule/python/dqmdbsGUI")
        if not os.path.exists(self.basedir):
            print "Unable to find directory '%s'"%self.basedir
            print "Have you checked out the appropriate package in your release area?"
            sys.exit()

        os.chdir(self.basedir)  # put all output into basedir


        # init function should define all variables needed for the creation of the GUI
        self.debug=debug
        if (self.debug):
            print self.__doc__
            print self.__init__.__doc__

        self.parent=parent  # parent window in which GUI is drawn (a new window is created if parent==None)
        # Create GUI window
        if (self.parent==None):
            self.root=Tk()
            self.root.title("HCAL DQM from DBS GUI")
            self.root.geometry('+25+25') # set initial position of GUI
        else:
            self.root=self.parent # could conceivably put GUI within another window

        #######################################
        # Set up bg, fg colors for use by GUI #
        #######################################
        self.bg="#ffff73cb7"  # basic background color -- peach-ish
        self.bg_alt="#b001d0180" # alternate bg - dark red-ish
        self.fg="#180580410" # basic fg color -- green/grey-ish
        self.alt_active="gold3" # active bg for buttons

        self.enableSCP=BooleanVar() # variable to determine whether 'scp' copying is enabled
        self.enableSCP.set(True)

        # Set possible auto-update times (in minutes) for DBS, DQM
        self.updateTimes=(2,5,10,20,30,60,120)

        # Variables used for starting point, range of runs to search when looking at DBS
        self.dbsRange=IntVar()
        self.lastFoundDBS=IntVar()

        # Create boolean for determining whether or not Auto-running
        # is enabled
        self.Automated=BooleanVar()
        self.Automated.set(False)
        # "Sub"-booleans for toggling DBS, DQM independently when Auto-running is enabled
        self.dbsAutoVar=BooleanVar()
        self.dbsAutoVar.set(False)
        self.dqmAutoVar=BooleanVar()
        self.dqmAutoVar.set(False)
        
        # DQM output is initially stored locally;
        #self.finalDir determines where
        # it will be sent once the DQM has finished running.
        self.finalDir=StringVar()
       
        # Store maximum # of events to be run for each DQM job 
        self.maxDQMEvents=IntVar()
        self.maxDQMEvents.set(1000)

        # TO DO:  Make this default value changeable by user?  Save in cPickle?
        self.dbsRange.set(100) # specify range of runs over which to search, starting at the LastDBS value

        self.lastFoundDBS.set(42100) # specify last run # found in DBS

        self.foundfiles=0 # number of files found in the latest DBS search -- deprecated variable?

        self.dbsSearchInProgress=False
        self.pickleFileOpen=False
        self.runningDQM=False


        # Set initial DBS auto-update interval to 20 minutes
        self.dbsAutoUpdateTime=IntVar()
        self.dbsAutoUpdateTime.set(20) # dbs update time in minutes
        self.dbsAutoCounter=0
        # Set initial DQM auto-update interval to 20 minutes
        self.dqmAutoUpdateTime=IntVar()
        self.dqmAutoUpdateTime.set(20) # dqm update time in minutes

        self.dqmAutoCounter=0
        self.autoRunning=False
        self.hbcolor=self.bg # heartbeat color

        self.cfgFileName=StringVar()
        self.mycfg="hcal_dqm_dbsgui.cfg"

        self.autoRunShift=True # automatically updates run entry when new run found

        self.cmsRunOutput=[] # will store files, dirs successfully created by cmsRun
        return



    


    def DrawGUI(self):
        '''
        ** self.DrawGUI() **
        Creates GUI window, grids all Tkinter objects within the window.
        This function should only contain calls to objects in the GUI display.
        Variables attached to those objects are created in the __init__ method.
        '''

        if (self.debug):
            print self.DrawGUI.__doc__
            
        self.root.configure(bg=self.bg)
        rootrow=0
        
        # Make menubar
        self.menubar=Frame(self.root,borderwidth=1,
                           bg=self.bg,
                           relief='raised')
        self.root.columnconfigure(0,weight=1) # allows column 0 to expand
        self.root.rowconfigure(1,weight=1)
        
        self.menubar.grid(row=rootrow,column=0,sticky=EW)


        # Create frame that holds search values (i.e., run range to search in DBS)
        rootrow=rootrow+1
        self.searchFrame=Frame(self.root,
                               bg=self.bg)
                               
        self.searchFrame.grid(row=rootrow,
                              sticky=EW,
                              column=0)


        # Create main Frame (holds "Check DBS" and "Check DQM" buttons and status values)
        rootrow=rootrow+1
        self.mainFrame=Frame(self.root,
                             bg=self.bg)
        self.mainFrame.grid(row=rootrow,column=0,sticky=EW)

        # Frame that will display overall status messages
        rootrow=rootrow+1
        self.statusFrame=Frame(self.root,
                               bg=self.bg
                               )
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
        self.scpAutoButton=Checkbutton(self.menubar,
                                       bg=self.bg,
                                       fg=self.fg,
                                       text="scp copying enabled",
                                       activebackground=self.alt_active,
                                       variable=self.enableSCP,
                                       padx=10,
                                       command=self.toggleSCP)
        
        self.scpAutoButton.grid(row=0,column=mycol,sticky=E)

        # This is an old implementation of scp choices.
        # Can it be removed?  Jeff, 10 May 2008
        
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


        # for now, use a button to copy files with scp
        self.copyLoc=Button(self.menubar,
                            text="Copy Output!",
                            command=lambda x=self:x.tempSCP())
        self.copyLoc.configure(background=self.bg_alt,
                               foreground=self.bg,
                               activebackground=self.alt_active)
        self.copyLoc.grid(row=0,column=mycol,sticky=E)

        # Turn off copying by default if user is not "cchcal"
        if os.getenv("USER")<>"cchcal":
            self.enableSCP.set(False)
            self.scpAutoButton.configure(text="scp copying disabled")
            self.copyLoc.configure(state=DISABLED)
                                    


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
        # Call Quit command
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
        temptext="DQMfromDBS GUI\n\nv1.3 Beta\nby Jeff Temple\n10 May 2008\n\n"
        temptext=temptext+"GUI allows users to query DBS for files in a specified\nrun range, and then run HCAL DQM over those files.\n\nQuestions or comments?\nSend to:  jtemple@fnal.gov\n"
        self.aboutmenu.add_command(label="Info",
                                   command = lambda x=helpfunctions:
                                   x.Helpwin(temptext,usetext=1,title="About this program..."))
        self.aboutmenu.add_command(label="Help",
                                   command = lambda x=helpfunctions:
                                   x.Helpwin("%s"%os.path.join(self.basedir,dqmdbs_instructions.txt),title="Basic instructions for the user"))
        self.BAbout['menu']=self.aboutmenu


        
        # Fill 'DBS Options' Menu
        self.dbsmenu=Menu(self.Bdbs,
                          bg="white",
                          tearoff=0)
        self.dbsmenu.add_command(label="Change DBS settings",
                                 command = lambda x=self:x.printDBS())
        
        self.dbsmenu.add_separator()

        # Create submenu to allow changing of DBS auto-update interval
        self.dbsUpdateMenu=Menu(self.dbsmenu,
                                bg="white",
                                tearoff=0)
        self.dbsUpdateMenu.choices=Menu(self.dbsUpdateMenu,
                                        bg="white",
                                        tearoff=0)

        for upTime in range(len(self.updateTimes)):
            self.dbsUpdateMenu.choices.add_command(label='%s minutes'%self.updateTimes[upTime],
                                                   command = lambda x=upTime,y=self.updateTimes:self.dbsSetUpdateMenu(x,y)
                                                   )
        self.dbsmenu.add_cascade(label="Set DBS update time",
                                 menu=self.dbsUpdateMenu.choices)
        
        self.Bdbs['menu']=self.dbsmenu

        # Fill 'DQM Options' Menu
        self.dqmmenu=Menu(self.Bdqm,
                          bg="white",
                          tearoff=0)
        self.dqmmenu.add_command(label="Change DQM settings",
                                 command = lambda x=self:x.printDQM())
        self.dqmmenu.add_separator()

        # Create submenu to allow changing of DQM auto-update interval
        self.dqmUpdateMenu=Menu(self.dqmmenu,
                                bg="white",
                                tearoff=0)
        self.dqmUpdateMenu.choices=Menu(self.dqmUpdateMenu,
                                        bg="white",
                                        tearoff=0)

        for upTime in range(len(self.updateTimes)):
            self.dqmUpdateMenu.choices.add_command(label='%s minutes'%self.updateTimes[upTime],
                                                   command = lambda x=upTime,y=self.updateTimes:self.dqmSetUpdateMenu(x,y)
                                                   )
        self.dqmmenu.add_cascade(label="Set DQM update time",
                                 menu=self.dqmUpdateMenu.choices)
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
        self.stripe=Label(self.mainFrame,text="",
                          font = ('Times',2,'bold italic'),
                          bg=self.fg)
        self.stripe.grid(row=mainrow,column=0,
                         columnspan=10,sticky=EW)


        # Make row showing column headings, Auto-Update button
        mainrow=mainrow+1
        Label(self.mainFrame,text="Current Status",
              bg=self.bg,
              fg=self.bg_alt).grid(row=mainrow,column=1)
        Label(self.mainFrame,text="Last Update",
              bg=self.bg,
              fg=self.bg_alt).grid(row=mainrow,column=2)

        # Button for enabling/disabling auto updating
        self.autoButton=Button(self.mainFrame,
                               text="Auto-Update\nDisabled!!",
                               bg="black",
                               fg="white",
                               command = lambda x=self:x.checkAutoUpdate())
        self.autoButton.grid(row=mainrow,column=4,padx=10,pady=6)


        # Make labels/entries/buttons dealing with DBS
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

        self.dbsAutoButton=Checkbutton(self.mainFrame,
                                       text="Auto DBS\nupdate OFF",
                                       state=DISABLED,
                                       bg=self.bg,
                                       fg=self.fg,
                                       width=20,
                                       activebackground=self.alt_active,
                                       variable=self.dbsAutoVar,
                                       command=self.toggleAutoDBS)


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

        self.dqmAutoButton=Checkbutton(self.mainFrame,
                                       text="Auto DQM\nupdate OFF",
                                       state=DISABLED,
                                       bg=self.bg,
                                       fg=self.fg,
                                       width=20,
                                       activebackground=self.alt_active,
                                       variable=self.dqmAutoVar,
                                       command=self.toggleAutoDQM)


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
                                height=2)

        statusrow=0
        self.commentLabel.grid(row=statusrow,column=0,sticky=EW)

        # Call setup (initializes remaining needed variables)
        self.setup()
        return



    ##########################################################################
    def setup(self):
        '''
        **self.setup() **
        Setup performs some GUI tuning that cannot be completed until variables are declared and
        GUI is drawn.
        
            '''

        if self.debug:  print self.setup.__doc__

        #self.commentLabel.configure("Welcome to the HCAL DBS/DQM GUI")
        #self.commentLabel.update_idletasks()

        os.chdir(self.basedir) # cd to the self.basedir directory

        self.finalDir.set(self.basedir) # set this to some other location later?

        # Create DBS accessor
        self.myDBS = dbsAccessor(self.basedir,debug=self.debug) # Will access runs from DBS
        self.myDBS.getDefaultsFromPickle()


        # Default file settings may only be read/set once correct basedir location
        # is set.
        
        self.cfgFileName.set(os.path.join(self.basedir,self.mycfg))
        self.getDefaultDQMFromPickle(startup=True)

        self.readPickle() # Read defaults from cPickle file
        
        # Set lastFoundDBS to most recent run in filesInDBS 
        if len(self.filesInDBS.keys()):
            x=self.filesInDBS.keys()
            x.sort()
            x.reverse()
            self.lastFoundDBS.set(x[0])

        # Set update time in UpdateMenus to 20 minutes (they will appear in red in the menu)
        for temptime in range(len(self.updateTimes)):
            if self.updateTimes[temptime]==20:
                self.dbsUpdateMenu.choices.entryconfig(temptime,foreground="red")
                self.dqmUpdateMenu.choices.entryconfig(temptime,foreground="red")

        # Hidden trick to freeze starting run value!
        self.lastFoundDBSEntry.bind("<Shift-Up>",self.toggleAutoRunShift)
        self.lastFoundDBSEntry.bind("<Shift-Down>",self.toggleAutoRunShift)

        if not os.path.isdir(self.finalDir.get()):
            self.commentLabel.configure(text="WARNING -- specified Final DQM Save Directory does not exist!\nCheck settings in DQM options!")
            self.commentLabel.update_idletasks()
        return


    ############################################################
    def checkAutoUpdate(self):
        '''
        ** self.checkAutoUpdate() **
        This is the function associated with the "Auto Update" button.
        It toggles the self.Automated variable.
        If self.Automated is true, then DBS searches and DQM running
        are performed automatically.
        '''

        if (self.debug):
            print self.checkAutoUpdate.__doc__
        
        #self.dqmAutoButton.flash()
        self.Automated.set(1-self.Automated.get())  # toggle boolean
        if (self.debug):
            print "<checkAutoUpdate>  Checkpoint 1"
        if (self.Automated.get()==True):
            self.autoButton.configure(text="Auto Update\nEnabled",
                                      bg=self.bg_alt,
                                      fg=self.bg)
            # enable DQM, DBS buttons
            self.dqmAutoButton.configure(state=NORMAL,text="Auto DQM update\nevery %s minutes"%self.dqmAutoUpdateTime.get(),
                                         bg=self.bg,fg=self.fg)
            self.dbsAutoButton.configure(state=NORMAL,text="Auto DBS update\nevery %s minutes"%self.dbsAutoUpdateTime.get(),
                                         bg=self.bg,fg=self.fg)
            self.dbsAutoVar.set(True)
            self.dqmAutoVar.set(True)
            # Start autoUpdater thread
            if (self.debug):
                print "<checkAutoUpdate> Starting autoUpdater thread"
            thread.start_new(self.autoUpdater,())

        else:

            if (self.debug):
                print "<checkAutoUpdate>  Auto update turned off"
            # Boolean false; turn off auto updater
            self.autoButton.configure(text="Auto Update\nDisabled!!",
                                      bg="black",
                                      fg="white")
            self.dqmAutoButton.configure(text="Auto DQM \nupdate OFF",state=DISABLED)
            self.dbsAutoButton.configure(text="Auto DBS \nupdate OFF",state=DISABLED)
            self.dbsAutoVar.set(False)
            self.dqmAutoVar.set(False)

        self.commentLabel.update_idletasks()

        return

    #########################################################
    def heartbeat(self,interval=1):
        '''
        ** self.heartbeat(interval =1 ) **
        Make heartbeat label flash once per second.
        '''

        if (self.debug):
            print self.heartbeat.__doc__
        
        while (self.Automated.get()):
            if (self.hbcolor==self.bg):
                self.hbcolor=self.bg_alt
            else:
                self.hbcolor=self.bg
            self.HeartBeat.configure(bg=self.hbcolor)
            self.commentLabel.update_idletasks()
            time.sleep(interval)

        self.HeartBeat.configure(bg=self.bg)
        return

    #########################################################
    def autoUpdater(self):
        '''
        ** autoUpdate() **
        DQM/DBS Auto updater.
        '''

        if (self.debug):
            print self.autoUpdater.__doc__

        if self.autoRunning==True:
            # Don't allow more than one autoUpdater to run at one time
            # (I don't think this is possible anyway)
            self.commentLabel.configure(text="Auto Updater is already running!")
            self.commentLabel.update_idletasks()
            return
        if self.Automated.get()==False:
            self.commentLabel.configure(text="Auto Updater is disabled")
            self.commentLabel.update_idletasks()
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
            if (self.dbsAutoCounter >= self.dbsAutoUpdateTime.get()):
                # Reset counter if auto dbs disabled
                if (self.dbsAutoVar.get()==False):
                    self.dbsAutoCounter=0
                else:
                    #print "Checking DBS!"
                    if (self.checkDBS()): # search was successful; reset counter
                        self.dbsAutoCounter=0
                        #print "DBS Check succeeded!"
                    else: # search unsuccessful; try again in 1 minute
                        self.dbsAutoCounter=(self.dbsAutoUpdateTime.get()-1)*60
                        #print "DBS Check unsuccessful"

            # repeat for DQM checking
            if (self.dqmAutoCounter >= self.dqmAutoUpdateTime.get()):
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
        ** self.printDBS() **
        Create new window showing DBS values; allow user to change them.
        '''

        if (self.debug):
            print self.printDBS.__doc__
            
        try:
            self.dbsvaluewin.destroy()
            self.dbsvaluewin=Toplevel()
        except:
            self.dbsvaluewin=Toplevel()

        self.dbsvaluewin.title('Change DBS values')
        try:
            maingeom=self.root.winfo_geometry()
            maingeomx=string.split(maingeom,"+")[1]
            maingeomy=string.split(maingeom,"+")[2]
            maingeomx=int(maingeomx)
            maingeomy=int(maingeomy)
            self.dbsvaluewin.geometry('+%i+%i'%(maingeomx+575,maingeomy+275))
        except:
            self.dbsvaluewin.geometry('+600+300')

        myrow=0

        # Variables to be shown in window
        # Add spaces in front of some keys so that they appear
        # first when keys are sorted.
        myvars={"  DBS File Search String = ":self.myDBS.searchStringFile,
                "  DBS Dataset Search String = ":self.myDBS.searchStringDataset,
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
                  bg="white",
                  textvar=myvars[i]).grid(row=myrow,column=1)
            myrow=myrow+1

        # Grid buttons for saving, restoring values
        buttonwin=Frame(self.dbsvaluewin)
        buttonwin.columnconfigure(0,weight=1)
        buttonwin.columnconfigure(1,weight=1)
        buttonwin.columnconfigure(2,weight=1)
        buttonwin.grid(row=myrow,column=0,columnspan=2,sticky=EW)
        Button(buttonwin,text="Save as new default values",
               command = lambda x=self.myDBS:x.writeDefaultsToPickle()).grid(row=0,column=0)
        Button(buttonwin,text="Restore default values",
               command = lambda x=self.myDBS:x.getDefaultsFromPickle()).grid(row=0,
                                                                             column=1)
        Button(buttonwin,text="Close window",
               command = lambda x=self.dbsvaluewin:x.destroy()).grid(row=0,column=2)
        return


    def printDQM(self):
        '''
        ** self.printDQM() **
        Create window for editing DQM values.
        '''

        if (self.debug):
            print self.printDQM.__doc__

        try:
            self.dqmvaluewin.destroy()
            self.dqmvaluewin=Toplevel()
        except:
            self.dqmvaluewin=Toplevel()

        try:
            maingeom=self.root.winfo_geometry()
            maingeomx=string.split(maingeom,"+")[1]
            maingeomy=string.split(maingeom,"+")[2]
            maingeomx=int(maingeomx)
            maingeomy=int(maingeomy)
            self.dqmvaluewin.geometry('+%i+%i'%(maingeomx+375,maingeomy+275))
        except:
            self.dqmvaluewin.geometry('+600+300')

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
                          bg="white",
                          textvar=myvars[i])
            tempEnt.grid(row=myrow,column=1)
            if i=="  Final DQM Save Directory = ": 
                tempEnt.bind("<Return>",(lambda event:self.checkExistence(self.finalDir)))
            elif i== "  .cfg file to run for each DQM = ":
                tempEnt.bind("<Return>",(lambda event:self.checkExistence(self.cfgFileName)))
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
        # Hey!  It seems that values are set whenever you change them in the entry window,
        # even if you don't press <Enter>.  Keen!
        Button(newFrame,text="Exit",
               command = lambda x=self.dqmvaluewin:x.destroy()).grid(row=0,column=2)
        return


    def getDefaultDQMFromPickle(self,startup=False):
        '''
        ** self.getDefaultDQMFromPickle(startup=False) **
        Get DQM default values from .dqmDefaults.cPickle.
        startup variable is set True for first function call on startup.
        (This only affects the message displayed in the comment label after the function call.)
        '''

        if (self.debug):
            print self.getDefaultDQMFromPickle.__doc__

        if os.path.isfile(os.path.join(self.basedir,".dqmDefaults.cPickle")):
            try:
                pcl=open(os.path.join(self.basedir,".dqmDefaults.cPickle"),'rb')
                self.finalDir.set(cPickle.load(pcl))
                self.maxDQMEvents.set(cPickle.load(pcl))
                self.cfgFileName.set(cPickle.load(pcl))
                pcl.close()
            except:
                self.commentLabel.configure(text="Could not read file '.dqmDefaults.cPickle' ")
                self.commentLabel.update_idletasks()
        else:
            if not startup:
                self.commentLabel.configure(text="Sorry, no default values were found")
                self.commentLabel.update_idletasks()
        return

    def writeDefaultDQMToPickle(self):
        '''
        ** self.writeDefaultDQMToPickle **
        Write DQM default values to basedir/.dqmDefaults.cPickle.
        '''

        if (self.debug):
            print self.writeDefaultDQMToPickle.__doc__

        try:
            pcl=open(os.path.join(self.basedir,".dqmDefaults.cPickle"),'wb')
            cPickle.dump(self.finalDir.get(),pcl)
            cPickle.dump(self.maxDQMEvents.get(),pcl)
            cPickle.dump(self.cfgFileName.get(),pcl)
            pcl.close()
            os.system("chmod a+rw %s"%os.path.join(self.basedir,".dqmDefaults.cPickle"))
                      
        except SyntaxError:
            self.commentLabel.configure(text="Could not write file '.dqmDefaults.cPickle' ")
            self.commentLabel.update_idletasks()
        return


    def readPickle(self):
        '''
        ** self.readPickle() **
        Read list of found runs from basedir/.filesInDBS.cPickle.
        '''
        
        if (self.debug):
            print self.readPickle.__doc__
        
        if (self.pickleFileOpen):
            self.commentLabel.configure(text="Sorry, .filesInDBS.cPickle is already open")
            return
        self.pickleFileOpen=True

        if os.path.isfile(os.path.join(self.basedir,".filesInDBS.cPickle")):
            try:
                temp=open(os.path.join(self.basedir,".filesInDBS.cPickle"),'rb')
                self.filesInDBS=cPickle.load(temp)
                self.commentLabel.configure(text = "Loaded previously-read DBS entries from cPickle file")
                self.commentLabel.update_idletasks()
            except:
                self.commentLabel.configure(text="WARNING!  Could not read .filesInDBS.cPickle file!\n-- Starting DBS list from scratch")
                self.filesInDBS={}
        else:
            self.filesInDBS={}
            self.commentLabel.configure(text = "Could not find file .filesInDBS.cPickle\n-- Starting DBS list from scratch")
            self.commentLabel.update_idletasks()

        self.pickleFileOpen=False
        return


    def writePickle(self):
        '''
        ** self.writePickle() **
        Write list of found runs to basedir/.filesInDBS.cPickle.
        '''

        if (self.debug):
            print self.writePickle.__doc__

        if (self.pickleFileOpen):
            self.commentLabel.configure(text="Sorry, could not write information to .filesInDBS.cPickle.\ncPickle file is currently in use.")
            self.commentLabel.update_idletasks()
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
                self.commentLabel.update_idletasks()
        self.pickleFileOpen=False
        return


    def clearPickle(self):
        '''
        ** self.clearPickle() **
        Clear list of found runs, copying .cPickle info to backup file.
        '''

        if (self.debug):
            print self.clearPickle.__doc__

        if not (os.path.isfile(os.path.join(self.basedir,".filesInDBS.cPickle"))):
            self.commentLabel.configure(text="No run list .filesInDBS.cPickle exists!\nThere is nothing yet to clear!")
            self.commentLabel.update_idletasks()
            return
                
        if tkMessageBox.askyesno("Remove .filesInDBS.cPickle?",
                                 "Clearing the list of runs is a major change!\nAre you sure you wish to proceed?"):
            os.system("mv %s %s"%(os.path.join(self.basedir,".filesInDBS.cPickle"),
                                  os.path.join(self.basedir,".backup_filesInDBS.cPickle")))

            self.filesInDBS={} # cleared files in memory
            self.commentLabel.configure(text="Run list cleared (saved as .backup_filesInDBS.cPickle)")
            self.commentLabel.update_idletasks()

        return

    def restoreFromBackupPickle(self):
        '''
        ** self.restoreFromBackupPickle() **
        Restore list of found runs from basedir/.backup_filesInDBS.cPickle file
        '''

        if (self.debug):
            print self.restoreFromBackupPickle.__doc__
        
        if not (os.path.isfile(os.path.join(self.basedir,".backup_filesInDBS.cPickle"))):
            self.commentLabel.configure("Sorry, backup file does not exist!")
            self.commentLabel.update_idletasks()
            return
        if tkMessageBox.askyesno("Restore from .backup_filesInDBS.cPickle",
                                 "Are you sure you want to restore files\nfrom backup?"):
            os.system("mv %s %s"%(os.path.join(self.basedir,".backup_filesInDBS.cPickle"),
                                  os.path.join(self.basedir,".filesInDBS.cPickle")))
            self.readPickle()
            self.commentLabel.configure(text="Run list restored from .backup_filesInDBS.cPickle")
            self.commentLabel.update_idletasks()
        return
    


    def removeFiles(self,removeAll=False):
        '''
        ** self.removeFiles(removeAll=False) **
        Removes hidden files (files starting with "."), such as default option settings, etc.
        If removeAll is set true, then the .filesInDBS.cPickle file that is used to store run history is also removed.
        One exception:  .backup_filesInDBS.cPickle can never be removed via the GUI
        '''

        if (self.debug):
            print self.removeFiles.__doc__
        
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
                  self.commentLabel.update_idletasks()
                  os.system("rm -f %s"%x)
                  time.sleep(0.5)
        return

    def goodQuit(self):
        '''
        ** self.goodQuit() **
        A "clean exit" from the GUI program.
        Checks that DBS/ DQM calls are not currently in progress, and
        then closes the GUI.
        '''

        if (self.debug):
            print self.goodQuit.__doc__
        
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
        ** self.runDQM_thread() **
        Starts new thread for running DQM,
        as long as DQM process is not already running
        '''

        if (self.debug):
            print self.runDQM_thread.__doc__

        if self.runningDQM:
            self.commentLabel.configure(text="Sorry, DQM is already running!")
            self.commentLabel.update_idletasks()
        else:
            thread.start_new(self.runDQM,())
        return


    def runDQM(self):
        '''
        ** self.runDQM() **
        Runs DQM over all found files.
        '''

        if (self.debug):
            print self.runDQM.__doc__
        
        mytime=time.time()
        
        if self.runningDQM:
            self.commentLabel.configure(text="Sorry, DQM is already running")
            self.commentLabel.update_idletasks()
            return

        self.dqmProgress.configure(text="Running DQM on available runs",
                                   bg=self.bg_alt)
        self.dqmStatus.configure(text="%s"%time.strftime("%d %b %Y at %H:%M:%S",time.localtime()))
        
        # Get list of runs -- whenever we change info, we write to pickle file
        # Therefore, read from the file to get the latest & greatest
        self.readPickle() 

        if (self.debug):
            print "<runDQM>  Read pickle file"
        if len(self.filesInDBS.keys())==0:
            self.commentLabel.configure(text = "Sorry, no file info available.\nTry the 'Check DBS for Runs' button first.")
            self.dqmProgress.configure(text="No Run Info available",
                                       bg="black")
            self.commentLabel.update_idletasks()
            return

        # If runs found, sort by run number (largest number first)
        if len(self.filesInDBS.keys()):
            foundruns=self.filesInDBS.keys()
            foundruns.sort()
            foundruns.reverse()
        else:
            self.commentLabel.configure(text="No unprocessed runs found")
            self.commentLabel.update_idletasks()
            return
        
        self.runningDQM=True
        self.dqmButton.configure(state=DISABLED)

        unfinished_run=0
        finished_run=0
        all_run=len(foundruns)
        if (self.debug):
            print "<runDQM>  Set finished, unfinished run vars"
        for i in foundruns:

            if self.filesInDBS[i].ignoreRun==False and self.filesInDBS[i].finishedDQM==False:
                unfinished_run=unfinished_run+1

        newFiles=False # stores whether a new file is found in the midst of DQM processing
        for i in foundruns:
            if (self.debug):
                print "<runDQM> Checking run #%i"%i
            self.commentLabel.configure(text="Running DQM on run #%i"%i)
            self.dqmProgress.configure(text="Running DQM on run #%i"%i,
                                       bg=self.bg_alt)
            self.commentLabel.update_idletasks()
            # Allow user to break loop via setting the runningDQM variable
            # (change to BooleanVar?)
            if (self.debug):
                print "<runDQM> runningDQM bool = ",self.runningDQM
            if (self.runningDQM==False):
                if (self.debug):
                    print "<runDQM> runningDQM bool = False"
                self.dqmButton.configure(state=NORMAL)
                break
            # ignore files if necessary
            if self.filesInDBS[i].ignoreRun:
                continue
            # if DQM started, check to see if DQM has finished
            if self.filesInDBS[i].startedDQM:
                
                # Case 1:  DQM finished; no problem
                if self.filesInDBS[i].finishedDQM:
                    self.filesInDBS[i].previouslyFinishedDQM=True # don't use this variable any more
                    #finished_run=finished_run+1
                    continue

                # Case 2:  DQM not finished; look to see if its output files/dirs exist
                #     Revise this in the future?  Check to see if output was moved to finalDir, but finishedDQM
                #     boolean wasn't set True?  That should never be able to happen, right?

                success=self.getcmsRunOutput(i)
                if not success:
                    print "Problem with Run # %i -- DQM started but did not finish!"%i
                    self.commentLabel.configure(text="Problem with Run # %i -- DQM started but did not finish!"%i)
                    self.commentLabel.update_idletasks()

                else:
                    # files have finished; need to update status
                    self.filesInDBS[i].finishedDQM=True
                    finished_run=finished_run+1
                    continue
            else:
                # nothing started yet; begin DQM

                # First check that cmsRun is available
                if (self.debug):
                    print "<runDQM> looking for cmsRun"
                checkcmsRun=os.popen3("which cmsRun")
                # popen3 returns 3 streams -- in, out, and stderr
                # check that stderr is empty
                if len(checkcmsRun[2].readlines())>0:
                    self.commentLabel.configure(text="Could not find 'cmsRun'\nHave you set up your CMSSW environment?")
                    self.commentLabel.update_idletasks()
                    return

                self.runningDQM=True
                self.filesInDBS[i].startedDQM=True
                # Here is where the cmsRun command is sent!
                if (self.callDQMscript(i)):
                    self.filesInDBS[i].finishedDQM=True
                    finished_run=finished_run+1
                
            if (self.debug):
                print "<runDQM> made it through callDQMscript"

            # Every 20 minutes or so, check for updates to DBS files
            # We shouldn't need this check any more -- users can call
            # DBS on the fly, and set the ignore flags on the files if
            # they want dqm to finish early
            
            if (time.time()-mytime)>20*60:
            #if (1<0):
                if (self.debug):
                    print "<runDQM> getting time info"
                mytime=time.time()
                self.checkDBS()
                if len(self.filesInDBS.keys())<>len(foundruns):
                    self.commentLabel.configure(text="DBS files have been added since last call to DQM.\n  Restarting DQM.")
                    self.commentLabel.update_idletasks()
                    newFiles=True
                    break  # end loop on foundruns
        if (newFiles):
            # Save current progress
            self.writePickle()
            # Could this have caused the bug of .gif files written
            # to directory?
            # What happens if move call occurs during dqm running?
            
            self.dqmButton.configure(state=ACTIVE)
            self.runningDQM=False
            self.runDQM()
            return  # if need to runDQM again, do so, but then return out of this loop
        else:
            self.runningDQM=True


        self.runningDQM=False
        self.writePickle()

        if (finished_run==unfinished_run): # all unfinished_runs are now finished
            self.dqmProgress.configure(text="Successfully finished running DQM",
                                       bg="black")
        else:
            self.dqmProgress.configure(text="Ran DQM on %i/%i runs"%(finished_run, unfinished_run))
        self.dqmStatus.configure(text="%s"%time.strftime("%d %b %Y at %H:%M:%S",time.localtime()))
        self.commentLabel.configure(text="Finished running DQM:\n%i out of %i runs successfully processed"%(finished_run,unfinished_run))
        time.sleep(3)
        self.tempSCP() # Call scp copying routine once dqm has finished
        self.dqmButton.configure(state=NORMAL)
        
        self.commentLabel.update_idletasks()
        return  #end of runDQM
                

    def callDQMscript(self,i):
        '''
        ** self.callDQMscript(i) **
        Here is where we actually perform the cmsRun call for the given run number `i`.
        '''

        if (self.debug):
            print self.callDQMscript.__doc__
        
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
            self.commentLabel.update_idletasks()
            time.sleep(2)
            return False
        
        # Allow a different # for each file?
        #temp.write("replace maxEvents.input=%i\n"%self.filesInDBS[i].maxEvents)

        temp.write("replace maxEvents.input=%i\n"%self.maxDQMEvents.get())
        filelength=len(self.filesInDBS[i].files)
        if (filelength==0):
            self.commentLabel.configure(text = "<ERROR> No files found for run %i!"%i)
            self.commentLabel.update_idletasks()
            time.sleep(2)
            return
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
            self.commentLabel.update_idletasks()
            time.sleep(2)
            return False

        os.system("cmsRun %s"%self.cfgFileName.get())


        # Get list of output files, directories produced by cmsRun call
        success=self.getcmsRunOutput(i)
        if (success):
            for myobject in self.cmsRunOutput:

                if (
                    self.finalDir.get()<>self.basedir
                    ):

                    # Delete old copies of output (stored in self.finalDir.get() directory)
                    if os.path.exists(os.path.join(self.finalDir.get(),os.path.basename(myobject))):
                        os.system("rm -rf %s"%os.path.join(self.finalDir.get(),os.path.basename(myobject)))
                    # now move object to final directory
                    os.system("mv %s %s"%(myobject,os.path.join(self.finalDir.get(),os.path.basename(myobject))))

                    self.commentLabel.configure(text = "moved %s\n to directory %s"%(myobject,
                                                                                     self.finalDir.get()))
                    self.commentLabel.update_idletasks()
                else:
                    self.commentLabel("ERROR -- Can't find %s\nDo you know where your output is?"%myobject)
                    self.commentLabel.update_idletasks()
                time.sleep(3)

                # Call scp at the completion of each run?
                self.tempSCP()


        else: # success = False
            self.commentLabel("ERROR -- did not retrieve all expected output from cmsRun")
            self.commentLabel.update_idletasks()
            time.sleep(3)
            
        if self.debug:
            print "<CallDQMScript> Success = %s"%success
        
        return success


        
    def getcmsRunOutput(self,runnum):
        '''
        ** self.getcmsRunOutput(runnum) **
        Looks for all the output files, directories that should be
        produced by cmsRun call for run #(runnum).
        Returns a boolean =  whether or not all files and directories have
        been found.
        '''

        if (self.debug):
            print self.getcmsRunOutput.__doc__

        success=True
        
        self.cmsRunOutput=[]
        if (runnum<100000):
            outname="DQM_Hcal_R0000%i"%runnum
        else:
            outname="DQM_Hcal_R000%i"%runnum

        # make fancier success requirement later -- for now, just check that directory exists
        if (self.debug):
            print "%s exists? %i"%(os.path.join(self.basedir,outname),os.path.isdir(os.path.join(self.basedir,outname)))

        outputdir=os.path.join(self.basedir,outname)

        success=success and (os.path.isdir(outputdir))
        # if directory exists, add it to cmsRunOutput
        if (success):
            if (self.debug):
                print "<getcmsRunOutput> success=True!"
            self.cmsRunOutput.append(outputdir)

        # now check that root file exists
        outputroot="%s.root"%(os.path.join(self.basedir,outname))
        success=success and (os.path.exists(outputroot))
        if os.path.exists(outputroot):
            self.cmsRunOutput.append(outputroot)

        if (self.debug):
            print "<getcmsRunOutput>  The following cmsRun outputs were found:"
            for i in self.cmsRunOutput:
                print "\t%s"%i
            print "\nAll files found? %s"%success
            
        return success




    def checkDBS(self):
        '''
        ** self.checkDBS() **
        Looks in DBS for files with given file/dataset names, and in specified run range.
        '''

        if (self.debug):
            print self.checkDBS.__doc__

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
        self.commentLabel.update_idletasks()
        self.myDBS.searchDBS(begin,end) # Search, getting run numbers

        if (self.parseDBSInfo()):
            self.commentLabel.configure(text="Finished checking DBS runs (%i-%i)\nFound a total of %i files"%(begin,end,self.foundfiles))

        self.dbsSearchInProgress=False
        self.dbsButton.configure(state=NORMAL)
        return True


    def parseDBSInfo(self):
        '''
        ** self.parseDBSInfo() **
        Once we've checked DBS, let's parse the output for runs!
        We'll get info by checking DBS for all run numbers within range.
        Then, for each found run number, we'll grab all the files for that run number
        '''

        if (self.debug):
            print self.parseDBSInfo.__doc__

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
                    self.commentLabel.update_idletasks()
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
            self.commentLabel.update_idletasks()
            return False


        # Now loop over each run to get final search result

        self.foundfiles=0
        badcount=0
        for r in runlist:
                        
            self.dbsProgress.configure(text="Found run %i in range (%i-%i)..."%(r,self.lastFoundDBS.get(),self.lastFoundDBS.get()+self.dbsRange.get()))
            self.commentLabel.update_idletasks()

            tempfiles=[]

            # For each run, create new accessor that will find files, datasets associated with the run
            x=dbsAccessor(debug=self.debug)
            x.host.set(self.myDBS.host.get())
            x.port.set(self.myDBS.port.get())
            x.dbsInst.set(self.myDBS.dbsInst.get())
            x.searchStringFile.set(self.myDBS.searchStringFile.get())
            x.searchStringDataset.set(self.myDBS.searchStringDataset.get())
            x.page.set(self.myDBS.page.get())
            x.limit.set(self.myDBS.limit.get())
            x.xml.set(self.myDBS.xml.get())
            x.case.set(self.myDBS.case.get())
            x.details.set(self.myDBS.details.get())
            x.debug.set(self.myDBS.debug.get())
            # beginRun, endRun don't need to be set -- we're looking for one specific run here
            
            # give new accessor same defaults as set for self.myDBS
            text="find file,dataset where %s run=%i"%(self.myDBS.formParsedString(),r)
            
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
                        self.commentLabel.update_idletasks()
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
                             #print "Could not parse DBS entry: %s"%i
                             self.commentLabel.update_idletasks()

                    except:
                        self.commentLabel.configure(text="Could not parse DBS entry:\n'%s'"%i)
                        badcount=badcount+1
                        #print "Could not parse DBS entry: %s"%i
                        self.commentLabel.update_idletasks()

            tempDBS=DBSRun(tempfiles)
            tempDBS.runnum=r
            tempDBS.maxEvents=self.maxDQMEvents.get()
            if (dataset<>None):
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
            self.commentLabel.update_idletasks()
            return False

        if badcount:
            self.dbsProgress.configure(text="%i lines from DBS could not be parsed!"%badcount)
            self.commentLabel.update_idletasks()
            return False
        
        self.dbsProgress.configure(text="Successfully grabbed runs %i-%i"%(begin,end),
                                   bg="black")
        self.commentLabel.update_idletasks()
        
        return True
    


    def displayFiles(self):
        '''
        ** self.displayFiles() **
        Show all run numbers that have been found from DBS, along with the
        DQM status of each run.
        '''

        if (self.debug):
            print self.displayFiles.__doc__

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
        ** self.changeFileSettings() **
        Allows user to change the DQM status of the runs found from DBS.
        (Mark runs as having already completed DQM, set ignoreRun true, etc.)
        '''

        if (self.debug):
            print self.changeFileSettings.__doc__

        # If window exists already, destroy it and recreate 
        try:
            self.changevaluewin.destroy()
            self.changevaluewin=Toplevel()
        except:
            self.changevaluewin=Toplevel()

        maingeom=self.root.winfo_geometry()
        maingeomx=string.split(maingeom,"+")[1]
        maingeomy=string.split(maingeom,"+")[2]
        try:
            maingeomx=string.atoi(maingeomx)
            maingeomy=string.atoi(maingeomy)
            # new window is ~380 pixels high, place it directly below main window (250 pix high),
            # if room exists
            if (self.root.winfo_screenheight()-(maingeomy+250)>350):
                self.changevaluewin.geometry('+%i+%i'%(maingeomx,maingeomy+250))
            elif (maingeomy>380):
                self.changevaluewin.geometry('+%i+%i'%(maingeomx,maingeomy-380))
        except:
            self.changevaluewin.geometry('+500+320')


        self.changevaluewin.title("Change status of files")
        
        # Add list of runs as a list box with attached scrollbar

        self.changevaluewin.rowconfigure(0,weight=1)
        self.changevaluewin.columnconfigure(0,weight=1)
        scrollwin=Frame(self.changevaluewin)
        scrollwin.grid(row=0,column=0,sticky=NSEW)
        scrollwin.rowconfigure(1,weight=1)
        scrollwin.columnconfigure(0,weight=1)
        myrow=0
        temp = "%s%s%s%s%s%s"%(string.ljust(" Run #",20),
                               string.ljust("Started DQM?",20),string.ljust("Finished DQM?",20),
                               string.ljust("IgnoreRun?",20),string.ljust("# of files",20),
                               string.ljust("Dataset",30))
        Label(scrollwin,
              text=temp).grid(row=myrow,column=0)
        myrow=myrow+1
        self.lb=Listbox(scrollwin,
                        bg="white",
                        selectmode = MULTIPLE)
        # Get list of runs
        self.listboxruns=self.filesInDBS.keys()
        self.listboxruns.sort()
        self.listboxruns.reverse()

        for i in self.listboxruns:
            self.lb.insert(END,self.filesInDBS[i].Print2())
            
        scroll=Scrollbar(scrollwin,command=self.lb.yview)
        
        self.lb.configure(yscrollcommand=scroll.set)

        xscroll=Scrollbar(scrollwin,command=self.lb.xview,
                          orient=HORIZONTAL)
        self.lb.configure(xscrollcommand=xscroll.set)
        
        self.lb.grid(row=myrow,column=0,sticky=NSEW)
        scroll.grid(row=myrow,column=1,sticky=NS)
        myrow=myrow+1
        xscroll.grid(row=myrow,column=0,sticky=EW)

        # Add buttons for changing DQM values
        myrow=myrow+1
        #self.changevaluewin.rowconfigure(myrow,weight=1)
        bFrame=Frame(self.changevaluewin)
        bFrame.grid(row=1,column=0)
        igY=Button(bFrame,
                   text="Set\n'Ignore Run'\nTrue",
                   command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                     "ignoreRun",True),
                   width=14,height=3)
        igN=Button(bFrame,
                   text="Set\n'Ignore Run'\nFalse",
                   command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                     "ignoreRun",False),
                   width=14,height=3)
        stY=Button(bFrame,
                   text="Set\n'Started DQM'\nTrue",
                    command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                      "startedDQM",True),
                   width=14,height=3)
        stN=Button(bFrame,
                   text="Set\n'Started DQM'\nFalse",
                   command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                     "startedDQM",False),
                   width=14,height=3)
        fiY=Button(bFrame,
                   text="Set\n'Finished DQM'\nTrue",
                   command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                     "finishedDQM",True),
                   width=14,height=3)
        fiN=Button(bFrame,
                   text="Set\n'Finished DQM'\nFalse",
                   command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                     "finishedDQM",False),
                   width=14,height=3)
        selAll=Button(bFrame,
                      text="Select\nall runs",
                      command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                        "selectall",True),
                      width=14,height=3)
        deselAll=Button(bFrame,
                        text="Deselect\nall runs",
                        command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                          "deselectall",False),
                        width=14,height=3)
        dbsSearch=Button(bFrame,
                         text="Search\nDBS for new\nruns",
                         bg=self.bg,
                         fg=self.fg,
                         command=lambda x=self:x.commandChangeFileSettings(self.lb.curselection(),
                                                                           "searchDBS",False))

        quitButton=Button(bFrame,
                          text="Close\nwindow",
                          bg=self.bg_alt,
                          fg=self.bg,
                          command=lambda x=self:x.changevaluewin.destroy())
        # Grid buttons
        igY.grid(row=0,column=3)
        stY.grid(row=0,column=1)
        fiY.grid(row=0,column=2)
        igN.grid(row=1,column=3)
        stN.grid(row=1,column=1)
        fiN.grid(row=1,column=2)
        selAll.grid(row=0,column=4)
        deselAll.grid(row=1,column=4)
        dbsSearch.grid(row=0,column=0,rowspan=2,sticky=NS)
        quitButton.grid(row=0,column=5,rowspan=2,sticky=NS)

        return


    def updateListbox(self):
        '''
        ** self.updateListbox() **
        Grabs updated run info from self.filesInDBS.keys
        and displays it in listbox.
        '''

        if (self.debug):
            print self.updateListbox.__doc__

        self.lb.delete(0,END)
        temp = "%s%s%s%s%s%s"%(string.ljust(" Run #",20),
                               string.ljust("Started DQM?",20),string.ljust("Finished DQM?",20),
                               string.ljust("IgnoreRun?",20),string.ljust("# of files",20),
                               string.ljust("Dataset",80))
        #self.lb.insert(END,temp)
        # Get list of runs
        self.listboxruns=self.filesInDBS.keys()
        self.listboxruns.sort()
        self.listboxruns.reverse()

        for i in self.listboxruns:

            self.lb.insert(END,self.filesInDBS[i].Print2())
        return
        
    def commandChangeFileSettings(self,selected,var,value=True):
        '''
        ** self.commandChangeFileSettings(selected, var, value=True) **
        Commands for changing DQM settings.
        "selected" is the set of listbox indices that have been
        highlighted by the user.
        (self.listboxruns[int(i)] returns the associated run #, for
         all i in selected.)
        Allowed options for var:
        "ignoreRun", "startedDQM", "finishedDQM", "selectall", "deselectall"
        Value indicates whether var should be set True or False.  Default is True.
        '''

        if (self.debug):
            print self.commandChangeFileSettings.__doc__

        if (var=="selectall"):
            for i in range(0,self.lb.size()):
                self.lb.selection_set(i)
            return
        elif (var=="deselectall"):
            for i in range(0,self.lb.size()):
                self.lb.selection_clear(i)
            return
        
        for i in selected:
            run=self.listboxruns[int(i)] # get run number from index

            if (var=="ignoreRun"):
                self.filesInDBS[run].ignoreRun=value
            elif (var=="startedDQM"):
                self.filesInDBS[run].startedDQM=value
            elif (var=="finishedDQM"):
                self.filesInDBS[run].finishedDQM=value
            
        if (var=="searchDBS"):
            self.checkDBS()
        self.writePickle() # save to pickle file?  I think this is the sensible option (user can always change back)
        self.updateListbox()
        return


    def toggleSCP(self):
        
        '''
        ** self.toggleSCP() **
        configures scp label based on self.enableSCP value.
        If SCP variable is off, no scp copying will take place.'''

        if (self.debug):
            print self.toggleSCP.__doc__

        if (self.enableSCP.get()==0):
            self.scpAutoButton.configure(text="scp copying disabled")
            self.copyLoc.configure(state=DISABLED)
        else:
            self.scpAutoButton.configure(text="scp copying enabled")
            self.copyLoc.configure(state=NORMAL)
        return

    
    def toggleAutoDBS(self):
        '''
        ** self.toggleSCP() **
        Changes DBS Auto update label based on state of self.dbsAutoVar.
        '''

        if (self.debug):
            print self.toggleAutoDBS.__doc__

        if (self.Automated.get()==False):
            self.dbsAutoButton.configure(text="Auto DQM\nupdate OFF",state=DISABLED)
            return
        
        if self.dbsAutoVar.get()==0 :
            self.dbsAutoButton.configure(text="Auto DBS\nupdate OFF")
        else:
            self.dbsAutoButton.configure(text="Auto DBS update\nevery %s minutes"%self.dbsAutoUpdateTime.get())
                                     
        return

    def toggleAutoDQM(self):
        '''
        ** self.toggleAutoDQM() **
        Changes DQM Auto update label based on state of self.dqmAutoVar.
        '''

        if (self.debug):
            print self.toggleAutoDQM.__doc__

        if (self.Automated.get()==False):
            self.dqmAutoButton.configure(text="Auto DQM\nupdate OFF",state=DISABLED)
            return
        if self.dqmAutoVar.get()==0 :
            self.dqmAutoButton.configure(text="Auto DQM\nupdate OFF")
        else:
            self.dqmAutoButton.configure(text="Auto DQM update\nevery %s minutes"%self.dqmAutoUpdateTime.get())
                                     
        return


    def toggleAutoRunShift(self,event):
        '''
        ** self.toggleAutoRunShift(event) **
        This toggles the autoRunShift variable.
        If autoRunShift is true, then the run entry
        value will increment whenever a new run is found.
        If not, the run entry value will remain the same.
        '''

        if (self.debug):
            print self.toggleAutoRunShift.__doc__
        
        self.autoRunShift=1-self.autoRunShift # toggle value
        # Change entry box color if auto shifting is not enabled
        if (self.autoRunShift==False):
              self.lastFoundDBSEntry.configure(bg="yellow")
        else:
            self.lastFoundDBSEntry.configure(bg="white")
        return


    def checkExistence(self,obj):
        '''
        ** self.checkExistence(obj) **
        Checks to see whether file/dir "obj" exists.
        Returns true/false boolean based on object existence.

        '''

        if (self.debug):
            print self.checkExistence.__doc__

        print obj.get()
        exists=True
        if not os.path.exists(obj.get()):
            self.commentLabel.configure(text="ERROR!\n Object '%s' does not exist!"%obj.get())
            self.commentLabel.update_idletasks()
            obj.set("ERROR -- FILE/DIR DOES NOT EXIST")
            exists=False
        else:
            self.commentLabel.configure(text="Set value to '%s'"%obj.get())
            self.commentLabel.update_idletasks()
        return exists

    def tempSCP(self):
        '''
        ** self.tempSCP() **
        Temporary method for running scp from local final directory
        to hcalusc55@cmshcal01:hcaldqm/global_auto/

        '''

        if (self.debug):
            print self.tempSCP.__doc__

        if not (self.enableSCP.get()):
            self.commentLabel.configure(text="scp copying is not currently enabled.\n(Check button in the middle of the menu bar)")
            self.commentLabel.update_idletasks()
            return
        
        if not (os.path.exists(self.finalDir.get())):
            self.commentLabel.configure(text="ERROR -- directory '%s' DOES NOT EXIST!!\nEdit the Final DQM Save Directory in DQM options!"%self.finalDir.get())
            return

        self.commentLabel.configure(text="Trying to scp results to cmshcal01")
        self.commentLabel.update_idletasks()
        
        # make directory for files/dirs that have already been copied.
        if not os.path.isdir(os.path.join(self.finalDir.get(),"copied_to_hcaldqm")):
            os.mkdir(os.path.join(self.finalDir.get(),"copied_to_hcaldqm"))

        movelist=os.listdir(self.finalDir.get())
        movelist.remove("copied_to_hcaldqm")
        if len(movelist)==0:
            self.commentLabel.configure(text="There are no files in %s\n to be copied to cmshcal01!"%self.finalDir.get())
            self.commentLabel.update_idletasks()
            return
        text1="scp -r "
        text="scp -r "
        for i in movelist:
            text=text+"%s "%os.path.join(self.finalDir.get(),"copied_to_hcaldqm",i)
            text1=text1+"%s "%os.path.join(self.finalDir.get(),i)
        text=text+" hcalusc55@cmshcal01:/hcaldqm/global_auto\n\n"
        text1=text1+" hcalusc55@cmshcal01:/hcaldqm/global_auto\n\n"

        
        #if at cms (specifically, on lxplus or caf (lxb...)):
        #if os.getenv("USER")=="cchcal":
        compname=os.uname()[1]

        if (string.find(compname,"lxplus")>-1 or string.find(compname,"lxb")) and string.find(compname,".cern.ch")>-1:
            zzz=os.system(text1)
            print text1
            #print zzz
            self.commentLabel.configure(text = "FINISHED!\nPerformed scp of files to cmshcal01!")
            self.commentLabel.update_idletasks()
        
        else:  # not at cern
            helpfunctions.Helpwin(text,usetext=1,title="Cut and paste this command into your lxplus window now!" )
            self.commentLabel.configure(text="Cannot auto-scp to cmshcal from your machine!\nFollow instructions in the help window!")
            self.commentLabel.update_idletasks()
            
        # move files to the copied_to_hcaldqm subdirectory (so they won't be scp'd again)
        for i in movelist:
            if os.path.isdir(os.path.join(self.finalDir.get(),"copied_to_hcaldqm",i)):
                os.system("rm -rf %s"%os.path.join(self.finalDir.get(),"copied_to_hcaldqm",i))
            cmd="mv %s %s\n"%(os.path.join(self.finalDir.get(),i),
                              os.path.join(self.finalDir.get(),"copied_to_hcaldqm",i))
            os.system(cmd)

        return



    def dbsSetUpdateMenu(self,upTime,allTimes):
        '''
        ** self.dbsUpdateMenu(upTime, allTimes) **
        Sets colors in "Set DBS Option Time" menu, and configures the DBS auto button label.
        allTimes = list of all time options specified in the menu
        upTime = list index of chosen time.
        Chosen time appears in red; all other times in black
        '''

        if (self.debug):
            print self.dbsSetUpdateMenu.__doc__

        self.dbsAutoUpdateTime.set(allTimes[upTime])
        for i in range(len(allTimes)):
            if i==upTime:
                self.dbsUpdateMenu.choices.entryconfig(i,foreground="red")
            else:
                self.dbsUpdateMenu.choices.entryconfig(i,foreground="black")
        if (self.Automated.get()==True):
            self.dbsAutoButton.configure(state=NORMAL,text="Auto DBS update\nevery %s minutes"%self.dbsAutoUpdateTime.get(),
                                         bg=self.bg,fg=self.fg)
        return


    def dqmSetUpdateMenu(self,upTime,allTimes):
        '''
        ** self.dqmUpdateMenu(upTime, allTimes) **
        Sets colors in "Set DQM Option Time" menu, and configures the DQM auto button label.
        allTimes = list of all time options specified in the menu
        upTime = list index of chosen time.
        Chosen time appears in red; all other times in black
        '''

        if (self.debug):
            print self.dqmSetUpdateMenu.__doc__

        
        self.dqmAutoUpdateTime.set(allTimes[upTime])
        for i in range(len(allTimes)):
            if i==upTime:
                self.dqmUpdateMenu.choices.entryconfig(i,foreground="red")
            else:
                self.dqmUpdateMenu.choices.entryconfig(i,foreground="black")
        if (self.Automated.get()==True):
            self.dqmAutoButton.configure(state=NORMAL,text="Auto DQM update\nevery %s minutes"%self.dqmAutoUpdateTime.get(),
                                         bg=self.bg,fg=self.fg)
        return


############################################

if __name__=="__main__":

    mygui=dbsBaseGui(debug=1)  # set up gui
    mygui.DrawGUI()
    mygui.root.mainloop() # run main loop



