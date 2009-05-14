#!/usr/bin/env python

import sys
try:
    from ROOT import *
except:
    print "ERROR:  Cannot import ROOT!"
    print "Make sure you are in a CMSSW release area, and have executed 'cmsenv'."
    sys.exit()
from urllib import urlopen
import sys
from Tkinter import *
import os
import string
from array import array


def convertID(ID):
    try:
        id=int(ID,16)
        eta=(id>>7)&0x3f
        if (id&0x2000==0):
            eta=eta*-1
        phi=id&0x7f
        depth=(id>>14)&0x7
        subdet=(id>>25)&0x7
        subdetmap={1:"HB",2:"HE",3:"HO",4:"HF"}
        if (subdet in subdetmap.keys()):
            subdet=subdetmap[subdet]
        else:
            subdet="Unknown"
        name="%s(%i,%i,%i)"%(subdet,eta,phi,depth)
    except:
        name="???"
    return name

def calcDetID(ID):
    ''' expect ID in form <subdet> (eta,phi,depth) '''
    ID=string.replace(ID,","," ")
    ID=string.replace(ID,")"," ")
    ID=string.replace(ID,"("," ")
    ID=string.split(ID)
    subdetmap={"HB":1,"HE":2,"HO":3,"HF":4}
    subdet=ID[0]
    if subdet not in subdetmap.keys():
        print "ERROR in calcDetID -- can't ID subdet '%s'"%subdet
        return
    subdet=subdetmap[subdet]
    eta=string.atoi(ID[1])
    phi=string.atoi(ID[2])
    depth=string.atoi(ID[3])
    etaplus=True
    if (eta<0):        
        eta=abs(eta)
        etaplus=False
    # Now calculate detID
    detid=4<<28
    detid=detid|((subdet&0x7)<<25)
    detid=detid|((depth&0x7)<<14)
    detid=detid|((eta&0x3f)<<7)
    if (etaplus):
        detid=detid|(0x2000)
    detid=detid|(phi&0x7f)
    name="%x"%detid
    name=string.upper(name)
    return name



class CellStat:

    def __init__(self,ID=None):
        self.ID=ID
        self.IDstring=None
        self.eta=None
        self.phi=None
        self.depth=None
        self.subdet=None
        self.subdetmap={1:"HB",2:"HE",3:"HO",4:"HF"}
        self.parseID()
        
        self.Alwayshot=False
        self.Alwaysdead=False
        
        self.status={}
        return

    # Need to changes these once HcalChannelStatus is updated (should be shifting 5 and 6 bits, respectively, not 4 and 5)
    def Dead(self,run):
        return ((self.status[run]>>4)&0x1)

    def Hot(self,run):
        return ((self.status[run]>>5)&0x1)

    def AlwaysHot(self):
        count=0
        hot=0
        for i in self.status.keys():
            if (self.status[i]&0x1==0): # only count runs where cell is present
                count=count+1
                if ((self.status[i]>>5)&0x1):
                    hot=hot+1

        if (count>0 and hot>=0.90*count):
            self.Alwayshot=True
        return self.Alwayshot
    
    def AlwaysDead(self):
        count=0
        dead=0
        for i in self.status.keys():
            # bit 1 is a disabled/not present bit
            if (self.status[i]&0x1==0): # only count urns where cell is present
                count=count+1
                if ((self.status[i]>>4)&0x1):
                    dead=dead+1
        #if (self.status[i]):
        #    print "count = ",count,"  DEAD = ",dead
        if (count>0 and dead>=0.90*count):
            self.Alwaysdead=True
        return self.Alwaysdead

    def read(self,run,text,hot=False,dead=False):
        temp=string.split(text)
        try:
            value=string.atoi(temp[4])
            ID=temp[5]
            
        except:
            print "Could not parse line '%s'"%text
            return
        if (self.ID==None):
            self.ID=ID
            self.parseID()
        elif (self.ID!=ID):
            print "Error, ID value mismatch"
            return
        if (hot):
            value=value*16
        elif (dead):
            value=value*32
        if run in self.status.keys():
            self.status[run]=(self.status[run]&value)
        else:
            self.status[run]=value
        return

    def parseID(self):
        id=int(self.ID,16)
        eta=(id>>7)&0x3f
        if (id&0x2000==0):
            eta=eta*-1
        phi=id&0x7f
        depth=(id>>14)&0x7
        subdet=(id>>25)&0x7
        if (subdet in self.subdetmap.keys()):
            subdet=self.subdetmap[subdet]
        else:
            subdet="Unknown"
        name="%s(%i,%i,%i)"%(subdet,eta,phi,depth)
        self.eta=eta
        self.phi=phi
        self.depth=depth
        self.subdet=subdet
        self.IDstring=name
        return
        
###

class RunStatusGui:

    def __init__(self,parent=None,debug=False):

        self.debug=debug
        self.Cells={}
        self.Runs=[]
        self.HotCellList=[]
        self.DeadCellList=[]
        self.AlwaysHotList=[]
        self.AlwaysDeadList=[]

        # Set Main Frame
        if (parent==None):
            self.root=Tk()
        else:
            self.root=parent

        self.root.title("Bad Cell Checker")

        # Set Size of TCanvas plot based on monitor screen size
        self.screenwidth=self.root.winfo_screenwidth()
        self.screenheight=self.root.winfo_screenheight()
        # Speficy height, width of canvas here
        self.canwidth=700
        self.canheight=500
        if (self.screenwidth<1000):
            self.canwidth=500
        if (self.screenheight<800):
            self.canheight=350
        
        # Set variables needed by GUI
        self.webname=StringVar()
        self.startrun=IntVar()
        self.endrun=IntVar()
        self.CellID=StringVar()
        
        self.root.columnconfigure(0,weight=1)
        row=0
        self.bg="grey80"
        self.MenuFrame=Frame(self.root, 
                             bg=self.bg)
        self.MenuFrame.grid(row=row,
                            column=0,
                            sticky=EW)

        row=row+1
        self.WebFrame=Frame(self.root, 
                            bg=self.bg)
        self.WebFrame.grid(row=row,
                           column=0,
                           sticky=EW)
        row=row+1
        self.RunRangeFrame=Frame(self.root, 
                                 bg=self.bg)
        self.RunRangeFrame.grid(row=row,
                                column=0,
                                sticky=EW)
        
        row=row+1
        self.MainFrame=Frame(self.root, 
                             bg=self.bg)
        self.MainFrame.grid(row=row,
                            column=0,
                            sticky=NSEW)
        self.root.rowconfigure(row,weight=1)
        row=row+1
        self.CommentFrame=Frame(self.root, 
                                bg=self.bg)
        self.CommentFrame.grid(row=row,
                               column=0,
                               sticky=EW)
        

        # Now fill the frames
        self.makeCommentFrame() # start here so that comments can be added for other methods
        self.makeWebFrame()
        self.makeRunRangeFrame()
        self.makeMainFrame()
        
        return

    def makeWebFrame(self):
        ''' Creates labels and entries for setting web page where cell status is kept '''

        self.webname.set('https://cms-project-hcal-dqm.web.cern.ch/cms-project-hcal-dqm/data/dqm/')
        Label(self.WebFrame,
              text="Web Address: ",
              bg=self.bg).grid(row=0,column=0)

        self.WebFrame.columnconfigure(1,weight=1)
        self.webEntry=Entry(self.WebFrame,
                            bg="black",
                            foreground="white",
                            textvar=self.webname)
        self.webEntry.grid(row=0,column=1,sticky=EW)
        return

    def makeRunRangeFrame(self):
        ''' Creates labels and entries that allow user to search over given run ranges '''

        self.startrun.set(70362)
        self.endrun.set(70363)
        
        Label(self.RunRangeFrame,
              text="Search Runs From: ",
              bg=self.bg).grid(row=0,column=0)
        self.startEntry=Entry(self.RunRangeFrame,
                              bg="black",
                              foreground="white",
                              textvar=self.startrun)
        self.RunRangeFrame.columnconfigure(1,weight=1)
        self.startEntry.grid(row=0,column=1,
                             sticky=EW)
        Label(self.RunRangeFrame,
              text = "to ",
              width=10,
              bg=self.bg).grid(row=0, column=2)
        self.endEntry=Entry(self.RunRangeFrame,
                            bg="black",
                            foreground="white",
                            textvar=self.endrun)
        self.RunRangeFrame.columnconfigure(3,weight=1)
        self.endEntry.grid(row=0,column=3,
                           sticky=EW)
        runsubframe=Frame(self.RunRangeFrame,
                          bg=self.bg)
        runsubframe.grid(row=1,column=0,columnspan=4,sticky=EW)
        self.TestButton = Button(runsubframe,
                                 height=2,
                                 bg="white",
                                 fg="black",
                                 text="Read Status from Web",
                                 command = lambda x=self:x.GetInfo())
        self.TestButton.grid(row=0,column=0,columnspan=3)
        return

    def makeMainFrame(self):
        ''' Create subframes for main frame.'''
        self.ChooserFrame=Frame(self.MainFrame,bg=self.bg)
        self.PicFrame=Frame(self.MainFrame,bg=self.bg)
        self.DisplayFrame=Frame(self.MainFrame,bg=self.bg)

        self.ChooserFrame.grid(row=0,column=0,sticky=NS)
        self.PicFrame.grid(row=0, column=1,sticky=NSEW)
        self.DisplayFrame.grid(row=0, column=2,sticky=NS)
        self.MainFrame.rowconfigure(0,weight=1)
        self.MainFrame.columnconfigure(1,weight=1)

        self.makeChooserFrame()
        self.makePicFrame()
        self.makeDisplayFrame()
        return

        
    def makeChooserFrame(self):
        ''' Provide entries to choose a cell to display.'''
        self.ShowAll=Button(self.ChooserFrame,
                            text="Back to\nCells vs. Time\n plot",
                            bg="white",
                            foreground="black",
                            command=lambda x=self:x.DrawCells())
        self.ShowAll.grid(row=0,column=1,pady=50,sticky=EW)
        Label(self.ChooserFrame,
              text="Choose Cell:",
              bg=self.bg).grid(row=1,column=0,columnspan=2)
        Label(self.ChooserFrame,
              text="Cell ID:",
              bg=self.bg).grid(row=2,column=0)
        self.ChooseCells=Entry(self.ChooserFrame,
                               textvar=self.CellID,
                               bg="white",
                               foreground="black"
                               )
        self.ChooseCells.bind('<Return>',lambda event:self.DrawCellStatus())

        self.ChooseCells.grid(row=2,column=1)
        return

    def makePicFrame(self):
        self.PicLabel=Label(self.PicFrame)
        if os.path.isfile("badcelldisplay_file.gif"):
            GUIimage=PhotoImage(file="badcelldisplay_file.gif")
        else:
            GUIimage=PhotoImage(width=self.canwidth, height=self.canheight)
        self.PicLabel.image=GUIimage
        self.PicLabel.configure(image=GUIimage)
        self.PicLabel.update()

        print self.PicLabel['width']
        self.PicFrame.rowconfigure(0,weight=1)
        self.PicFrame.columnconfigure(0,weight=1)
        self.PicLabel.grid(row=0,column=0,sticky=NSEW)
        return

    def makeDisplayFrame(self):
        for i in range(0,4):
            self.DisplayFrame.rowconfigure(i,weight=1)
        self.HotButton=Button(self.DisplayFrame,
                              text="List Hot Cells",
                              height=2,
                              width=20,
                              bg="white",
                              fg="black",
                               command = lambda x=self:x.printCellList(x.HotCellList))
        self.HotButton.grid(row=0,column=0)
        self.AlwaysHotButton=Button(self.DisplayFrame,
                                    text="List Always\nHot Cells",
                                    height=2,
                                    width=20,
                                    bg="white",
                                    fg="black",
                                    command = lambda x=self:x.printCellList(x.AlwaysHotList))
        self.AlwaysHotButton.grid(row=1,column=0)
        self.DeadButton=Button(self.DisplayFrame,
                               text="List Dead Cells",
                               height=2,
                               width=20,
                               bg="white",
                               fg="black",
                               command = lambda x=self:x.printCellList(x.DeadCellList) )
        self.DeadButton.grid(row=2,column=0)
        self.AlwaysDeadButton=Button(self.DisplayFrame,
                                     text="List Always\nDead Cells",
                                     height=2,
                                     width=20,
                                     bg="white",
                                     fg="black",
                                     command = lambda x=self:x.printCellList(x.AlwaysDeadList) )
        self.AlwaysDeadButton.grid(row=3,column=0)
        return
    
    def makeCommentFrame(self):
        ''' Creates comment label.'''
        self.commentLabel=Label(self.CommentFrame,
                                text="Hcal Cell Status Checker",
                                bg=self.bg)
        self.CommentFrame.columnconfigure(0,weight=1)
        self.commentLabel.grid(row=0,column=0,
                               sticky=EW)
        return

    def GetInfo(self):
        # Reset values
        self.Cells={}
        self.Runs=[]
        self.Print("Searching web page for files...")
        try:
            www=self.webname.get()
            out=urlopen(www).readlines()
        except IOError:
            if not (www.startswith("http://")):
                try:
                    www="http://"+www
                    self.webname.set(www)
                    out=urlopen(www).readlines()
                except:
                    self.Print("ERROR - Could not read web page '%s'"%www)
                    return

        self.Print("Reading from %s"%www)
        statfiles={}
        
        for i in out:
            if string.find(i,".txt")>-1:
                try:
                    temp=string.split(i,'"')
                    for z in temp:
                        if (string.strip(z)).endswith(".txt") and (string.upper(string.strip(z))).startswith("HCAL"):
                            temp=z
                            break
                    if (string.find(string.upper(temp),"HCAL")>-1):
                        run=string.split(temp,"_")[1]
                        if (run.startswith("run")):
                            run=run[3:]
                        run=string.split(run,".txt")[0]
                        run=string.atoi(run)
                        if (run>=self.startrun.get() and run<=self.endrun.get()):
                            print run
                            if run not in self.Runs:
                                self.Runs.append(run)
                            statfiles[temp]=run

                except:
                    self.Print("ERROR -- Could not parse web line '%s'"%i)

        for i in statfiles.keys():
            webname=self.webname.get()
            if (webname[-1]!="/"):
                webname=webname+"/"
            webname=webname+i
            run=statfiles[i]
            try:
                self.Print("Reading page '%s'"%webname)
                out=urlopen(webname).readlines()
            except:
                self.Print("ERROR -- Could not read '%s'"%webname)
                continue
            for line in out:
                try:
                    temp=string.split(line)
                    id=temp[5]
                    if (id=="value"):
                        continue # this is the header line
                    if id not in self.Cells.keys():
                        self.Cells[id]=CellStat(id)
                    self.Cells[id].read(run,line)
                except:
                    continue
                    #self.Print("ERROR -- Could not read line '%s'"%line)
        self.Print("Finished reading files")
        self.DrawCells()
        return

    def DrawCells(self):
        Runs=array('i')
        deadcells=array('i')
        hotcells=array('i')
        self.HotCellList=[]
        self.DeadCellList=[]
        self.AlwaysHotList=[]
        self.AlwaysDeadList=[]

        if (len(self.Runs)==0):
            self.Print("No runs found so far")
            return

        for i in self.Runs:
            Runs.append(i)
            deadcount=0
            hotcount=0
            for id in self.Cells.keys():
                if self.Cells[id].Dead(i):
                    #print "DEAD: ",id
                    deadcount=deadcount+1
                    if (id not in self.DeadCellList):
                        self.DeadCellList.append(id)
                if self.Cells[id].Hot(i):
                    hotcount=hotcount+1
                    if (id not in self.HotCellList):
                        self.HotCellList.append(id)
            deadcells.append(deadcount)
            hotcells.append(hotcount)
        for id in self.Cells.keys():
            if self.Cells[id].AlwaysHot():
                if id not in self.AlwaysHotList:
                    self.AlwaysHotList.append(id)
        for id in self.Cells.keys():
            if self.Cells[id].AlwaysDead():
                if id not in self.AlwaysDeadList:
                    self.AlwaysDeadList.append(id)
                         

        deadGraph=TGraph(len(Runs),Runs,deadcells)
        deadGraph.GetXaxis().SetTitle("Run #")
        deadGraph.SetMarkerColor(4)
        deadGraph.SetMarkerStyle(20)
        deadGraph.SetMinimum(0)
        hotGraph=TGraph(len(Runs),Runs,hotcells)
        hotGraph.GetXaxis().SetTitle("Run #")
        hotGraph.SetMarkerColor(2)
        hotGraph.SetMarkerStyle(20)
        hotGraph.SetMinimum(0)
        mg=TMultiGraph()
        mg.Add(deadGraph)
        mg.Add(hotGraph)
        c1 = TCanvas('c1','test',200, 10, self.canwidth, self.canheight )

        gStyle.SetOptTitle(1)
        mg.Draw("ap")
        mg.GetYaxis().SetTitle("# of cells")
        mg.GetXaxis().SetTitle("Run #")
        mg.SetTitle("# of Bad Cells/Run (Red = Dead, Blue = Hot)")
        mg.Draw("ap")
        c1.Update()
        c1.Print("badcelldisplay_file.gif")
        c1.Close()
        GUIimage=PhotoImage(file="badcelldisplay_file.gif")
        self.PicLabel.image=GUIimage
        self.PicLabel.configure(image=GUIimage)
        self.PicLabel.update()

        return

    def DrawCellStatus(self):
        ''' Draws status value of individual cell'''
        if (len(self.Runs)==0):
            self.GetInfo()
        Runs=array('i',self.Runs)
        newid=self.CellID.get()
        newid=string.strip(newid)
        print newid
        if newid not in self.Cells:
            self.Print("Checking for info on cell '%s'"%self.CellID.get())
            try:
                newid=calcDetID(self.CellID.get())
                #print newid
            except:
                #print newid
                self.Print("Could not find info for cell '%s'"%newid)
                return
        if newid not in self.Cells:
            #print newid
            self.Print("Could not find info for cell '%s'."%newid)
            return
        values=array('i')
        self.Runs.sort()
        if (len(self.Runs)>1):
            width=self.Runs[-1]-self.Runs[0]+1
            mymin=self.Runs[0]
            mymax=self.Runs[-1]+1
        else:
            width=1
            mymin=self.Runs[0]
            mymax=self.Runs[0]+1
        gStyle.SetOptStat(0)
        gStyle.SetPalette(1)
        gr=TH2F("gr","Single Cell Status",width,mymin,mymax,5,0,5)
        if (width==1):
            gr.GetXaxis().SetBinLabel(1,`self.Runs[0]`)
        gr.GetYaxis().SetBinLabel(1,"No Run")
        gr.GetYaxis().SetBinLabel(2,"Good")
        gr.GetYaxis().SetBinLabel(3,"Hot")
        gr.GetYaxis().SetBinLabel(4,"Dead")
        gr.GetYaxis().SetBinLabel(5,"Disabled")

        for i in range(mymin,mymax+1):
        #for i in self.Runs:
            if i in self.Cells[newid].status.keys():
                # Need to check bit assignments here
                if ((self.Cells[newid].status[i])&0x1):
                    gr.Fill(i,4)
                # hot cells               
                if ((self.Cells[newid].status[i]>>5)&0x1):
                    gr.Fill(i,2)
                # dead cells
                if ((self.Cells[newid].status[i]>>4)&0x1):
                    #print self.Cells[newid].status[i]
                    gr.Fill(i,3)
                if (self.Cells[newid].status[i]==0):
                    gr.Fill(i,1)
            else:
                gr.Fill(i,0)
        #gr=TGraph(len(Runs),Runs,values)
        #gr.SetTitle("Cell %s  (32 = dead, 16 = hot)"%newid)
        gr.GetXaxis().SetTitle("Run #")
        gr.SetMarkerColor(1)
        gr.SetMarkerStyle(20)
        gr.SetMinimum(0)
        c1 = TCanvas('c1','test',200, 10, self.canwidth, self.canheight )

        gr.Draw("col") # was "col" -- make configurable?
        c1.Update()

        c1.Print("badcelldisplay_file.gif")
        c1.Close()
        GUIimage=PhotoImage(file="badcelldisplay_file.gif")
        self.PicLabel.image=GUIimage
        self.PicLabel.configure(image=GUIimage)
        self.PicLabel.update()
        return
                
    def printCellList(self,text):
        print "\nPrinting cells:"
        text.sort()
        for i in text:
            print "%s\t\t%s"%(i, convertID(i))
        print
        return
    
    def Print(self,text):
        ''' Method to display text messages in comment frame.'''
        if (self.debug):
            print text
        self.commentLabel.configure(text=text)
        self.commentLabel.update()
        return

###############################################

if __name__=="__main__":
    if (os.path.isfile("badcelldisplay_file.gif")):
        os.system("rm -f badcelldisplay_file.gif")
    #print "%x"%calcDetID("HB (-16,2,1)")
    #print convertID("4200618D")
    x=RunStatusGui(debug=True)
    x.root.mainloop()
