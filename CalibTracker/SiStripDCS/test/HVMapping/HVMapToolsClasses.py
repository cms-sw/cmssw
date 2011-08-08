#! /usr/bin/env python
# Author:        Jake Herman
# Date:          20 June 2011
# Purpose: Contains two classes: 1) HVMapNoise has implementation to read in pedestal run noise from root file or pickle file to store in a (possibly) massive dictionary with keys for run type, sub detector, detID, and APVID. The dictionary can also be dumped to a pickle file whose name will be derived from the keys. 2) HVAnalysis implementation to Read in an HVMapNoise object or an HVMapNoise.Noise style dictionaries, perform assignment analysis methods and output pickles containing a dictionary of detID assignments and various .txt files. 


from ROOT import TFile, gDirectory, TCanvas, TH1F, TAttMarker, TAttLine

import pickle


class HVMapNoise:
    def __init__(self, name):
        self.Noise = {}
        self.name = name
        
        
    



    def ReadRootFile(self, RootFileDir, bool = False):
        """HVMapNoise.ReadRootFile(string RootFileDir[, bool bool])  Reads in the path of a .root file of pedestal run noise and stores it in self.noise. If bool is true then data taken in peak mode is multiplied by 1.7"""

        
        
        #make varibles to keep track of dictionary keys
        RunT = RootFileDir.split('/')[-1].split('-')[1]
        SubD = RootFileDir.split('/')[-1].split('-')[2]
        
    

        #intialize varible to record the number of entries read from RootFile
        Ndet = 0
        
        # Open the file:
        myfile = TFile(RootFileDir,'r')

        # retrieve the ntuple of interest
        mychain = gDirectory.Get('tree')
        entries = mychain.GetEntriesFast()

        #Loop over the trees entries
        for jentry in xrange(entries):

            #get the tree from the chain
            ientry = mychain.LoadTree(jentry)

            #copy the entry in to memory and verify
            nb = mychain.GetEntry(jentry)

            #use the values directly from the tree
            detID = str(mychain.DETID)
            APVID = int(mychain.APVID)
            NoiseMean = mychain.NOISEMEAN

            #compensate for effects of detector DAQ mode on noise means
            if bool:
                if 'PEAK' in RootFileDir:
                    NoiseMean = NoiseMean*1.7

                
            #Add to dictionary and check for detID/APVID uniqueness

            if RunT not in self.Noise.keys():
                self.Noise.update({RunT:{SubD:{detID:{APVID:NoiseMean}}}})
                Ndet = Ndet + 1
            elif SubD not in self.Noise[RunT].keys():
                self.Noise[RunT].update({SubD:{detID:{APVID:NoiseMean}}})
                Ndet = Ndet + 1
            elif detID not in self.Noise[RunT][SubD].keys():
                self.Noise[RunT][SubD].update({detID:{APVID:NoiseMean}})
                Ndet = Ndet + 1
            else:
                if APVID not in self.Noise[RunT][SubD][detID].keys():
                    self.Noise[RunT][SubD][detID].update({APVID : NoiseMean})

                else:
                    print "Warning: in year: ", Year,',' ," detID: ",RunT,SubD, detID, " has found two noise mean entries for APVID: ", APVID," could indicate file copy/naming error!"


        myfile.Close()
        print "Done reading in ", Ndet," detIDs from", RootFileDir.split('/')[-1]


        #End def ReadRootFile --------------------------------------



    def APVDetails(self, detID):
        """HVMapNoise.APVDetails(int detID) takes a run type and detID, prints a table  of APV noise details for that detID to the screen"""

        IdDict = {}

        for RunT in self.Noise.keys():
            for SubD in self.Noise[RunT].keys():
                if detID in self.Noise[RunT][SubD]:
                    for APVID in self.Noise[RunT][SubD][detID].keys():
                        try:
                            IdDict[APVID].update({RunT: self.Noise[RunT][SubD][detID][APVID]})
                        except:
                            IdDict.update({APVID:{RunT: self.Noise[RunT][SubD][detID][APVID]}})

        if len(IdDict.keys()) == 0:
            print "Warning call to HVMapNoise.APVDetails failed for:", detID
        else:
            sp = "    "
            print "--------------Details for:",detID,"----------------"
            print "APVID",sp,"HV1 ON",sp,"HV2 ON",sp,"BOTH ON",sp,"BOTH OFF"

            for APVID in IdDict:
                Line = str(APVID)+sp+sp

                try:
                    Line = Line + str(round(IdDict[APVID]['HV1'],6)) + sp
                except:
                    Line = Line + "No Info"+sp

                try:
                    Line = Line + str(round(IdDict[APVID]['HV2'],6)) + sp
                except:
                    Line = Line + "No Info"+sp

                try:
                    Line = Line + str(round(IdDict[APVID]['ON'],6)) + sp
                except:
                    Line = Line + "No Info"+sp

                try:
                    Line = Line + str(round(IdDict[APVID]['OFF'],6)) + sp
                except:
                    Line = Line + "No Info"+sp

                print Line
                
                   
                
                
                    
    def DetsInRun(self,RunT):
        """HVMapNoise.DetsInRun(string RunT) Returns the number of detIDs in a run"""

        Dets = 0

        if RunT not in self.Noise.keys():
            print "Warning: invalid call to HVMapNoise::DetsInRun"
            return Dets
        else:
            for SubD in self.Noise[RunT].keys():
                Dets = Dets + len(self.Noise[RunT][SubD].keys())
            return Dets
             
        




    def PickleNoise(self, directory):
        """HVMapNoise.PickleNoise(string directory) writes dictionary HVMapNoise.Noise to a picklefile in directory, directory should contain file name with .pkl extension"""
        

        file = open(directory, 'wb')
        pickle.dump(self.Noise, file)
        file.close
        print "Noise pickled in ", directory

        # End def PickleNoise--------------------------------------------


    def UnpickleNoise(self, directory):
        """HVMapNoise.UnpickleNoise(string directory) replaces  self.Noise with whatever dictionary is in directory. Directory is intended to point to a file created from HVMapNoise::PickleNoise"""

        self.Noise = clone(pickle.load(open(directory,'rb')))


    #End def UnpickleNoise ---------------------------------------------------


    def AddFromPickle(self, directory):
        """ HVMapNoise.AddFromPickle(string directory) reads in a dictionary of APV noise means and Updates them to whatever is already in self.Noise. Will overwrite if self.Noise[RunT][SubD][detID][APVID] exists."""

        try:
            #open file
            file = open(directory, 'rb')
            print directory, "Opened"
            #Unpack the noise dictionary
            PickleNoise =  pickle.load(file)

            #Update as opposed to overwrite self.Noise with PickleNoise, lots of parsing but just a strait forward check for existance overwriting if exists and making news keys/ entries if not
        
            
            for RunT in PickleNoise.keys():#Parse RunType
                for SubD in PickleNoise[RunT].keys():#Parse Sub-detector
                    for detID in PickleNoise[RunT][SubD].keys():#Parse detID
                        for APVID in PickleNoise[RunT][SubD][detID].keys():#Parse APVID
                            if RunT not in self.Noise.keys():
                                self.Noise.update({RunT: { SubD: { detID: { APVID : PickleNoise[RunT][SubD][detID][APVID]}}}})
                            elif SubD not in self.Noise[RunT].keys():
                                self.Noise[RunT].update({SubD : { detID : { APVID : PickleNoise[RunT][SubD][detID][APVID]}}})
                            elif detID not in self.Noise[RunT][SubD].keys():
                                self.Noise[RunT][SubD].update({detID : {APVID : PickleNoise[RunT][SubD][detID][APVID] }})
                            elif APVID not in self.Noise[RunT][SubD][detID].keys():
                                self.Noise[RunT][SubD][detID].update({APVID:PickleNoise[RunT][SubD][detID][APVID]})
                            else:
                                self.Noise[RunT][SubD][detID].update({APVID:PickleNoise[RunT][SubD][detID][APVID]})
                                print 'Warning overwritting:',RunT, SubD, detID, APVID, "in", self.name


            file.close()
        
            print  directory.split('/')[-1], "Added"

        except:
            print directory, "Could not be added"

        #End def AddFromPickleNoise ----------------------------------------------------
                                        
                                        
    def IntializeHisto(self,min, max,bins,name = None):
        """HVMapNoise.IntializeHisto(float min, float max, int bins[,string name]) sets name to the name and title of self.Histo which defaults to self.name. Sets the min, max, and bin number of the TH1F object it returns"""

        if name is None:

            if '.root' in self.name:
                name = self.name.replace('.root','')

            elif '.pkl' in self.name:
                name = self.name.replace('.pkl','')

            else:
                name = self.name
                
        histo = TH1F(name, name, bins, min, max)

        return histo


        #End def IntializeHisto --------------------------------------------------



    def AddToHisto(self,histo,RunT,SubD,ColorKey = None, Assignment= None, Reference = None):
        """HVMapNoise.AddToHisto(string RunT, string SubD[,int ColorKey[, string Assignment[,dict Reference]]]) Adds Noise information from self.Noise[RunT][SubD] to histo.If 'All' is passed as SubD all subdector noise info available will be added to histo. Optionally color can be added to the histogram to differentiate between histos if a multiplot is desired. Also includes optional implementation to plot only noise whose detID was labeled Assignment in dictionary of assignments Reference. Returns histo"""

        if ColorKey is not None:
            try:
                histo.SetLineColor(ColorKey)
            except:
                print "Warning: in invalid color key passed to",self.name,".AddToHisto"

        if Assignment is None:

            if SubD == 'All':
                for SDet in self.Noise[RunT].keys():
                    for detID in self.Noise[RunT][SDet].keys():
                        for APVID in self.Noise[RunT][SDet][detID].keys():
                            histo.Fill(self.Noise[RunT][SDet][detID][APVID])
                return histo

            else:

                for detID in self.Noise[RunT][SubD].keys():
                    for APVID in self.Noise[RunT][SubD][detID].keys():
                        histo.Fill(self.Noise[RunT][SubD][detID][APVID])
                return histo
        
        elif Reference is not None:

            if SubD == 'All':
                for SDet in self.Noise[RunT].keys():
                    for detID in self.Noise[RunT][SDet].keys():
                        if Reference[detID] == Assignment:
                            for APVID in self.Noise[RunT][SDet][detID].keys():
                                histo.Fill(self.Noise[RunT][SDet][detID][APVID])
                return histo

            else:

                for detID in self.Noise[RunT][SubD].keys():
                    for APVID in self.Noise[RunT][SubD][detID].keys():
                        if Reference[detID] == Assignment:
                            histo.Fill(self.Noise[RunT][SubD][detID][APVID])
                return histo
        else:
            print "Warning: Assignment passed to",self.name,".AddToHisto but not reference dictionary, skipping..."
            
                
        #End def AddToHisto -------------------------------------------------
    
        

            
#End def HVMapNoise class -----------------------------------------------------
#------------------------------------------------------------------------------
        
class HVAnalysis:
    def __init__(self, name, HVMNoise,Instructions, patch = True):
        """__init__(string name, dict or HVMapNoise HVMNoise, Instructions) intialization reads in a HVMapNoise object and makes dictionaries with keys for detID and APVID of HV1 - HV2, HVON/HV1, HVON/HV2, HV1/HVOFF, and HV2/HVOFF of the noise for each APV if data is availble. Alternatively HVMNoise can be an HVMapNoise.Noise style dictionary. Instructions is a list or a single string which indicates which noise analyses are to be done to intialize this object. 'DIFF' -> HV1 - HV2, 'OF'-> OFF - ON, 'RON' -> HV ON Ratios, 'ROFF' -> HV OFF Ratios"""
        self.name = name
        self.Diff = {}
        self.OnOffDiff = {}
        self.RatioOn1 = {}
        self.RatioOn2 = {}
        self.RatioOff1 = {}
        self.RatioOff2 = {}
        

        #Patch: I forgot some ' :)
        ON = 'ON'
        OFF = 'OFF'
        HV1 = 'HV1'
        HV2 = 'HV2'
        
        try:
            Noise = HVMNoise.Noise
        except:
            Noise = HVMNoise
        
        if 'HV1' in Noise.keys() and 'HV2' in Noise.keys():

            if 'DIFF' in Instructions or 'ALL' in Instructions:
                print "Intializing HV1 - HV2"

                try:
                    print HVMNoise.DetsInRun(HV1),"DetIds in HV1 run,", HVMNoise.DetsInRun(HV2),"DetIDs in HV2 run."

                except:
                    pass

                    
                #Make Noise difference HV1 - HV2 for diff analysis
                for SubD in Noise[HV1].keys():
                    if SubD in Noise[HV2].keys():           
                        for detID in Noise[HV1][SubD].keys():
                            if detID in Noise[HV2][SubD].keys():
                                for APVID in Noise[HV1][SubD][detID].keys():
                                    if APVID in Noise[HV2][SubD][detID].keys():
                                        if detID not in self.Diff.keys():
                                            self.Diff.update({detID:{APVID:Noise[HV1][SubD][detID][APVID] -Noise[HV2][SubD][detID][APVID]}})
                                        elif APVID not in self.Diff[detID].keys():
                                            self.Diff[detID].update({APVID:Noise[HV1][SubD][detID][APVID] -Noise[HV2][SubD][detID][APVID]})
                                        else:
                                            print"Warning duplicated detIDs detected in HV1 and HV2 runs for", self.name
                                    else:
                                        print APVID, "in",SubD,detID," found in HV1 Run but not in HV2 run"
                            else:
                                print detID,"in ", SubD," found in HV1 run but not HV2 run"
            
                    else:
                        print SubD,"found in HV1 run but not HV2 run skipping..."
            
                print len(self.Diff.keys()),"detIDs in HV1 - HV2"



            if 'OFF' in Noise.keys() and 'ON' in Noise.keys() and ('OF' in Instructions or 'ALL' in Instructions):

                
                print "Intializing OFF - ON"

                try:
                    print HVMNoise.DetsInRun(ON),"DetIds in HV ON run,", HVMNoise.DetsInRun(OFF),"DetIDs in HV OFF run."

                except:
                    pass
                
                for SubD in Noise[OFF].keys():
                    if SubD in Noise[ON].keys():           
                        for detID in Noise[OFF][SubD].keys():
                            if detID in Noise[ON][SubD].keys():
                                for APVID in Noise[OFF][SubD][detID].keys():
                                    if APVID in Noise[ON][SubD][detID].keys():
                                        if detID not in self.OnOffDiff.keys():
                                            self.OnOffDiff.update({detID:{APVID:Noise[OFF][SubD][detID][APVID] -Noise[ON][SubD][detID][APVID]}})
                                        elif APVID not in self.OnOffDiff[detID].keys():
                                            self.OnOffDiff[detID].update({APVID:Noise[OFF][SubD][detID][APVID] -Noise[ON][SubD][detID][APVID]})
                                        else:
                                            print"Warning duplicated detIDs detected in ON and OFF runs for", self.name
                                    else:
                                        print APVID, "in",SubD,detID," found in OFF Run but not in ON run"
                            else:
                                print detID,"in ", SubD," found in OFF run but not ON run"
            
                    else:
                        print SubD,"found in OFF run but not ON run skipping..."
            
                print len(self.OnOffDiff.keys()),"detIDs in OFF - ON"

                
            if 'ON' in Noise.keys() and ('RON' in Instructions or 'ALL' in Instructions):
    
                print "Intializing HVON/HV1"

                try:
                    print HVMNoise.DetsInRun(ON),"DetIds in HVON run,", HVMNoise.DetsInRun(HV1),"DetIDs in HV1 run"

                except:
                    pass
                
                for SubD in Noise[ON].keys():
                    if SubD in Noise[HV1].keys():           
                        for detID in Noise[ON][SubD].keys():
                            if detID in Noise[HV1][SubD].keys():
                                for APVID in Noise[ON][SubD][detID].keys():
                                    if APVID in Noise[HV1][SubD][detID].keys():
                                        if Noise[HV1][SubD][detID][APVID] == 0:
                                            if detID not in self.RatioOn1.keys():
                                                self.RatioOn1.update({detID:{APVID:0}})
                                            elif APVID not in self.RatioOn1[detID].keys():
                                                self.RatioOn1[detID].update({APVID:0})
                                            else:
                                                print "Warning duplicate detIDs detected in HV1 and HVON runs for" ,self.name                          
                                        elif detID not in self.RatioOn1.keys():
                                            self.RatioOn1.update({detID:{APVID:Noise[ON][SubD][detID][APVID]/Noise[HV1][SubD][detID][APVID]}})
                                        elif APVID not in self.RatioOn1[detID].keys() and Noise[HV1][SubD][detID][APVID] !=0:
                                            self.RatioOn1[detID].update({APVID:Noise[ON][SubD][detID][APVID]/Noise[HV1][SubD][detID][APVID]})
                                        else:
                                            print "Warning duplicate detIDs detected in HV1 and HVON runs for", self.name
                                    else:
                                        print APVID, "in",SubD,detID," found in HVON Run but not in HV1 run"
                            else:
                                print detID,"in ", SubD," found in HVON run but not HV1 run"
                                
                    else:
                        print SubD,"found in HVON run but not  HV1 run skipping..."
                print len(self.RatioOn1.keys()), "detIDs in ON/HV1"

                #Make Noise ratio for HVON/HV2
                print "Intializing HVON/HV2"

                try:
                    print HVMNoise.DetsInRun(ON),"DetIds in HVON run,", HVMNoise.DetsInRun(HV2),"DetIDs in HV2 run"

                except:
                    pass
                
                for SubD in Noise[ON].keys():
                    if SubD in Noise[HV2].keys():           
                        for detID in Noise[ON][SubD].keys():
                            if detID in Noise[HV2][SubD].keys():
                                for APVID in Noise[ON][SubD][detID].keys():
                                    if APVID in Noise[HV2][SubD][detID].keys():
                                        if Noise[HV2][SubD][detID][APVID] == 0:
                                            if detID not in self.RatioOn2.keys():
                                                self.RatioOn2.update({detID:{APVID:0}})
                                            elif APVID not in self.RatioOn2[detID].keys():
                                                self.RatioOn2[detID].update({APVID:0})
                                            else:
                                                print "Warning duplicate detIDs detected in HV2 and HVON runs for" ,self.name
                                        elif detID not in self.RatioOn2.keys():
                                            self.RatioOn2.update({detID:{APVID:Noise[ON][SubD][detID][APVID]/Noise[HV2][SubD][detID][APVID]}})
                                        elif APVID not in self.RatioOn2[detID].keys():
                                            self.RatioOn2[detID].update({APVID:Noise[ON][SubD][detID][APVID]/Noise[HV2][SubD][detID][APVID]})
                                        else:
                                            print "Warning duplicate detIDs detected in HV2 and HVON runs for", self.name
                                    else:
                                        print APVID, "in",SubD,detID," found in HVON Run but not in HV2 run"
                            else:
                                print detID,"in ", SubD," found in HVON run but not HV2 run"
            
                    else:
                        print SubD,"found in HVON run but not  HV2 run skipping..."

                print len(self.RatioOn2.keys()), "in ON/HV2"

                
            if 'OFF' in Noise.keys() and ('ROFF' in Instructions or 'ALL' in Instructions):
                print "Intializing HV1/HVOFF"

                try:
                    print HVMNoise.DetsInRun(OFF),"DetIds in OFF run,", HVMNoise.DetsInRun(HV1),"DetIDs in HV1 run"

                except:
                    pass
                
                for SubD in Noise[HV1].keys():
                    if SubD in Noise[OFF].keys():           
                        for detID in Noise[HV1][SubD].keys():
                            if detID in Noise[OFF][SubD].keys():
                                for APVID in Noise[HV1][SubD][detID].keys():
                                    if APVID in Noise[OFF][SubD][detID].keys():
                                        if Noise[OFF][SubD][detID][APVID] == 0:
                                            if detID not in self.RatioOff1.keys():
                                                self.RatioOff1.update({detID:{APVID:0}})
                                            elif APVID not in self.RatioOff1[detID].keys():
                                                self.RatioOff1[detID].update({APVID:0})
                                            else:
                                                print "Warning duplicate detIDs detected in HV1 and HVOFF runs for" ,self.name
                                        elif detID not in self.RatioOff1.keys():
                                            self.RatioOff1.update({detID:{APVID:Noise[HV1][SubD][detID][APVID]/Noise[OFF][SubD][detID][APVID]}})
                                        elif APVID not in self.RatioOff1[detID].keys():
                                            self.RatioOff1[detID].update({APVID:Noise[HV1][SubD][detID][APVID]/Noise[OFF][SubD][detID][APVID]})
                                        else:
                                            print "Warning duplicated detIDs detected in HV1 and HVOFF runs for", self.name
                                    else:
                                        print APVID, "in",SubD,detID," found in HV1 Run but not in HVOFF run"
                            else:
                                print detID,"in ", SubD," found in HV1 run but not HVOFF run"
            
                    else:
                        print SubD,"found in HV1 run but not  HVOFF run skipping..."

                print len(self.RatioOff1.keys()),"detIDs in HV1/OFF"


                #Make Noise ratio for HV2/HVOFF
                print "Intializing HV2/HVOFF"

                try:
                    print HVMNoise.DetsInRun(OFF),"DetIds in OFF run,", HVMNoise.DetsInRun(HV2),"DetIDs in HV2 run"
                    
                except:
                    pass
                 
                for SubD in Noise[HV2].keys():
                    if SubD in Noise[OFF].keys():           
                        for detID in Noise[HV2][SubD].keys():
                            if detID in Noise[OFF][SubD].keys():
                                for APVID in Noise[HV2][SubD][detID].keys():
                                    if APVID in Noise[OFF][SubD][detID].keys():
                                        if Noise[OFF][SubD][detID][APVID] == 0:
                                            if detID not in self.RatioOff2.keys():
                                                self.RatioOff2.update({detID:{APVID:0}})
                                            elif APVID not in self.RatioOff2[detID].keys():
                                                self.RatioOff2[detID].update({APVID:0})
                                            else:
                                                print "Warning duplicate detIDs detected in HV2 and HVOFF runs for" ,self.name
                                        elif detID not in self.RatioOff2.keys():
                                            self.RatioOff2.update({detID:{APVID:Noise[HV2][SubD][detID][APVID]/Noise[OFF][SubD][detID][APVID]}})
                                        elif APVID not in self.RatioOff2[detID].keys():
                                            self.RatioOff2[detID].update({APVID:Noise[HV2][SubD][detID][APVID]/Noise[OFF][SubD][detID][APVID]}) 
                                        else:
                                            print "Warning duplicated detIDs detected in HV2 and HVOFF runs for", self.name
                                    else:
                                        print APVID, "in",SubD,detID," found in HV2 Run but not in HVOFF run"
                            else:
                                print detID,"in ", SubD," found in HV2 run but not HVOFF run"
            
                    else:
                        print SubD,"found in HV2 run but not  HVOFF run skipping..."

                print len(self.RatioOff2.keys()),"detIDs in HV2/OFF\n"

        elif 'args' not in dir() and patch:
            print "No data passed to analysize HVAnalysis"

        elif patch:
            print "Insufficient noise data passed to", self.name,"to do analysis"


    

    #End Intializer------------------------------------


    def DiffMethod(self,cut = 0):
        """HVAnalysis.DiffMethod(float cut) makes HV assignments based on the noise difference sign of APVs in  self.Diff{} and the cut. Returns a dictionary of assignments"""

        assigned={}
        
        #loop through all entries
        for detid in self.Diff.keys():
            flag1=flag2=count=sanity=0
            for APV in self.Diff[detid].keys():
                count=count+1
                if abs(self.Diff[detid][APV])>=cut:
                    if self.Diff[detid][APV]>0:
                        flag2=flag2+1
                    if self.Diff[detid][APV]<0:
                        flag1=flag1+1
                
                    

            #make categorical assignment
            check=(flag1+flag2)
            if flag1 != count and flag2 != count:
                if check>(count/2):
                    if flag1>flag2:
                        if flag2 == 0:
                            assigned.update({detid:'HV1 with some unresponsive APVs'})
                        else:
                            assigned.update({detid:'HV1 with opposite pointing APVs'})

                    if flag2>flag1:
                        if flag1 == 0:
                            assigned.update({detid:'HV2 with some unresponsive APVs'})
                        else:
                            assigned.update({detid:'HV2 with opposite pointing APVs'})

                    if flag2==flag1:
                        assigned.update({detid:'Equal Indicators'})
                    
                
                else:
                    assigned.update({detid:'Unresponsive'})
            elif flag2 == 0:
                assigned.update({detid:'HV1'})
            elif flag1 == 0:
                assigned.update({detid:'HV2'})
        return assigned

    
    #End def DiffMethod---------------------------------------------

    def StrictDiffMethod(self, cut = 0):
        """HVAnalysis.DiffMethod(float cut) makes HV assignments based on the noise difference sign of APVs in  self.Diff{} and the cut. Returns a dictionary of assignments"""

        assigned={}
        
        #loop through all entries
        for detid in self.Diff.keys():
            flag1=flag2=count=sanity=0
            for APV in self.Diff[detid].keys():
                count=count+1
                if abs(self.Diff[detid][APV])>cut:
                    if self.Diff[detid][APV]>0:
                        flag2=flag2+1
                    if self.Diff[detid][APV]<0:
                        flag1=flag1+1
                
                    

            #make categorical assignment
            if flag1 == count:
                assigned.update({detid:'HV1'})
            elif flag2 == count:
                assigned.update({detid:'HV2'})    
            else:
                assigned.update({detid:'Undetermined'})
            

                                    
        #import list of all tracker detIDs so unmapped detIDs can be filled in
        #FIXME:
        #Link to the package ../../data/StripDetIDAlias.pkl
        #Absolute path is CalibTracker/SiStripDCS/data/StripDetIDAlias.pkl
        StripDetIDAliasFile=open("data/StripDetIDAlias.pkl","rb")
        StripDetIDAliasDict=pickle.load(StripDetIDAliasFile)
        AllDetIDs=[str(detid) for detid in StripDetIDAliasDict.keys()]

        for detid in AllDetIDs:
            if detid not in assigned.keys():
                assigned.update({detid:'Masked'})

        return assigned

    
    #End def StrictDiffMethod---------------------------------------------

    def OnOffDiffMethod(self, cut):
        """Takes a dictionary of of Off - On noise and returns a dictionary of assignments as either 'Cross-Talking' or 'No-HV'"""

        assigned = {}
   
        for detID in self.OnOffDiff.keys():
            flagCT = flagNHV = count = 0
            for APVID in self.OnOffDiff[detID].keys():
                count = count + 1
                if self.OnOffDiff[detID][APVID] >= cut:
                    flagCT = flagCT + 1
                else:
                    flagNHV = flagNHV + 1

            if flagCT == count:
                assigned.update({detID : 'Cross-Talking'})
            else:
                assigned.update({detID : 'No-HV'})

        return assigned


    def RatioMethod(self, opt, CutHV1, CutHV2):
        """HVAnalysis.RatioMethod(string opt, float CutHV1, float CutHV1) has two options 'ON' or 'OFF' and makes HV assigments based on the noise contents of self.Ratio_ . Returns a dictionary of assignments"""

        assigned={}
        
        if opt == 'ON':
            RatioDict1 = self.RatioOn1
            RatioDict2 = self.RatioOn2
        elif opt == 'OFF':
            RatioDict1 = self.RatioOff2
            RatioDict2 = self.RatioOff1
        else:
            print "Invalid opt argument passed to Analysis::RatioMethod for object:", self.name
        
        hv1_51=hv1_42=hv1_31=hv2_51=hv2_42=hv2_31=0
        #loop through all entries
        for detid in RatioDict1.keys():
            count1=count2=sanity=flag1=flag2=0
            for APV in RatioDict1[detid].keys():
                count1=count1+1
                #assigns flags for each APV
                if RatioDict1[detid][APV]>CutHV1:
                    if APV in RatioDict2[detid].keys():
                        if RatioDict2[detid][APV]<CutHV2:
                            flag1=flag1+1
                    else:
                        print "APV misssing!", detid, "APV:",APV

                if RatioDict1[detid][APV]<CutHV1:
                    if APV in RatioDict2[detid].keys():
                        if RatioDict2[detid][APV]>CutHV2:
                            flag2=flag2+1
                    else:
                        print "APV misssing!", detid, "APV:",APV

            #sum flags from both files
            flag=flag1+flag2
                
            #make categorical assignment
            if flag>=count1/2:
                if flag1==0:
                    assigned.update({detid:'HV2'})
                    sanity=sanity+1
                if flag2==0:
                    assigned.update({detid:'HV1'})
                    sanity=sanity+1
                if flag1>flag2 and flag2!=0:
                    assigned.update({detid:'Undetermined'})
                    sanity=sanity+1
                    #print "detid: ",detid," flag1: ",flag1," flag2: ",flag2
                    if flag1==5 and flag2==1:
                        hv1_51=hv1_51+1
                    if flag1==4 and flag2==2:
                        hv1_42=hv1_42+1
                    if flag1==3 and flag2==1:
                        hv1_31=hv1_31+1
                if flag2>flag1 and flag1!=0:
                    assigned.update({detid:'Undetermined'})
                    sanity=sanity+1
                    #print "detid: ",detid," flag1: ",flag1," flag2: ",flag2
                    if flag2==5 and flag1==1:
                        hv2_51=hv2_51+1
                    if flag2==4 and flag1==2:
                        hv2_42=hv2_42+1
                    if flag2==3 and flag1==1:
                        hv2_31=hv2_31+1
            else:
                assigned.update({detid:'Undetermined'})
                sanity=sanity+1
                
 
            #make sure no modules double assigned
            if sanity==0:
                print detid,"no analysis"
            if sanity>1 and flag !=0:
                print detid,"double analyzed"
            

        #import list of all tracker detIDs so unmapped detIDs can be filled in
        fullmap=open('data/FullMap.txt','r')
        full=[]
        for line in fullmap:
            try:
                detid=line.split()[0]
            except:
                pass
            if detid not in full:
                full.append(str(detid))
        fullmap.close()

        for detid in full:
            if detid not in assigned.keys():
                assigned.update({detid:'Masked'})

        return assigned
    
     #End def RatioMethod ---------------------------------

     
    def PrintResults(self,assigned,type):#Should add implementation to use and option ('Diff' ect.) and self
        """HVAnalysis.PrintResults(dict assigned) recieves a dictionary with detid's and assignments and prints them to screen. if detid not in assigned dictionary, method assigns unmapped tag"""

        
        hv1=hv2=unHV1=unHV2=unEq=disc=unmap=undet=0
         
        for detid in assigned.keys():
            if assigned[detid]=='HV1' or assigned[detid]=='channel002':
                hv1=hv1+1
            if assigned[detid]=='HV2' or assigned[detid]=='channel003':
                hv2=hv2+1
            if assigned[detid]=='Masked':
                unmap=unmap+1
            if assigned[detid] == 'Undetermined':
                undet = undet + 1

        print "\n\n--------",type,"----------"
        print "HV1 modules:\t\t",hv1
        print "HV2 modules:\t\t",hv2
        print "Masked:\t",unmap
        print "Undetermined modules:\t",undet
        print "TOTAL detIDs:\t\t",len(assigned.keys()),'\n'

     #End ftn ------------------------------------------

    def Mktxt(self,assign1,savepath, Cate=None):
        """HVAnalysis.Mktxt(dict assign1, string savepath [, string Cate]) method writes out a final .txt file formatted so that can be used with hvtkmapcreator to create pictoral maps. Can either map a specific assignment or a dict of 'HV1', 'HV2', 'Masked', 'Undetermined' assignments"""
        
        
    
        results=open(savepath+ '.txt','w')

        if Cate is None:
            for detid in assign1.keys():
                if assign1[detid]=='HV1':
                    results.write('\n%s\tchannel002'%detid)
            
                if assign1[detid]=='HV2':
                    results.write('\n%s\tchannel003'%detid)
                

                if assign1[detid]=='Masked':
                    results.write('\n%s\tchannel000'%detid)
                
                
                if assign1[detid]=='Undetermined':
                    results.write('\n%s\tchannel001'%detid)
        else:
            for detid in assign1.keys():
                if assign1[detid] == Cate:
                    results.write('\n%s\tchannel002'%detid)
            
        results.close()
            
    # ------------ def Mktxt  ---------------------

    def MkList(self,Assignments, savepath):
        """ HVAnalysis.MkList(dict Assignments, string savepath) makes a .txt file of a list of detIDs Aliases and Assignments for a given Assignments dictionary"""
        
        file = open('data/StripDetIDAlias.pkl','rb')
        AliasDict = pickle.load(file)
        file.close()

        file = open(savepath +'.txt','wb')


        for detID in Assignments.keys():

            file.write('|%s|\t%s|\t%s|\n'%(str(detID),str(AliasDict[int(detID)]).replace('set([','').replace('])',''),Assignments[detID]))
      
        file.close()
    
    #-------------End MkList---------------------------------------

    def MkSList(self,Assigndict, Assignment, savepath, mode):
        """ HVAnalysis.MkSList(dict Assigndict, string Assignment, string savepath,  string mode) Makes a .txt list of modules of Assignment in Assigndict either sorted by alias or detID depending or where mode is 'A' or 'D' saves at savepath"""

        file = open('data/StripDetIDAlias.pkl','rb')
        AliasDict = pickle.load(file)
        file.close()

        file = open(savepath + '.txt','wb')

        for detID in Assigndict.keys():
            Assigndict.update({int(detID): Assigndict[detID]})
        
        if mode == 'A':
            Assigndict.keys().sort()
            for detID in Assigndict.keys():
                if Assigndict[detID] == Assignment: 
                    file.write('|%s\t|%s|\n'%(str(AliasDict[int(detID)]).replace('set([','').replace('])','') ,str(detID)))

        elif mode == 'D':
            Assigndict.keys().sort()
            for detID in Assigndict.keys():
                if Assigndict[detID] == Assignment:
                    file.write('|%s\t|%s|\n'%(str(detID), str(AliasDict[int(detID)]).replace('set([','').replace('])','')))

        else:
            print "invalid mode passed to HVAnalysis.MkSlist"

        file.close()
        
    def MkHisto(self,opt,min,max,bins,name = None):
        """HVAnalysis.Plot(string opt, float min, float max, int bins[,string name]) intializes intializes a TH1F object with domain limits 'min', 'max' and number of 'bins' and name and title name which defaults to self.name. Fills histo with analysis dictionary based on 'opt', 'DIFF'->self.Diff, 'ON1' -> self.RatioOn1, 'OFF1' -> self.RatioOff1, ect. Returns the TH1F object"""

        
        if name is None:
            name = self.name
            
        histo = TH1F(name, name, bins, min, max)

        

        if opt == 'DIFF':
            for detID in self.Diff.keys():
                for APVID in self.Diff[detID].keys():
                    histo.Fill(self.Diff[detID][APVID])
            return histo

        elif opt == 'ON1':
            for detID in self.RatioOn1.keys():
                for APVID in self.RatioOn1[detID].keys():
                    histo.Fill(self.RatioOn1[detID][APVID])
            return histo

        elif opt == 'ON2':
            for detID in self.RatioOn2.keys():
                for APVID in self.RatioOn2[detID].keys():
                    histo.Fill(self.RatioOn2[detID][APVID])
            return histo

        elif opt == 'OFF1':
            for detID in self.RatioOff1.keys():
                for APVID in self.RatioOff1[detID].keys():
                    histo.Fill(self.RatioOff1[detID][APVID])
            return histo

        elif opt == 'OFF2':
            for detID in self.RatioOff2.keys():
                for APVID in self.RatioOff2[detID].keys():
                    histo.Fill(self.RatioOff2[detID][APVID])
                    
        elif opt == 'ONOFF':
            for detID in self.OnOffDiff.keys():
                for APVID in self.OnOffDiff[detID].keys():
                    histo.Fill(self.OnOffDiff[detID][APVID])
        else:
            print "Warning call to HVAnalysis.MkHisto failed to fill histogram"

        return histo
        


#End def Plot ------------------------------------------------------------------
