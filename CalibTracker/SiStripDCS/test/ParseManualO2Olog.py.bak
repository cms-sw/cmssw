#! /usr/bin/env python
#Script to parse the output of ManualO2O.py for various debugging tasks
#First use case is to debug issue with HV1/HV2 channels handling

import os,datetime, pickle
def GetLogTimestamps(ManualO2Ologfilename):
    """
    This function opens the ManualO2O log chosen, and it prints out the Tmin/Tmax intervals
    for the individual O2O jobs that were run.
    This information can in turn be used with the function GetLogInfoByTimeStamp,
    to extract from the log the (DEBUG) information only from the wanted time interval.
    """
    ManualO2Olog=open(ManualO2Ologfilename,'r')
    LinesCounter=0
    IOVCounter=1
    TmaxDict={}#This dictionary will have Tmax as keys and the lines corresponding to that Tmax as value
    for line in ManualO2Olog:
        LinesCounter+=1
        if 'Tmin:' in line:
            Tmin=line.split(':')[1].strip()
        if 'Tmax:' in line:
            Tmax=line.split(':')[1].strip()
            #Assume that when Tmax is read Tmin was read too since we are parsing lines like this:
            #     Tmin: 2010 8 27 10 0 0 0
            #     Tmax: 2010 8 29 9 0 0 0
            print "%s- Tmin: %s and Tmax: %s (number log lines %s)"%(IOVCounter,Tmin,Tmax,LinesCounter)
            #Actually Tmin is not necessary... since they all will be set to the first one in the dcs_o2O_template.py used by the ManualO2O.py that created the log we're parsing.
            if Tmax not in ListOfTmax.keys():
                TmaxDict.update({Tmax:LinesCounter})
            else:
                print "This Tmax (%s) seems to be duplicated!!!"%Tmax
            LinesCounter=0
            IOVCounter+=1
    ManualO2Olog.close()
    return TmaxDict
            
            
def GetLogInfoByTimeStamp(ManualO2Ologfilename,StartTime,StopTime):
    """
    This function takes a ManualO2Ologfilename, a start and a stop timestamps (that should be
    in the "YYYY M D H M S"  format, e.g. Tmin: "2010 8 27 10 0 0 0",Tmax: "2010 8 27 11 0 0 0",
    and it parses the given logfile to just print out the lines relevant to the given timestamp. 
    """
    #Open the log:
    ManualO2Olog=open(ManualO2Ologfilename,'r')
    #Loop to extract only the wanted lines
    PrintLines=False
    WantedLines=[]
    for line in ManualO2Olog:
        if 'Tmax:' in line and StopTime in line:
            PrintLines=True
        if PrintLines:
            if 'Tmin:' in line:
                break
            else:
                print line.strip()
                WantedLines.append(line)
    ManualO2Olog.close()
    return WantedLines
#TO BE DONE!
#implement tzinfo subclasses to get GMT1 behaviour to translate always the timestamp right...
#from datetime import timedelta, datetime, tzinfo
#   import datetime
#   class GMT1(datetime.tzinfo):
#       def __init__(self,dt):         # DST starts last Sunday in March
#           d = datetime.datetime(dt.year, 4, 1)   # ends last Sunday in October
#           self.dston = d - datetime.timedelta(days=d.weekday() + 1)
#           d = datetime.datetime(dt.year, 11, 1)
#           self.dstoff = d - datetime.timedelta(days=d.weekday() + 1)
#       def utcoffset(self, dt):
#           return datetime.timedelta(hours=1) + self.dst(dt)
#       def dst(self, dt):
#           if self.dston <=  dt.replace(tzinfo=None) < self.dstoff:
#               return datetime.timedelta(hours=1)
#           else:
#               return datetime.timedelta(0)
#       def tzname(self,dt):
#           return "GMT +1"
#   
#   import time as _time
#   
#   STDOFFSET = datetime.timedelta(seconds = -_time.timezone)
#   if _time.daylight:
#       DSTOFFSET = datetime.timedelta(seconds = -_time.altzone)
#   else:
#       DSTOFFSET = STDOFFSET
#   
#   DSTDIFF = DSTOFFSET - STDOFFSET
#   
#   class LocalTimezone(datetime.tzinfo):
#   
#       def utcoffset(self, dt):
#           if self._isdst(dt):
#               return DSTOFFSET
#           else:
#               return STDOFFSET
#   
#       def dst(self, dt):
#           if self._isdst(dt):
#               return DSTDIFF
#           else:
#               return ZERO
#   
#       def tzname(self, dt):
#           return _time.tzname[self._isdst(dt)]
#   
#       def _isdst(self, dt):
#           tt = (dt.year, dt.month, dt.day,
#                 dt.hour, dt.minute, dt.second,
#                 dt.weekday(), 0, -1)
#           stamp = _time.mktime(tt)
#           tt = _time.localtime(stamp)
#           return tt.tm_isdst > 0
#   
#   Local = LocalTimezone()
#   
def GetQueryResults(ManualO2Ologfilename):
    """
    This function takes a ManualO2Ologfilename,
    and if it was run with the debug option True,
    it extracts all the query results.
    It uses StartTime/StopTime to identify the interval of the query (i.e. the ManualO2O.py running)
    and uses the tuple (StartTime,StopTime) as a key, as stores the results rows as a list.
    It returns the QueryResults dictionary that has for keys the (Tmin,Tmax) tuple, and as values lists of rows,
    where the rows are dictionaries with 3 keys:
    {
    'change_date':datetime.datetime,
    'actual_status':int,
    'dpname':str
    }
    The idea is that any massaging of this information is done later, while this function returns the query results in bulk.
    """
    print "Parsing file %s to extract query results"%ManualO2Ologfilename
    #Open the log:
    ManualO2Olog=open(ManualO2Ologfilename,'r')
    #Loop to extract the start and stop times
    Tmin=""
    Tmax=""
    ReadRows=False
    QueryResultsDict={}
    for line in ManualO2Olog:
        if 'Tmin:' in line:
            if Tmin=="":#This is valid only for the first time... after the  first query use the previous Tmax as Tmin.
                Tmin=datetime.datetime(*[int(x) for x in (line.split(':')[1].strip()).split()])
            else:
                Tmin=Tmax
        if 'Tmax:' in line:
            Tmax=datetime.datetime(*[int(x) for x in (line.split(':')[1].strip()).split()])
            #Assume that when Tmax is read Tmin was read too since we are parsing lines like this:
            #     Tmin: 2010 8 27 10 0 0 0
            #     Tmax: 2010 8 29 9 0 0 0
            #print "%s- Tmin: %s and Tmax: %s (number log lines %s)"%(IOVCounter,Tmin,Tmax,LinesCounter)
            if (Tmin,Tmax) not in QueryResultsDict.keys():
                QueryResultsDict.update({(Tmin,Tmax):[]})
            else:
                print "This Tmax (%s) seems to be duplicated!!!"%Tmax
        if "Dumping all query results" in line:
            ReadRows=True
        if "Finished dumping query results" in line:
            NumOfRows=int(line.split(",")[1].split()[0])
            ReadRows=False
            #Debug output:
            #print "Log reports %s rows, we counted %s rows"%(NumOfRows,len(QueryResultsDict[(Tmin,Tmax)]))
        if ReadRows:
            Row={}
            RowTokens=line.split(",")
            for token in RowTokens:
                if 'CHANGE_DATE' in token:
                    try:
                        #Approximate time to the second, millisecond seems just to be too much information anyway
                        #(one can uncomment following line to do detailed tests if needed for a short time interval)
                        #Row.update({'change_date':datetime.datetime.strptime(token[token.index(":")+2:-5],"%Y/%m/%d %H:%M:%S")})
                        #Complete timestamp including millisecond is parsed with the following commented for now line:
                        changeDate=datetime.datetime.strptime(token[token.index(":")+2:-1],"%Y/%m/%d %H:%M:%S.%f")
                        #Now introduce the check on the timezone:
                        #Based on the fact that we have (in Tmin/Tmax) the 1 hr interval of the query time, we can univocally determine the timezone!
                        #FIXME:
                        #Should not recalculate this at every query result... just store it!
                        #deltaTimezone=datetime.timedelta(seconds=((changeDate-Tmin).seconds/3600)*3600) #Using int / as modulo...
                        #The above is wrong! I did not realize that Tmin is also in UTC!
                        #Need to implement a subclass of tzinfo to handle it properly all the time... to be done later!
                        #For now:
                        deltaTimezone=datetime.timedelta(seconds=7200) #DST+GMT+1 (not valid in non DST times but ok for Run144086...
                        Row.update({'change_date':changeDate+deltaTimezone})
                    except:
                        print "WEIRD!\n Timestamp had a format YYYY/MM/DD HH:MM:0SS.millisecond (extra 0 tabbing for seconds!)"
                        print line
                        #Approximate time to the second, millisecond seems just to be too much information anyway
                        #(one can uncomment following line to do detailed tests if needed for a short time interval)
                        #Row.update({'change_date':datetime.datetime.strptime(token[token.index(":")+2:-5],"%Y/%m/%d %H:%M:0%S")})
                        #Complete timestamp including millisecond is parsed with the following commented for now line:
                        changeDate=datetime.datetime.strptime(token[token.index(":")+2:-1],"%Y/%m/%d %H:%M:0%S.%f")
                        #Now introduce the check on the timezone:
                        #Based on the fact that we have (in Tmin/Tmax) the 1 hr interval of the query time, we can univocally determine the timezone!
                        #FIXME:
                        #Should not recalculate this at every query result... just store it!
                        #deltaTimezone=datetime.timedelta(seconds=((changeDate-Tmin).seconds/3600)*3600) #Using int / as modulo...
                        #The above is wrong! I did not realize that Tmin is also in UTC!
                        #Need to implement a subclass of tzinfo to handle it properly all the time... to be done later!
                        #For now:
                        deltaTimezone=datetime.timedelta(seconds=7200) #DST+GMT+1 (not valid in non DST times but ok for Run144086...
                        Row.update({'change_date':changeDate+deltaTimezone})
                if 'ACTUAL_STATUS' in token:
                    Row.update({'actual_status':int(token[token.index(":")+2:-1])})
                if 'DPNAME' in token:
                    Row.update({'dpname':token[token.index(":")+2:-2]})
            if len(Row)>0: #To avoid the case of an empty Row
                QueryResultsDict[(Tmin,Tmax)].append(Row)
    ManualO2Olog.close()
    return QueryResultsDict 

class TrkVoltageStatus:    
    """
    #How about we make a class and instantiate an object TrackerVoltageStatus that we update at each row:
    #1-initialize it with a list of detID (from the most complete maps) OFF and ON (the masked ones):
    #2-Has a member history, a dictionary, with keeps, by timestamp (key) a list of detIDs with their channel 0,1,2,[3] status (to decide later if we need to already handle here the checks that
    #   a-channel 0==1, for example by having a list of detIDs that do not have matching LV channels statuses (easy to check later, for sure it will be not empty during little transition IOVs due to different sub-second timing... we could introduce deadbands later)
    #   b-a LV ON/OFF bit
    #   c-a HV ON/OFF bit
    #   d-add extra (VMON, V0_SETTINGS) info in the query?
    #   e-add the power group turning off info in the query to check the effect
    #   f-add a RAMPING UP/DOWN flag on top of the conventional O2O ON/OFF?
    #3-define an update(row) method that for each row takes the timestamp:
    #   a-if timestamp already exist, updates the channel status of the relevant detIDs contained
    #   b-if timestamp does not exist, creates an entry for it carrying over the information for the relevant detIDs from previous timestamp and update it with the information in the row
    #4-define a getVoltageHistory(start,end) method to dump the list of IOVs, and for each IOV dump a list with the detIDs and their LV/HV status, the number of LV OFF modules and HV OFF modules, (check that there is no module with LV OFF and HV ON... basic check can be done at  the update level)
    """
    import datetime,sys,copy
    def __init__(self,detIDPSUmapfilename="map.txt",mask_mapfilename="mask_map.txt",DetIDAliasFile="StripDetIDAlias.pkl",startTime=datetime.datetime.now(),debug=False):
        """Initialize the object with a timestamp for the start time of the query (defaults to datetime.datetime.now()) and a list of modules OFF and ON depending on masking map (for now all OFF)"""
        self.debug=debug
        try:
            mapfile=open(detIDPSUmapfilename,'r')
            maplines=mapfile.readlines()
            mapfile.close()
            #Extract the detIDs from the map:
            self.detIDs=[int(line.split()[0]) for line in maplines]
            #Check number of detIDs
            #FIXME PIXEL:
            #Add a flag for pixels (name of the map?) to implement the check for either detector
            if len(list(set(self.detIDs)))!=15148:
                print "There is an issue with the map provided: not reporting 15148 unique detIDs!"
            #Parse the PSU channel part of the map
            #This one does not list all channels, but follows this convention:
            #-detids that are HV mapped list channel002 or channel003
            #-detids that are NOT HV mapped list channel000
            #-detids that are connected to HV1/HV2 crosstalking PSUs list channel999
            self.PSUChannelListed=[line.split()[1] for line in maplines]
            #In any case all PSU are listed so we can extract them this way:
            self.PSUs=list(set([PSUChannel[:-10] for PSUChannel in self.PSUChannelListed]))
            #Check number of PSUs:
            if len(self.PSUs)!=1944:
                print "There is an issue with the map provided: not reporting 1944 unique Power Supplies!"
            #Building now a list of all PSUChannels
            #(since queries to the DB will report results in DPNAMEs, i.e. PSU channels
            self.PSUChannels=[]
            for PSU in self.PSUs:
                self.PSUChannels.append(PSU+"channel000")
                self.PSUChannels.append(PSU+"channel001")
                self.PSUChannels.append(PSU+"channel002")
                self.PSUChannels.append(PSU+"channel003")
            #This part should be unnecessary, since we do not mask any detid anymore...
            #But we leave it since the possibility is in the O2O code itself still.
            try:
                maskfile=open(mask_mapfilename,'r')
                self.masked_detIDs=[int(detID.rstrip()) for detID in maskfile]
                maskfile.close()
            except:
                self.masked_detIDs=[] #empty list of detIDs to be "masked"
                print "No mask map was provided, assuming to start from a complete Tracker OFF state"
            try:
                DetIDAlias=open(DetIDAliasFile,"rb")
                DetIDAliasDict=pickle.load(DetIDAlias)
                DetIDAlias.close()
                self.DetIDAliasDict=DetIDAliasDict
            except:
                self.DetIDAliasDict={}
                print "No valid detID-Alias map pickle file was provided!"
                print "The TkrVoltageStatus object could not be initialized properly!"
                print "Please use an existing detIDPSUChannel mapfilename as argument!"
                sys.exit()
        except:
            print "Could not find detID-PSUchannel map to initialize the detIDs"
            print "The TkrVoltageStatus object could not be initialized properly! Please use an existing detIDPSUChannel mapfilename as argument!"
            sys.exit()
        #With the information from the map build the 2 map dictionaries of interest:
        #DetID->PSUChannelListed (same convention as the map)
        self.DetIDPSUmap={}
        for detID in self.detIDs:
            self.DetIDPSUmap.update({detID:self.PSUChannelListed[self.detIDs.index(detID)]})
        #PSUChannel->DetID map:
        self.PSUDetIDmap={}
        #This map is a bit more complicated than the DetIDPSUmap (that was just a dictionary form of the given detIDPSUChannel map passed as argument):
        #Here we make a 1 to 1 map for all the PSUChannels to all detIDs connected, so for a given PSU:
        #1-all detIDs listed as channel000 and as channel999 will be listed for all PSUchannels (000/001/002/003)
        #2-all detIDs listed as channel002 will be listed for channel002 and for channel000/001
        #3-all detIDs listed as channel003 will be listed for channel003 and for channel000/001
        #NOTE:
        #For Unmapped and Crosstalking channel even though we list all the detids as above (for all channels of the given PSU), we will handle them separately
        #when determining whether the detids are going ON or OFF... see below.
        for PSUChannel in self.PSUChannels:
            if PSUChannel.endswith("0") or PSUChannel.endswith("1"): #Report all channels that match at the PSU level!
                self.PSUDetIDmap.update({PSUChannel:[detid for detid in self.DetIDPSUmap.keys() if PSUChannel[:-3] in self.DetIDPSUmap[detid]]}) #PSU matching (picks all detids by PSU) 
            else: #For channel002 and channel003 list in this map ONLY the actual mapped ones, the unmapped or crosstalking are listed in the corresponding PSUDetIDUnmappedMap and PSUDetIDCrosstalkingMap (see below).
                self.PSUDetIDmap.update({PSUChannel:[detid for detid in self.DetIDPSUmap.keys() if (PSUChannel in self.DetIDPSUmap[detid])]})

        #Separate PSU-based maps for unmapped and crosstalking detids:
        self.PSUDetIDUnmappedMap={}
        UnmappedPSUs=list(set([psuchannel[:-10] for psuchannel in self.DetIDPSUmap.values() if psuchannel.endswith("000")]))
        self.PSUDetIDCrosstalkingMap={}
        CrosstalkingPSUs=list(set([psuchannel[:-10] for psuchannel in self.DetIDPSUmap.values() if psuchannel.endswith("999")]))
        for PSU in self.PSUs:
            if PSU in UnmappedPSUs:
                self.PSUDetIDUnmappedMap.update({PSU:[detid for detid in self.DetIDPSUmap.keys() if (self.DetIDPSUmap[detid].endswith("000") and PSU in self.DetIDPSUmap[detid])]})
            if PSU in CrosstalkingPSUs:
                self.PSUDetIDCrosstalkingMap.update({PSU:[detid for detid in self.DetIDPSUmap.keys() if (self.DetIDPSUmap[detid].endswith("999") and PSU in self.DetIDPSUmap[detid])]})
        #Need also the list of PSU channels that are unmapped and crosstalking with their status!
        #Since the state for those detIDs is determined by the knowledge of the other PSU channel, we need a dictionary for both that keeps their "last value" at all times.
        self.UnmappedPSUChannelStatus={}
        #Initialize the dictionary with all the unmapped (only HV is relevant, channel002/003) PSU channels set to 0 (off).
        for PSU in self.PSUDetIDUnmappedMap.keys():
            self.UnmappedPSUChannelStatus.update({PSU+"channel002":0})
            self.UnmappedPSUChannelStatus.update({PSU+"channel003":0})
        self.CrosstalkingPSUChannelStatus={}
        #Initialize the dictionary with all the crosstalking (only HV is relevant, channel002/003) PSU channels set to 0 (off).
        for PSU in self.PSUDetIDCrosstalkingMap.keys():
            self.CrosstalkingPSUChannelStatus.update({PSU+"channel002":0})
            self.CrosstalkingPSUChannelStatus.update({PSU+"channel003":0})
        
        #Now let's initialize the object (data member) we will use to keep track of the tracker status:
        
        #Make the startTime of the initialization an attribute of the object:
        self.startTime=startTime
        #Basic structure of the internal history dictionary, based on query results
        self.PSUChannelHistory={startTime:range(len(self.PSUChannels))}
        #NOTE:
        #Using the indeces to the self.PSUChannels list of PSUChannels to spare memory...
        #Strategy:
        #keep a timestamped list of indeces to PSUChannels
        #Dynamically fetch the PSUChannels (or corresponding detIDs via PSUDetIDmap dict) on the fly whey querying the status.
        
        #For Masked DetIDs set them always ON (since they would be detIDs that we don't know to with PSU to associate them to
        #This should NOT BE THE CASE any longer, since all detIDs are now mapped at least to a PSU
        #TOBEIMPLEMENTED:
        #(some missing HV channel, but we treat those differently: turn them off in case either HV1/HV2 is OFF)
        #(some having cross-talking PSUs, we treat those differently: turn them OFF only if both HV1/HV2 are OFF)
        for detID in self.masked_detIDs:
            #For each detID get the PSUChannelListed, then remove last digit (the channel number that can be 0,2 or 3)
            #and then make sure to remove all channels 000,001,002,003 
            self.PSUChannelHistory[startTime].remove(index(detIDPSUmap[detid][-1]+'0')) #Channel000
            self.PSUChannelHistory[startTime].remove(index(detIDPSUmap[detid][-1]+'1')) #Channel001
            self.PSUChannelHistory[startTime].remove(index(detIDPSUmap[detid][-1]+'2')) #Channel002
            self.PSUChannelHistory[startTime].remove(index(detIDPSUmap[detid][-1]+'3')) #Channel003
        #Let's try to keep the one following dict that reports the number of detIDs with HV OFF and LV OFF at a given timestamp...
        #Not sure even just this dict makes sense... but it is the most used information, quite minimal compare to the status of each detID
        #and it provides a global IOV for the whole tracker
        #NOTE:
        #it won't track the case of a channel going down at the exact time another goes up!
        #self.TkNumberOfDetIDsOff={startTime:(len([PSUChannel[i] for i in self.PSUChannelHistory]),len(self.detIDs))} 

    def updateO2OQuery(self,row):
        """Function to automatically handle the updating the of history dictionary
        with the results rows from the O2O query (PSUchannel name based).
        Note that row can be an actual row from SQLAlchemy from direct online DB querying
        but also a dictionary extracted parsing ManualO2O.log files.
        """
        #NOTE:May need to introduce IOV reduction at this level,
        #but I'd prefer to handle it separately so to conserve the original data for validation purposes.
        #For new IOV create a new entry in the dict and initialize it to the last IOV status
        #since only change is reported with STATUS_CHANGE queries.

        #First check if we care about this channel (could be a Pixel channel while we want only Strip channels or viceversa):
        if row['dpname'] not in self.PSUChannels:
            if self.debug:
                print "Discarding results row since it is not in the currently considered map:\n%s"%row['dpname']
            return        
        
        #NOTE:
        #Actually use the getIOV to allow for "asynchronous" updating:
        lastTimeStamp,nextTimeStamp=self.getIOV(row['change_date'],self.PSUChannelHistory)
        #Print a warning when the timeStamp is not the last one in the history!
        if self.debug:
            if row['change_date'] not in self.PSUChannelHistory.keys():
                if nextTimeStamp!="Infinity":
                    print "WARNING! Asynchronous updating of the Tracker Voltage Status"
                    print "WARNING! Inserting an IOV between %s, and %s existing timestamps!"%(lastTimeStamp,nextTimeStamp)
                    print "The state will be correctly updated (if necessary) starting from the state at %s"%lastTimeStamp
                else:
                    print "Appending one more IOV to the PSUChannelHistory dictionary"
            else:
                print "Updating already present IOV with timestamp %s"%row['change_date']
        #The fact that we edit the lasttimestamp takes care of updating existing IOVs the same way as for new ones...
        #Update the internal dictionary modifying the last time stamp state...
        #Case of channel being ON (it's only ON if CAEN reports 1)
        PSUChannelOFFList=self.PSUChannelHistory[lastTimeStamp][:]
        if row['actual_status']==1: #Switching from string to int here to make sure it will be compatible with SQLAlchemy query for testing with direct DB access.
            try:
                PSUChannelOFFList.remove(self.PSUChannels.index(row['dpname']))
                self.PSUChannelHistory.update({row['change_date']:PSUChannelOFFList})
            except:
                if self.debug:
                    print "Found a channel that turned ON but as already ON apparently!"
                    print row
        else: #Case of channel being OFF (it's considered OFF in any non 1 state, ramping, error, off)
            PSUChannelOFFList.append(self.PSUChannels.index(row['dpname']))
            self.PSUChannelHistory.update({row['change_date']:list(set(PSUChannelOFFList))})
        
        #First handle the detID based dict:
        #Use the map to translate the DPNAME (PSUChannel) into a detid via internal map:
        #CAREFUL... need to protect against pixel PSU channels being reported by the query!
        #try:
        #    detIDList=self.PSUDetIDmap[row['dpname']]
        #except:
        #    #FIXME: develop the pixel testing part here later...
        #    print "PSU channel (DPNAME) reported in not part of the wanted subdetector (only Pixel or Strips at one time)"
        #    print "DPNAME=%s"%row['dpname']
        #    detIDList=[] #Set it to empty list...
        ##print detID, type(detID)
        #for detID in detIDList:
        #    #Get the previous list of channel statuses for the relevant detID:
        #    ModuleStatus=self.history[row['change_date']][detID]
        #    #Update it with the current query result row:
        #    #DISTINGUISH LV and HV case:
        #    if row['dpname'].endswith('channel000') or row['dpname'].endswith('channel001'):
        #        ModuleStatus[0]=row['actual_status'] #{detid:[LV,HV]} convention is ModuleStatus[0]=0 is LV OFF
        #    else:
        #        ModuleStatus[1]=row['actual_status']
        #    #Update the history dictionary:
        #    self.history[row['change_date']].update({detID:ModuleStatus})
        ##Add a check for the possibility of having a second IOV when only channel000/channel001 changed,
        ##i.e. the status did not change as a consequence for the detID based history dictionary
        ##(this does not affect the historyPSUChannel dictionary by definition)
        #if self.history[row['change_date']]==self.history[TimeStamps['detID'][0]] and row['change_date']!=TimeStamps['detID'][0]: #This last condition is just in case something funky happens, to avoid removing good IOVs
        #    if self.debug:
        #        print "Found that the update for timestamp %s, does not change the detid status w.r.t timestamp %s"%(row['change_date'],TimeStamps['detID'][0])
        #        print "Eliminating the temporarily created entry in the history dictionary (it will still be logged in the historyPSUChannel dictionary"
        #    self.history.pop(row['change_date'])
        ##Then handle the PSUChannel based one:
        #self.historyPSUChannel[row['change_date']].update({row['dpname']:row['actual_status']})
    
                    
    def updateDetIDQuery(self,row):
        """Function to automatically handle the updating of the history dictionary
        with the result rows from detID based query
        """
        #FIXME:
        #this will have to be developed once the detID query is ready and being used in the Tracker DCS O2O
        if row['change_date'] not in self.history.keys():#New timestamp (i.e. new IOV)
            #For new IOV create a new entry in the dict and initialize it to the last IOV status
            #since only change is reported with STATUS_CHANGE queries.
            lastTimeStamp=sorted(self.history.keys()).pop(0)
            self.history.update({row['change_date']:self.history[lastTimeStamp]})
        #Get the previous list of channel statuses for the relevant detID:            
        channelStatus=self.history[row['change_date']][row['detid']]
        #Update it with the current query result row:
        channelStatus[row['channel']]=row['voltage_status']
        #Update the history dictionary:
        self.history[row['change_date']].update({row['detid']:channelStatus})
        
        
    #Define the getters:

    #First the TIMESTAMP based getters:
    
    def getPSUChannelsOff(self,timestamp):
        """
        Function that given a timestamp, returns the list of PSUChannels OFF, the list of PSUChannels ON and the IOV (start/stop timestamps) of the reported status.
        NOTE: No distinction between HV and LV is made since all PSUChannels (000/001 and 002/003) are reported.
        """
        StartTime,StopTime=self.getIOV(timestamp,self.PSUChannelHistory)
        PSUChannelsOFF=map(lambda x: self.PSUChannels[x],self.PSUChannelHistory[StartTime])
        PSUChannelsON=list(set(self.PSUChannels).difference(set(PSUChannelsOFF)))
        return PSUChannelsOFF,PSUChannelsON,StartTime,StopTime
    
    def getDetIDsHVOff(self,timestamp):
        """
        Function that given a timestamp, returns the list of detIDs with HV OFF, the list of detIDs with HV ON and the IOV (start/stop times) of the reported status.
        """
        StartTime,StopTime=self.getIOV(timestamp,self.PSUChannelHistory)
        DetIDsHVOFF=[]
        #A little too nasty python oneliner:
        #[DetIDsHVOFF.extend(i) for i in map(lambda x: self.PSUDetIDmap[self.PSUChannels[x]],[index for index in self.PSUChannelHistory[StartTime] if (self.PSUChannels[index].endswith("2") or self.PSUChannels[index].endswith("3"))])]
        #It actually does not work, since we need to consider unmapped and crosstalking channels and handle them differently, let's see:
        for index in self.PSUChannelHistory[StartTime]:
            #First check only for HV channels!
            if self.PSUChannels[index].endswith("2") or self.PSUChannels[index].endswith("3") :
                #Then check HV MAPPED channels:
                if self.PSUChannels[index] in self.PSUDetIDmap.keys():
                    #print "Extending the list of DetIdsHVOff with the positively matched detids:",self.PSUDetIDmap[self.PSUChannels[index]]
                    DetIDsHVOFF.extend(self.PSUDetIDmap[self.PSUChannels[index]])
                #Furthermore check the unmapped channels:
                if self.PSUChannels[index][:-10] in self.PSUDetIDUnmappedMap.keys():
                    #To turn OFF unmapped channels there is no need to check the "other" channel:
                    #print "Extending the list of DetIdsHVOff with the HV unmapped (PSU-)matched detids:",self.PSUDetIDUnmappedMap[self.PSUChannels[index][:-10]]
                    DetIDsHVOFF.extend(self.PSUDetIDUnmappedMap[self.PSUChannels[index][:-10]])
                #Further check the crosstalking channels:
                if self.PSUChannels[index][:-10] in self.PSUDetIDCrosstalkingMap.keys():
                    #To turn OFF crosstalking channels we need to check that the other channel is OFF too!
                    if (self.PSUChannels.index(self.PSUChannels[index][:-1]+"2") in self.PSUChannelHistory[StartTime]) and (self.PSUChannels.index(self.PSUChannels[index][:-1]+"3") in self.PSUChannelHistory[StartTime]):
                        #print "Extending the list of DetIdsHVOff with the HV-CROSSTALKING (PSU-)matched detids:",self.PSUDetIDCrosstalkingMap[self.PSUChannels[index][:-10]]
                        DetIDsHVOFF.extend(self.PSUDetIDCrosstalkingMap[self.PSUChannels[index][:-10]])
        DetIDsHVON=list(set(self.detIDs).difference(set(DetIDsHVOFF)))
        return list(set(DetIDsHVOFF)),DetIDsHVON,StartTime,StopTime
    
    def getDetIDsLVOff(self,timestamp):
        """
        Function that given a timestamp, returns the list of detIDs with LV OFF, the list of detIDs with LV ON and the IOV (start/stop times) of the reported status.
        """
        #Note that channels with LV OFF naturally should have HV OFF too!
        StartTime,StopTime=self.getIOV(timestamp,self.PSUChannelHistory)
        DetIDsLVOFF=[]
        for detids in map(lambda x: self.PSUDetIDmap[self.PSUChannels[x]],[index for index in self.PSUChannelHistory[StartTime] if (self.PSUChannels[index].endswith("0") or self.PSUChannels[index].endswith("1"))]):
            DetIDsLVOFF.extend(detids)
        DetIDsLVON=list(set(self.detIDs).difference(set(DetIDsLVOFF)))
        return list(set(DetIDsLVOFF)),DetIDsLVON,StartTime,StopTime

    def getAliasesHVOff(self,timestamp):
        """
        Function that given a timestamp, returns the list of (PSU) Aliases with HV OFF, the list of (PSU) Aliases with HV ON and the IOV (start/stop times) of the reported status. 
        """
        DetIDsHVOFF,DetIDsHVON,StartTime,StopTime=self.getDetIDsHVOff(timestamp)
        AliasesHVOFF=list(set([list(self.DetIDAliasDict[detid])[0] for detid in DetIDsHVOFF])) #FIXME: check on fixing the StripDetIDAlias.pkl dictionary... no need of a set as a result! actually need to check if there is more than one element, that would be an ERROR!
        AliasesHVON=list(set([list(self.DetIDAliasDict[detid])[0] for detid in DetIDsHVON]))
        return AliasesHVOFF,AliasesHVON,StartTime,StopTime

    def getAliasesLVOff(self,timestamp):
        """
        Function that given a timestamp, returns the list of (PSU) Aliases with HV OFF, the list of (PSU) Aliases with HV ON and the IOV (start/stop times) of the reported status.
        """
        DetIDsLVOFF,DetIDsLVON,StartTime,StopTime=self.getDetIDsLVOff(timestamp)
        AliasesLVOFF=list(set([list(self.DetIDAliasDict[detid])[0] for detid in DetIDsLVOFF])) #FIXME: check on fixing the StripDetIDAlias.pkl dictionary... no need of a set as a result! actually need to check if there is more than one element, that would be an ERROR!
        AliasesLVON=list(set([list(self.DetIDAliasDict[detid])[0] for detid in DetIDsLVON]))
        return AliasesLVOFF,AliasesLVON,StartTime,StopTime

    #Then the Value vs. time getters:

    def getDetIDsHVOffVsTime(self):
        """
        Function that returns the number of DetIDs with HV OFF vs time for all IOVs available in the PSUChannelHistory.
        The results is one list of tuples [(datetime.datetime,#DetIDsHVOFF),...] and the timestamp of the last entry processed in the PSUChannelHistory.
        This is important, so that the user can check until when in time the last tuple in the list is valid until (closing the last IOV).
        This can be easily used to
        -Do spot checks by IOV: picking a timestamp inside one of the IOVs one can use the getDetIDsHVOff function to get a list of the detIDs ON/OFF.
        -Plot the graph of number of HV OFF (or ON by complementing to 15148) vs time using pyRoot
        """
        #Loop over the timestamps:
        DetIDsHVOffVsTime=[]
        PreviousTimeStamp=None
        PreviousDetIDsHVOFF=[]
        for timestamp in sorted(self.PSUChannelHistory.keys()):
            DetIDsHVOFF=self.getDetIDsHVOff(timestamp)[0] #Use the first returned value (DetIDsHVOFF) from getDetIDsHVOff(timestamp)!
            #Check with respect to previous IOV, using set functionality, so that only relevant IOVs where there was an actual HV change are reported!
            if PreviousTimeStamp:
                if DetIDsHVOFF!=PreviousDetIDsHVOFF: #If there is a previous timestamp to compare to, look for differences before reporting!
                    DetIDsHVOffVsTime.append((timestamp,len(DetIDsHVOFF)))
            else: #First IOV start... nothing to compare to:
                DetIDsHVOffVsTime.append((timestamp,len(DetIDsHVOFF)))
            PreviousTimeStamp=timestamp
            PreviousDetIDsHVOFF=DetIDsHVOFF
        LastTimeStamp=PreviousTimeStamp
        return DetIDsHVOffVsTime,PreviousTimeStamp 

    def getDetIDsLVOffVsTime(self):
        """
        Function that returns the number of DetIDs with LV OFF vs time for all IOVs available in the PSUChannelHistory.
        The results is one list of tuples [(datetime.datetime,#DetIDsLVOFF),...], basically 1 entry per IOV (change in the LV channels only),
        and the timestamp of the last entry processed in the PSUChannelHistory. This timestamp is important, so that the user can check until when in time the last tuple in the list is valid until (closing the last IOV).
        This can be easily used to
        -Do spot checks by IOV: picking a timestamp inside one of the IOVs one can use the getDetIDsHVOff function to get a list of the detIDs ON/OFF.
        -Plot the graph of number of HV OFF (or ON by complementing to 15148) vs time using pyRoot
        """
        #Loop over the timestamps:
        DetIDsLVOffVsTime=[]
        PreviousTimeStamp=None
        for timestamp in sorted(self.PSUChannelHistory.keys()):
            DetIDsLVOFF=set(self.getDetIDsLVOff(timestamp)[0]) #Use the first returned value (DetIDsLVOFF) from getDetIDsLVOff(timestamp)!Use set() to be able to compare the sets with != otherwise for lists, the order also makes them different!
            #Check with respect to previous IOV, using set functionality, so that only relevant IOVs where there was an actual LV change are reported!
            if PreviousTimeStamp:
                if DetIDsLVOFF!=PreviousDetIDsLVOFF: #If there is a previous timestamp to compare to, look for differences before reporting! 
                    DetIDsLVOffVsTime.append((timestamp,len(DetIDsLVOFF)))
            else: #First IOV start... nothing to compare to:
                DetIDsLVOffVsTime.append((timestamp,len(DetIDsLVOFF)))
            PreviousTimeStamp=timestamp
            PreviousDetIDsLVOFF=DetIDsLVOFF
        LastTimeStamp=PreviousTimeStamp
        return DetIDsLVOffVsTime,LastTimeStamp
    
    #Data massagers:
    def getArraysForTimePlots(self,TimeStampsValuesTuple):
        """
        Function that given a tuple with (datetime.datetime,values) returns the arrays ready to the plotting vs time function (basically renders data histogram-like adding extra data points).
        """
        import array, time

        #Initialize the lists of timestamps (to be doubled to get the histogram/IOV look)
        Timestamps=[]
        Values=[]
        #Loop over the list of tuples with (timestamp,numberofdetidswithLVOFF):
        for item in sorted(TimeStampsValuesTuple):
           #FIXME:
           #NEED to check the effect/use of tzinfo to avoid issues with UTC/localtime DST etc.
           #Will have to massage the datetime.datetime object in a proper way not necessary now.
           #Also need to decide if we want add the milliseconds into the "timestamps for plotting in root... not straightforward so will implement only if needed. For now know that the approximation is by truncation for now!
           timestamp=int(time.mktime(item[0].timetuple())) #Need to get first a time tuple, then translate it into a unix timestamp (seconds since epoc). This means no resolution under 1 second by default
           if item==TimeStampsValuesTuple[0]: #First item does not need duplication
               Timestamps.append(timestamp)
               Value=item[1]
               Values.append(Value)
           else: #Add twice each timestamp except the first one (to make a histogram looking graph)
               Timestamps.append(timestamp)
               Values.append(Value) #Input previous value with new timestamp
               Value=item[1]
               Timestamps.append(timestamp)
               Values.append(Value) #Input new value with new timestamp
        #Need to render the two lists as arrays for pyROOT
        TimestampsArray=array.array('i',Timestamps)
        ValuesArray=array.array('i',Values)

        return TimestampsArray,ValuesArray

    def getReducedArraysForTimePlots(self,TimeStamps,Values):
        """
        Implement IOV reduction based on timestamp delta, following O2O algorithm.
        """
        
        return
    
    #The plotters:
    def plotGraphSeconds(self,TimeArray,ValuesArray,GraphTitle="Graph",YTitle="Number of channels",GraphFilename="test.gif"):
        """
        Function that given an array with timestamps (massaged to introduce a second timestamp for each value to produce an histogram-looking plot) and a corresponding array with values to be plotted, a title, a Y axis title and a plot filename, produces with pyROOT a time plot and saves it locally.
        The function can be used for cumulative plots (number of channels with HV/LV OFF/ON vs time) or for individual (single detID HV/LV status vs time) plots.
        """
        import ROOT
        canvas=ROOT.TCanvas()
        graph=ROOT.TGraph(len(TimeArray),TimeArray,ValuesArray)
        graph.GetXaxis().SetTimeDisplay(1)
        graph.GetXaxis().SetLabelOffset(0.02)
        #Set the time format for the X Axis labels depending on the total time interval of the plot!
        TotalTimeIntervalSecs=TimeArray[-1]-TimeArray[0]
        if TotalTimeIntervalSecs <= 120: #When zooming into less than 2 mins total interval report seconds too
            graph.GetXaxis().SetTimeFormat("#splitline{   %d/%m}{%H:%M:%S}")
        elif 120 < TotalTimeIntervalSecs < 6400: #When zooming into less than 2 hrs total interval report minutes too
            graph.GetXaxis().SetTimeFormat("#splitline{%d/%m}{%H:%M}")
        else: #When plotting more than 2 hrs only report the date and hour of day
            graph.GetXaxis().SetTimeFormat("#splitline{%d/%m}{  %H}")
        graph.GetYaxis().SetTitle(YTitle)
        graph.GetYaxis().SetTitleOffset(1.4)
        graph.SetTitle(GraphTitle)
        graph.Draw("APL")
        canvas.SaveAs(GraphFilename)
        print "Saved graph as %s"%GraphFilename
        return
    def plotPSUChannelvsTime(self,TimeArray,ValuesArray,GraphTitle="PSUChannelGraph",YTitle="Channel HV Status",GraphFilename="PSUChannel.gif"):
        """
        Function that given an array with timestamps (massaged to introduce a second timestamp for each value to produce an histogram-looking plot) and a corresponding array with values to be plotted, a title, a Y axis title and a plot filename, produces with pyROOT a time plot and saves it locally.
        The function can be used for cumulative plots (number of channels with HV/LV OFF/ON vs time) or for individual (single detID HV/LV status vs time) plots.
        """
        import ROOT
        canvas=ROOT.TCanvas()
        graph=ROOT.TGraph(len(TimeArray),TimeArray,ValuesArray)
        graph.GetXaxis().SetTimeDisplay(1)
        graph.GetXaxis().SetLabelOffset(0.02)
        #Set the time format for the X Axis labels depending on the total time interval of the plot!
        TotalTimeIntervalSecs=TimeArray[-1]-TimeArray[0]
        if TotalTimeIntervalSecs <= 120: #When zooming into less than 2 mins total interval report seconds too
            graph.GetXaxis().SetTimeFormat("#splitline{   %d/%m}{%H:%M:%S}")
        elif 120 < TotalTimeIntervalSecs < 6400: #When zooming into less than 2 hrs total interval report minutes too
            graph.GetXaxis().SetTimeFormat("#splitline{%d/%m}{%H:%M}")
        else: #When plotting more than 2 hrs only report the date and hour of day
            graph.GetXaxis().SetTimeFormat("#splitline{%d/%m}{  %H}")
        graph.GetYaxis().SetTitle(YTitle)
        graph.GetYaxis().SetTitleOffset(1.4)
        graph.SetTitle(GraphTitle)
        graph.Draw("APL")
        canvas.SaveAs(GraphFilename)
        print "Saved graph as %s"%GraphFilename
        return
    
    def plotHVOFFvsTime(self):
        """
        """
        return
    
    def plotLVOFFvsTime(self):
        """
        """
        return

    def plotDetIDHVOFFVsTime(self,detID):
        """
        """
        return
    def plotDetIDLVOFFVsTime(self,detID):
        """
        """
        return

    def plotDetIDHistory(self):
        return

    def getIOV(self,timeStamp,HistoryDict):
        """
        Function that given a timeStamp return the TimeStampStart and TimeStampStop of the IOV in the wanted HistoryDict (assumed to have timestamps as keys) that contains the timestamp.
        This can be used by all functions that need to access HistoryDict by timeStamp.
        """        
        
        TimeStamps=HistoryDict.keys()[:]
        #Add the wanted timeStamp to the list of TimeStamps (if it is not there already!)
        if timeStamp not in TimeStamps:
            TimeStamps.append(timeStamp)
            #Sort the list with the timestamp we added, so it will fall in between the wanted StartTime and EndTime.
            TimeStamps.sort()            
            TimeStampStart=TimeStamps[TimeStamps.index(timeStamp)-1]
            #Remember to remove it now, so that TimeStamps is still the list of available timestamps!
            TimeStamps.remove(timeStamp)
        else:#If the timeStamp matches one in the file, then the StartTime is that timestamp ;)
            TimeStamps.sort() #this is still needed since the keys of a dictionary are not automatically sorted, but we assume TimeStamps to be sorted to spot TimeStampStop...
            TimeStampStart=timeStamp
        #For the TimeStampStop check the case of hitting the last IOV that is valid until infinity...
        if len(TimeStamps)>TimeStamps.index(TimeStampStart)+1:
            TimeStampStop=TimeStamps[TimeStamps.index(TimeStampStart)+1]
        else:
            TimeStampStop="Infinity"
        if self.debug:
            print "TimeStamp %s falls into IOV starting at %s and ending at %s"%(timeStamp,TimeStampStart,TimeStampStop)
        return (TimeStampStart,TimeStampStop)

    def getIOVsInTimeInterval(self,StartTime,StopTime,HistoryDict):
        """
        Function that returns the IOV timestamps (ordered) contained in a given interval.
        """
        #Copy and sort the timestamps in a list
        TimeStamps=sorted(HistoryDict.keys()[:])
        IOVsInTimeInterval=[]
        #loop over them:
        for timestamp in TimeStamps:
            if timestamp>=StartTime and timestamp<=StopTime: #Pick only the timestamps inside the time interval specified
                if self.debug:
                    print "Found timestamp %s in the wanted interval [%s,%s]"%(timestamp,StartTime,StopTime)
                IOVsInTimeInterval.append(timestamp)
        return IOVsInTimeInterval

    def getReducedIOVs (self,StartTime,StopTime,HistoryDict,deltaT,maxIOVLength):
        """
        Apply the reduction algorithm to the timeintervals and return them so that one can test the reduction (and do plots more easily).
        """
        deltaTime=datetime.timedelta(seconds=deltaT)
        maxSequenceLength=datetime.timedelta(seconds=maxIOVLength)
        #Copy and sort the timestamps in a list:
        TimeStamps=sorted(HistoryDict.keys()[:])
        ReducedIOVs=TimeStamps[:]
        PreviousTimestamp=TimeStamps[0] 
        SequenceStart=TimeStamps[0]  #Initialization irrelevant see loop
        SequenceOn=False
        #Loop over timestamps:
        #Note:Leave the ramp-up/down logic for later (an other function), for now we just do IOV reduction: don't care if we are ramping up or down!
        for timestamp in TimeStamps[1:]: #Skip the first timestamp!(since we initialize PreviousTimeStamp to the first timestamp!)
            #Check only within the interval specified:
            if timestamp>=StartTime and timestamp<=StopTime:
                #Check if the timestamp is within the wanted deltaT from previous timestamp
                if timestamp-PreviousTimestamp<=deltaTime:
                    #Check if there is already an ongoing sequence of IOVs:
                    if SequenceOn:
                        #Check that this timestamp is not farther away than the maximum IOV sequence
                        if (timestamp-SequenceStart)<=maxSequenceLength and timestamp!=TimeStamps[-1]:#need to handle the last time stamp differently!
                            if self.debug:
                                print "Eliminating timestamp %s since it is %s (<=%s)seconds from previous timestamp %s and %s (<=%s) seconds from the IOV sequence start %s!"%(timestamp,timestamp-PreviousTimestamp,deltaTime,PreviousTimestamp,timestamp-SequenceStart,maxSequenceLength,SequenceStart)
                            ReducedIOVs.remove(timestamp)
                        elif timestamp==TimeStamps[-1]: #special case of last timestamp in the list of input timestamps!
                            if self.debug:
                                print "###Terminating the IOV sequence started with %s, since current timestamp %s is the last one in the input list of timestamps to REDUCE!"%(SequenceStart,timestamp)
                        else:
                            #Terminate the sequence (keep the timestamp):
                            SequenceOn=False
                            #Reappend the last item of the sequence (that was automatically removed not knowing it would be the last of the sequence):
                            ReducedIOVs.append(PreviousTimestamp)
                            #Re-order the list via sort():
                            ReducedIOVs.sort()
                            if self.debug:
                                print "###Terminating the IOV sequence started with %s, since current timestamp %s is %s seconds (>%s) away from the IOV sequence starting timestamp (%s). Re-appending the last timestamp in the sequence %s."%(SequenceStart,timestamp,timestamp-SequenceStart,maxIOVLength,PreviousTimestamp,PreviousTimestamp)
                    else:
                        #Start the sequence
                        SequenceOn=True
                        #Save the first timestamp of the sequence (it will be used to check sequence length)
                        if self.debug:
                            print "@@@Starting a new IOV sequence with previous timestamp %s (current being %s)"%(PreviousTimestamp,timestamp)
                        #Still get rid of the current (second in the sequence) IOV:
                        ReducedIOVs.remove(timestamp)
                        SequenceStart=PreviousTimestamp
                else:
                    #Check if there is already an ongoing sequence of IOVs:
                    if SequenceOn:
                        #Terminate the sequence since timestamp is further than deltaT from previous timestamp:
                        SequenceOn=False
                        #Reappend the last item of the sequence (that was automatically removed not knowing it would be the last of the sequence):
                        ReducedIOVs.append(PreviousTimestamp)
                        #Re-order the list via sort():
                        ReducedIOVs.sort()
                        if self.debug:
                            print "$$$Terminating the IOV sequence started with %s, since current timestamp %s is %s seconds (>%s) away from the previous timestamp (%s) in the sequence. Re-appending the last timestamp in the sequence %s."%(SequenceStart,timestamp,timestamp-PreviousTimestamp,deltaT,PreviousTimestamp,PreviousTimestamp)
                                                            
            else:
                #The following is conventional of course:
                #Since we are outside the wanted time interval we should still kick out the timestamp,
                #Basically we will only report a reduce sequence of timestamps within a certain interval
                ReducedIOVs.remove(timestamp)
                #If there is an ongoing sequence that stretches outside the StopTime:
                if SequenceOn:
                    #Terminate sequence (should check the results if they finish with a timestamp>StopTime):
                    SequenceOn=False
                    #Reappend the last item of the sequence (that was automatically removed not knowing it would be the last of the sequence):
                    ReducedIOVs.append(PreviousTimestamp)
                    #Re-order the list via sort():
                    ReducedIOVs.sort()
                    if self.debug:
                        print "^^^ Terminating the IOV sequence started with %s, since current timestamp %s is no more in the wanted time interval under investigation ([%s,%s]). Sequence might have been continuing beyond the StopTime %s limit! Re-appending the last timestamp in the sequence %s."%(SequenceStart,timestamp,StartTime,StopTime,StopTime,PreviousTimestamp)
            PreviousTimestamp=timestamp
        return ReducedIOVs

                    
#Messing around...
TkStatus=TrkVoltageStatus(detIDPSUmapfilename=os.path.join(os.getenv('CMSSW_BASE'),'src/CalibTracker/SiStripDCS/data/StripPSUDetIDMap_FromJan132010_Crosstalk.dat'),DetIDAliasFile=os.path.join(os.getenv('CMSSW_BASE'),'src/CalibTracker/SiStripDCS/data/StripDetIDAlias.pkl'),startTime=datetime.datetime(2010,8,27,12,00,00),debug=True)

#row1={'change_date':datetime.datetime(2010,8,27,10,37,54,732),'actual_status':1,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel001"}
#
#row2={'change_date':datetime.datetime(2010,8,27,11,37,54,732),'actual_status':0,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel001"}
#
#row3={'change_date':datetime.datetime(2010,8,27,12,37,54,732),'actual_status':1,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel001"}
#
#row4={'change_date':datetime.datetime(2010,8,27,10,37,54,732),'actual_status':1,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel002"}
#
#row5={'change_date':datetime.datetime(2010,8,27,11,37,54,732),'actual_status':0,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel002"}
#
#row6={'change_date':datetime.datetime(2010,8,27,12,37,54,732),'actual_status':1,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel002"}
#
#row7={'change_date':datetime.datetime(2010,8,27,11,37,54,732),'actual_status':0,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel002"}
#
#row8={'change_date':datetime.datetime(2010,8,27,10,37,54,735),'actual_status':1,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_1/branchController04/easyCrate0/easyBoard05/channel002"}
#
##Initialization IOV:
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row1)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row2)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row3)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row4)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row5)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row6)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row7)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.updateO2OQuery(row8)
#print "detID IOVs",TkStatus.history.keys()
#print "PSUChannel IOVs",TkStatus.historyPSUChannel.keys()
#
#TkStatus.getHVOnModules(datetime.datetime(2010, 8, 27, 10, 00, 54, 732))

#Test Crosstalking channels handling:
#rowCross1={'change_date':datetime.datetime(2010,8,27,10,37,54,732),'actual_status':1,'dpname':"cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel002"}
#rowCross2={'change_date':datetime.datetime(2010,8,27,10,37,57,732),'actual_status':1,'dpname':"cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel003"}
#rowCross3={'change_date':datetime.datetime(2010,8,27,10,38,00,732),'actual_status':0,'dpname':"cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel002"}
#rowCross4={'change_date':datetime.datetime(2010,8,27,10,38,05,732),'actual_status':0,'dpname':"cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel003"}
#print "Starting number of PSUChannels OFF is %s and of DetIDs ON is %s"%(len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,10,30,00),TkStatus.PSUChannelHistory)[0]]),len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010,8,27,10,30,00),TkStatus.PSUChannelHistory)[0])[1]))                                                                         
#TkStatus.updateO2OQuery(rowCross1)
#print "Turning ON HV crosstalking channel cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel002 number of channels OFF is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,10,37,55),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010,8,27,10,37,55),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010,8,27,10,37,55),TkStatus.PSUChannelHistory)[0])[1]
#TkStatus.updateO2OQuery(rowCross2)
#print "Turning ON HV crosstalking channel cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel003 number of channels OFF is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,10,37,58),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 10, 37, 58),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 10, 37, 58),TkStatus.PSUChannelHistory)[0])[1]
#TkStatus.updateO2OQuery(rowCross3)
#print "Turning OFF HV crosstalking channel cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel002 number of channels off is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,10,38,02),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 10, 38, 02),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 10, 38, 02),TkStatus.PSUChannelHistory)[0])[1]
#TkStatus.updateO2OQuery(rowCross4)
#print "Turning OFF HV crosstalking channel cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel003 number of channels off is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,10,38,07),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 10, 38, 07),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 10, 38, 07),TkStatus.PSUChannelHistory)[0])[1]
#
##Test Unmapped channels handling:
#rowUnmap1={'change_date':datetime.datetime(2010,8,27,11,37,54,732),'actual_status':1,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate1/easyBoard07/channel002"}
#rowUnmap2={'change_date':datetime.datetime(2010,8,27,11,37,57,732),'actual_status':1,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate1/easyBoard07/channel003"}
#rowUnmap3={'change_date':datetime.datetime(2010,8,27,11,38,00,732),'actual_status':0,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate1/easyBoard07/channel002"}
#rowUnmap4={'change_date':datetime.datetime(2010,8,27,11,38,05,732),'actual_status':0,'dpname':"cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate1/easyBoard07/channel003"}
#print "Starting number of PSUChannels OFF is %s and of DetIDs ON is %s"%(len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,11,30,00),TkStatus.PSUChannelHistory)[0]]),len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010,8,27,11,30,00),TkStatus.PSUChannelHistory)[0])[1]))                                                                         
#TkStatus.updateO2OQuery(rowUnmap1)
#print "Turning ON HV crosstalking channel cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate1/easyBoard07/channel002  number of channels OFF is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,11,37,55),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010,8,27,11,37,55),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010,8,27,11,37,55),TkStatus.PSUChannelHistory)[0])[1]
#TkStatus.updateO2OQuery(rowUnmap2)
#print "Turning ON HV crosstalking channel cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate1/easyBoard07/channel003 number of channels OFF is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,11,37,58),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 11, 37, 58),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 11, 37, 58),TkStatus.PSUChannelHistory)[0])[1]
#TkStatus.updateO2OQuery(rowUnmap3)
#print "Turning OFF HV crosstalking channel cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate1/easyBoard07/channel002 number of channels off is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,11,38,02),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 11, 38, 02),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 11, 38, 02),TkStatus.PSUChannelHistory)[0])[1]
#TkStatus.updateO2OQuery(rowUnmap4)
#print "Turning OFF HV crosstalking channel cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_9/branchController04/easyCrate0/easyBoard09/channel003 number of channels off is %s"%len(TkStatus.PSUChannelHistory[TkStatus.getIOV(datetime.datetime(2010,8,27,11,38,07),TkStatus.PSUChannelHistory)[0]])
#print "and related number of detIDs listed as ON is %s"%len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 11, 38, 07),TkStatus.PSUChannelHistory)[0])[1])
#print TkStatus.getDetIDsHVOff(TkStatus.getIOV(datetime.datetime(2010, 8, 27, 11, 38, 07),TkStatus.PSUChannelHistory)[0])[1]
#
#Test function that parses ManualO2O.log to get by timeinterval the results of each O2O query
QueryResults=GetQueryResults("ManualO2O.log")
#QueryResultsPickle=open("QueryResultsDict.pkl","wb")
#pickle.dump(QueryResults,QueryResultsPickle)

#Now that we have this we can do:
counter=0
hours=72 #introduced to test only a few hours, setting it to 1000 to process ALL IOVS. 
for interval in sorted(QueryResults.keys()):
    counter+=1
    if counter<hours: #Hours
        print "Updating TkStatus with query results for time interval %s to %s"%interval
        for row in QueryResults[interval]:
            TkStatus.updateO2OQuery(row)
            print len(TkStatus.PSUChannelHistory)
if counter-hours>0:
    print "Number of intervals skipped %s"%(counter-hours)
#Dump the PSUChannelHistory dictionary!
#TkHistoryPickle=open("TkPSUChannelHistory.pkl","wb")
#pickle.dump(TkStatus.PSUChannelHistory,TkHistoryPickle)
#TkHistoryPickle.close()
#
#for timestamp in sorted(TkStatus.PSUChannelHistory.keys()):
#    if len(TkStatus.PSUChannelHistory[timestamp])<4000:
#        print len(TkStatus.PSUChannelHistory[timestamp]),timestamp                     

#Test getters as I implement them:
#ChannelsOFF,ChannelsON,Start,Stop=TkStatus.getPSUChannelsOff(datetime.datetime(2010,8,27,10,38,26,700000))
#print "IOV: %s -> %s"%(Start,Stop)
#print "Number of PSU channels reported OFF: %s"%len(ChannelsOFF)
#print "Number of PSU channels reported ON: %s"%len(ChannelsON)
#DetIDsHVOFF,DetIDsHVON,Start,Stop=TkStatus.getDetIDsHVOff(datetime.datetime(2010,8,27,10,38,26,700000))
#print "IOV: %s -> %s"%(Start,Stop)
#print "Number of detID reported with HV OFF: %s"%len(DetIDsHVOFF)
#print "Number of detID reported with HV ON: %s"%len(DetIDsHVON)
#DetIDsLVOFF,DetIDsLVON,Start,Stop=TkStatus.getDetIDsLVOff(datetime.datetime(2010,8,27,10,38,26,700000))
#print "IOV: %s -> %s"%(Start,Stop)
#print "Number of detID reported with LV OFF: %s"%len(DetIDsLVOFF)
#print "Number of detID reported LV ON: %s"%len(DetIDsLVON)
#AliasesHVOFF,AliasesHVON,Start,Stop=TkStatus.getAliasesHVOff(datetime.datetime(2010,8,27,10,38,26,700000))
#print "IOV: %s -> %s"%(Start,Stop)
#print "Number of aliases reported with HV OFF: %s"%len(AliasesHVOFF)
#print "Number of aliases reported HV ON: %s"%len(AliasesHVON)
#AliasesLVOFF,AliasesLVON,Start,Stop=TkStatus.getAliasesLVOff(datetime.datetime(2010,8,27,10,38,26,700000))
#print "IOV: %s -> %s"%(Start,Stop)
#print "Number of aliases reported with LV OFF: %s"%len(AliasesLVOFF)
#print "Number of aliases reported LV ON: %s"%len(AliasesLVON)

#Check the number of channels off in the Run144086 IOV:
#DetIDsHVOFF,DetIDsHVON,Start,Stop=TkStatus.getDetIDsHVOff(datetime.datetime(2010,8,29,8,0,0,0))
#print "IOV: %s -> %s"%(Start,Stop)
#print "Number of detID reported with HV OFF: %s"%len(DetIDsHVOFF)
#print "Number of detID reported with HV ON: %s"%len(DetIDsHVON)
#DetIDsLVOFF,DetIDsLVON,Start,Stop=TkStatus.getDetIDsLVOff(datetime.datetime(2010,8,29,8,0,0,0))
#print "IOV: %s -> %s"%(Start,Stop)
#print "Number of detID reported with LV OFF: %s"%len(DetIDsLVOFF)
#print "Number of detID reported LV ON: %s"%len(DetIDsLVON)

#Now implement the following test:
#-Read in all the logs (either the DetVOffReaderDebug or the ones produced by the MakeTkMaps
#-Extract the IOV from the filename
#-Get the corresponding IOVs list from the PSUChannelHistory
#-Print out the number of DetIDs with LV ON/OFF (and of crosstalking and unmapped detids separately)
#-DetID comparison for each IOV (picking the stable one in PSUChannelHistory)
#-Implement the reduction as a method of TrkVoltageStatus that takes as argument deltaTmin and  MaxIOVlength
#and returns the lists of HV and LV channels off (IOVs are for any change of either HV or LV PSU channels)
#-Check the reduction result (number of IOVs, suppression during ramp up/down, ...)
#-DetID comparison for each IOV
#-Implement the direct query check of the query results and report anomalies (old results at the interface of the 1-hour queries:
#   -Get the query dumped by Coral by turning on verbosity and off
#   -Pick the query from the code too
#-Mapping part, integrate results of dictionaries from HV mapping code from Ben, dump all relevant dictionaries
#Cross-check stuff with cabling info?
#Check time changes and effect DST etc timezone issues and plotting limitations...
#Add number of ramp up/downs plots minima/maxima of HV/LV OFF/ON
#Look out for individual channels going down (trips) instead of full ramping
#Talk to Online guys about cross-talking and HV mapping
#Look into the query with the turn off commands and into giving the map for the new table to Robert
#Talk with Frank about extra info needed
#Look into Pixel integration extras


## #This next step is REALLY SLOW if there are thousands of IOVs to handle!
## 
## #Turn debug off to avoid a lot of printouts from getIOV:
## TkStatus.debug=False
## DetIDsHVOffVsTime,LastHVTimeStamp=TkStatus.getDetIDsHVOffVsTime()
## TkStatus.debug=True
## print "There are %s timestamps for DetIDs HV Off. The last timestamp (end of last IOV) is %s"%(len(DetIDsHVOffVsTime),LastHVTimeStamp)
## 
## #Turn debug off to avoid a lot of printouts from getIOV:
## TkStatus.debug=False
## DetIDsLVOffVsTime,LastLVTimeStamp=TkStatus.getDetIDsLVOffVsTime()
## TkStatus.debug=True
## print "There are %s timestamps for DetIDs LV Off. The last timestamp (end of last IOV) is %s"%(len(DetIDsLVOffVsTime),LastLVTimeStamp)
## 
## #Prepare the data for LV/HV OFF Graph plotting with PyROOT:
## import array, time
## 
## #Initialize the lists of timestamps (to be doubled to get the histogram/IOV look)
## LVOFFTimestamps=[]
## NumberOfLVOFFChannels=[]
## #Loop over the list of tuples with (timestamp,numberofdetidswithLVOFF):
## for item in sorted(DetIDsLVOffVsTime):
##     #FIXME:
##     #NEED to check the effect/use of tzinfo to avoid issues with UTC/localtime DST etc.
##     #Will have to massage the datetime.datetime object in a proper way not necessary now.
##     #Also need to decide if we want add the milliseconds into the "timestamps for plotting in root... not straightforward so will implement only if needed. For now know that the approximation is by truncation for now!
##     timestamp=int(time.mktime(item[0].timetuple())) #Need to get first a time tuple, then translate it into a unix timestamp (seconds since epoc). This means no resolution under 1 second by default
##     if item==DetIDsLVOffVsTime[0]: #First item does not need duplication
##         LVOFFTimestamps.append(timestamp)
##         LVOff=item[1]
##         NumberOfLVOFFChannels.append(LVOff)
##     else: #Add twice each timestamp except the first one (to make a histogram looking graph)
##         LVOFFTimestamps.append(timestamp)
##         NumberOfLVOFFChannels.append(LVOff) #Input previous value with new timestamp
##         LVOff=item[1]
##         LVOFFTimestamps.append(timestamp)
##         NumberOfLVOFFChannels.append(LVOff) #Input new value with new timestamp
##         
## LVOFFTimestampsArray=array.array('i',LVOFFTimestamps)
## NumberOfLVOFFChannelsArray=array.array('i',NumberOfLVOFFChannels)
## 
## #Initialize the lists of timestamps (to be doubled to get the histogram/IOV look)
## HVOFFTimestamps=[]
## NumberOfHVOFFChannels=[]
## #Loop over the list of tuples with (timestamp,numberofdetidswithHVOFF):
## for item in sorted(DetIDsHVOffVsTime):
##     timestamp=int(time.mktime(item[0].timetuple()))
##     if item==DetIDsHVOffVsTime[0]: #First item does not need duplication
##         HVOFFTimestamps.append(timestamp)
##         HVOff=item[1]
##         NumberOfHVOFFChannels.append(HVOff)
##     else: #Add twice each timestamp except the first one (to make a histogram looking graph)
##         HVOFFTimestamps.append(timestamp)
##         NumberOfHVOFFChannels.append(HVOff) #Input previous value with new timestamp
##         HVOff=item[1]
##         HVOFFTimestamps.append(timestamp)
##         NumberOfHVOFFChannels.append(HVOff) #Input new value with new timestamp
##         
## HVOFFTimestampsArray=array.array('i',HVOFFTimestamps) #NEED TO USE DOUBLES if we want the microseconds!!!! or it screws up approximating the timestamp
## NumberOfHVOFFChannelsArray=array.array('i',NumberOfHVOFFChannels)
## 
## #Testing the plotting function
## TkStatus.plotGraphSeconds(LVOFFTimestampsArray,NumberOfLVOFFChannelsArray,GraphTitle="Modules with LV OFF",YTitle="Number of modules with LV OFF",GraphFilename="Test2.gif")
## #TkStatus.plotGraphSeconds(LVOFFTimestampsArray,NumberOfLVOFFChannelsArray,"Modules with LV OFF","Number of modules with LV OFF","Test2.gif")

#Develop the Validation based on the TkStatus object on one side and the actual output of the O2O reader (CheckAllIOVs) in the current dir:

#FIXME:
#Issues to look into:
#1-first IOV not starting from beginning of the query: that should always be the case!
#2-Reduction seems to not be doing the right thing...
#3-timestamp timezone translation...

#Get the ReducedIOVsTimestamps (using the TkVoltage.getReducedIOVs() using the same deltaT=15s, and maxIOVSequenceLength=120s):
ReducedIOVsTimestamps=TkStatus.getReducedIOVs(datetime.datetime(2010, 8, 27, 12, 0),datetime.datetime(2010, 8, 29, 17, 45),TkStatus.PSUChannelHistory,15,120)
#Print out for debugging the reduced timestamps with their index (to see when a sequence is present) and the number of LV/HV channels OFF for the given timestamp in the TkStatus object!
print "Dumping ReducedIOVsTimestamps and the corresponding timestamp index and number of HV/LV channels off:"
for timestamp in ReducedIOVsTimestamps:
    TkStatus.debug=False
    print timestamp,sorted(TkStatus.PSUChannelHistory.keys()).index(timestamp),len(TkStatus.getDetIDsHVOff(timestamp)[0]), len(TkStatus.getDetIDsLVOff(timestamp)[0])

#Following function will be moved inside the TkVoltageStatus class once it's perfected:
def ReducedIOVsHistory(ReducedIOVsTimestamps):
    """
    Function that given a list of reduced IOVs timestamps (output of getReducedIOVs()), analyses the timestamps to identify the IOV sequences and treats them differently when ramping-up or ramping-down, to return a dictionary that has timestamp as a key and the list of LV and HV channels OFF, that can be compared with the content of the CheckAllIOVs output DetVOffReaderDebug logs for validation.
    """
    
    AllTimestamps=sorted(TkStatus.PSUChannelHistory.keys())
    PreviousTimestampIndex=-1
    ReducedIOVsDict={}
    for timestamp in ReducedIOVsTimestamps:
        if AllTimestamps.index(timestamp)!=(PreviousTimestampIndex+1): #Sequence end!
            #Get the current timestamp LVOff channels:
            LVOffEnd=TkStatus.getDetIDsLVOff(timestamp)[0]
            #and the LV Off ones for the previous timestamp (beginning of the sequence):
            LVOffStart=TkStatus.getDetIDsLVOff(AllTimestamps[PreviousTimestampIndex])[0]
            #Get the current timestamp HVOff channels:
            HVOffEnd=TkStatus.getDetIDsHVOff(timestamp)[0]
            #and the HV Off ones for the previous timestamp (beginning of the sequence):
            HVOffStart=TkStatus.getDetIDsHVOff(AllTimestamps[PreviousTimestampIndex])[0]
            #This can be complicated... let's go for the same approach as the official O2O reduction
            #We can just test if the other conditions not captured by the ifs ever happen!
            #Turning OFF case:
            if len(LVOffEnd)>len(LVOffStart) or len(HVOffEnd)>len(HVOffStart): 
                ReducedIOVsDict.update({AllTimestamps[PreviousTimestampIndex]:(TkStatus.getDetIDsHVOff(timestamp)[0],TkStatus.getDetIDsLVOff(timestamp)[0])}) #use the LVOff/HVOff form the last element in the sequence and set the first element of the sequence to it!
            #Turning (Staying) ON case:
            #Nothing special to do (same as not a sequence)! We're happy with having thrown away all the intermediate timestamps, and keep the validity of the first timestamp of the sequence throughout the sequence:
        #For all timestamps reported (if they are a start of a sequence they will be "overwritten" once we process the end of the sequence timestamp) in particular also the end of a ramp-up sequence does not need no special treatement:
        #Actually check if the LV Off or HVOff are the same as the previous timestamp: if they are do nothing, if they are not then add an IOV...
        if set(TkStatus.getDetIDsHVOff(timestamp)[0])!=set(TkStatus.getDetIDsHVOff(AllTimestamps[PreviousTimestampIndex])[0]) or set(TkStatus.getDetIDsLVOff(timestamp)[0])!=set(TkStatus.getDetIDsLVOff(AllTimestamps[PreviousTimestampIndex])[0]):
            ReducedIOVsDict.update({timestamp:(TkStatus.getDetIDsHVOff(timestamp)[0],TkStatus.getDetIDsLVOff(timestamp)[0])})
        PreviousTimestampIndex=AllTimestamps.index(timestamp)
    return ReducedIOVsDict

#Now using the ReducedIOVs timestamps we can get the actual ReducedIOVs using the same algorithm as the O2O (implemented in the function ReducedIOVsHistory):
ValidationReducedIOVsHistory=ReducedIOVsHistory(ReducedIOVsTimestamps)
#Print out for debugging the timestamp, the number of HV/LV channels off from the TkStatus object for each of the REDUCED timestamps (AGAIN)
#for timestamp in ReducedIOVsTimestamps:
#    print timestamp,len(TkStatus.getDetIDsHVOff(timestamp)[0]),len(TkStatus.getDetIDsLVOff(timestamp)[0])

#Print out for debugging the timestamp, the number of HV/LV channels OFF from the ValidationReducedIOVsHistory object directly!
#i=0
print "Dumping ValidationReducedIOVsHistory contents:"
for timestamp in sorted(ValidationReducedIOVsHistory.keys()):
    print timestamp, len(ValidationReducedIOVsHistory[timestamp][0]),len(ValidationReducedIOVsHistory[timestamp][1])#,sorted(O2OReducedIOVs.keys())[i],len(O2OReducedIOVs[sorted(O2OReducedIOVs.keys())[i]][0]),len(O2OReducedIOVs[sorted(O2OReducedIOVs.keys())[i]][1])
#    i=i+1
    
#for i in range(42):
#    print sorted(ValidationReducedIOVsHistory.keys())[i],sorted(O2OReducedIOVs.keys())[i]

#Now extract the DetVOffInfo from the logfiles directly and then we can compare!

#Cut&Paste of a quick and dirty script that reads the CheckAllIOVS.py output (ReaderDebug) logs and produces an O2OData dictionary
#that has timestamps (from filename) as keys and a tuple (HVOff,LVOff) as value, where HVOff and LVOff are lists of detids that have HV/LV off respectively.
#The union of the two would be the total number of detids listed as OFF (in either way, OFF-OFF or OFF-ON)
#Can implement a direct IOV comparison using the reduction above (need to be careful on the matching of IOVs)
#import datetime
def ExtractDetVOffInfo(directory=os.getcwd()):
    """
    Function that given a directory (defaults to the local one in case no dir indicated), parses all local DetVOffReaderDebug*.log files and returna a dictionary with timestamps for keys and a tuple with the list of LV and HV channels OFF (LVOff,HVOff). 
    """
    ls=os.listdir(directory)
    TimesLogs=[]
    O2OData={}
    for log in ls:
        if "DetVOffReaderDebug__FROM" in log:
            (start,end)=log[:-4].split("FROM_")[1].split("_TO_")
            TimeStamp=datetime.datetime.strptime(start.replace("__","_0"),"%a_%b_%d_%H_%M_%S_%Y")
            #print start,TimeStamp
            file=open(log,'r')
            filelines=file.readlines()
            file.close()
            LVOff=[]
            HVOff=[]
            for line in filelines:
                #print line
                if "OFF" in line:
                    detid,hv,lv=line.split()
                    #print line,detid,hv,lv
                    if hv=="OFF":
                        HVOff.append(int(detid))
                    if lv=="OFF":
                        LVOff.append(int(detid))
                        
                    O2OData.update({TimeStamp:(HVOff,LVOff)})
    return O2OData

#Extract the O2O Reduced IOVs data from the logfiles in the current directory 
O2OReducedIOVs=ExtractDetVOffInfo()
#Print out for debugging the timestamp, the number of HV/LV  channels OFF reported by the O2O
print "Dumping the O2OReducedIOVs contents:"
for timestamp in sorted(O2OReducedIOVs.keys()):
    print timestamp, len(O2OReducedIOVs[timestamp][0]),len(O2OReducedIOVs[timestamp][1])#,len(TkStatus.getDetIDsHVOff(TkStatus.getIOV(timestamp,TkStatus.PSUChannelHistory)[0])[0]), len(TkStatus.getDetIDsLVOff(TkStatus.getIOV(timestamp,TkStatus.PSUChannelHistory)[0])[0])
    #Compare the actual detids after doing the reduction the way we want to do it!
    
# len(TkStatus.getDetIDsHVOff(sorted(TkStatus.PSUChannelHistory.keys())[-1])[0])


#Now compare the reduced histories:
def CompareReducedDetIDs(FirstDict,SecondDict):
    """
    Function that given 2 Dictionaries (key=timestamp, value=(LVOff,HVOff)) loops through the first one and compared the content of its IOVs with the IOV in the other dict that overlaps with it.
    """
    DifferenceDict={}
    for timestamp in sorted(FirstDict.keys()):
        if timestamp.replace(microsecond=0) in SecondDict.keys():
            secondtimestamp=timestamp.replace(microsecond=0)
            print "Timestamp %s is present in both Dictionaries!"%timestamp
        else:
            secondtimestamps=sorted(SecondDict.keys())
            secondtimestamps.append(timestamp)
            secondtimestamps.sort()
            if secondtimestamps.index(timestamp)!=0:#To avoid wrapping up to the end of the list of timestamps!!!
                secondtimestamp=secondtimestamps[secondtimestamps.index(timestamp)-1]
            else:#Default to the earliest timestamp in the second dictionary...
                secondtimestamp=secondtimestamps[secondtimestamps.index(timestamp)+1]
            print "Comparing the IOV with timestamp %s (1st dict) with IOV with timestamp %s (2nd dict)"%(timestamp,secondtimestamp) 
        if set(map(lambda x:int(x),FirstDict[timestamp][0]))!=set(map(lambda x:int(x),SecondDict[secondtimestamp][0])) or set(map(lambda x:int(x),FirstDict[timestamp][1]))!=set(map(lambda x:int(x),SecondDict[secondtimestamp][1])): #Change!
            if len(set(FirstDict[timestamp][0]))<=len(set(SecondDict[secondtimestamp][0])):
                differenceHV=set(map(lambda x:int(x),SecondDict[secondtimestamp][0]))-set(map(lambda x:int(x),FirstDict[timestamp][0]))
            else:
            #elif len(set(SecondDict[secondtimestamp][0]))<len(set(FirstDict[timestamp][0])):
                differenceHV=set(map(lambda x:int(x),FirstDict[timestamp][0]))-set(map(lambda x:int(x),SecondDict[secondtimestamp][0]))
            #else:
            #    print "SCREAM! Something weird going on one of the two should be a subset of the other!"
            #    differenceLV=set([])
            #    differenceHV=set([])
            if len(set(FirstDict[timestamp][1]))<=len(set(SecondDict[secondtimestamp][1])):
                differenceLV=set(map(lambda x:int(x),SecondDict[secondtimestamp][1]))-set(map(lambda x:int(x),FirstDict[timestamp][1]))
            else:
            #elif set(SecondDict[secondtimestamp][1]).issubset(set(FirstDict[timestamp][1])):
                differenceLV=set(map(lambda x:int(x),FirstDict[timestamp][1]))-set(map(lambda x:int(x),SecondDict[secondtimestamp][1]))
            #else:
            #    print "SCREAM! Something weird going on one of the two should be a subset of the other!"
            #    differenceLV=set([])
            #    differenceHV=set([])
            DifferenceDict.update({(timestamp,secondtimestamp):(differenceHV,differenceLV)})
            print "Difference in timestamp %s (corresponding to %s):"%(timestamp,secondtimestamp)
            #print "LV OFF:"
            #for LVChannel in differenceLV:
            #    print LVChannel
            #print "HV OFF:"
            #for HVChannel in differenceHV:
            #    print HVChannel
        else:
            print "Timestamp %s is identical in both dictionaries"%timestamp
    return DifferenceDict
Comparison=CompareReducedDetIDs(ValidationReducedIOVsHistory,O2OReducedIOVs)
print "Dumping the results of the comparisons of the two dictionaries:"
for timestamps in sorted(Comparison.keys()):
    print timestamps, Comparison[timestamps]
    if Comparison[timestamps][0]:
        print "HV:"
        if Comparison[timestamps][0].issubset(set(O2OReducedIOVs[timestamps[1]][0])):
            print "Only in O2O Dict!"
        else:
            print "Only in Validation Dict!"
        for detid in Comparison[timestamps][0]:
            print detid,TkStatus.DetIDAliasDict[detid]
    if Comparison[timestamps][1]:
        print "LV:"
        if Comparison[timestamps][1].issubset(set(O2OReducedIOVs[timestamps[1]][1])):
            print "Only in O2O Dict!"
        else:
            print "Only in Validation Dict!"
        for detid in Comparison[timestamps][1]:
            print detid,TkStatus.DetIDAliasDict[detid]


#Add a check with the query using sqlalchemy
#Add a check for channels that never show updates in the queries (even when several ramp cycles happened)
#Add a check that inside the sequence change is consistently up or down

#Check of the direct DB query:
#Need to first run another python script (using the local python of cmstrko2ovm02 that has sqlalchemy, cx-oracle etc) that dumps a pkl with a list of the result rows of the O2O query....

TkStatusFromQuery=TrkVoltageStatus(detIDPSUmapfilename=os.path.join(os.getenv('CMSSW_BASE'),'src/CalibTracker/SiStripDCS/data/StripPSUDetIDMap_FromJan132010_Crosstalk.dat'),DetIDAliasFile=os.path.join(os.getenv('CMSSW_BASE'),'src/CalibTracker/SiStripDCS/data/StripDetIDAlias.pkl'),startTime=datetime.datetime(2010,8,27,10,00,00),debug=True)
DBQueryPickle=open('/afs/cern.ch/user/g/gbenelli/scratch0/O2OValidation/QueryResults.pkl','rb')
DBQueryResults=[]
DBQueryResults=pickle.load(DBQueryPickle)
DBQueryPickle.close()
for row in DBQueryResults:
    TkStatusFromQuery.updateO2OQuery(row)
    print len(TkStatusFromQuery.PSUChannelHistory)

delta=datetime.timedelta(seconds=7200)
counter=0
DifferingTimestamps=[]
for timestamp in sorted(TkStatus.PSUChannelHistory.keys()):
    if timestamp-delta!=sorted(TkStatusFromQuery.PSUChannelHistory.keys())[counter]:
        print timestamp, sorted(TkStatusFromQuery.PSUChannelHistory.keys())[counter]
        DifferingTimestamps.append(timestamp,sorted(TkStatusFromQuery.PSUChannelHistory.keys())[counter])
    counter=counter+1
if DifferingTimestamps:
    print "There are %s timestamps that are different in the 2 TkVoltageStatus objects!"%len(DifferingTimestamps)

#Test issue with channel000 and channel001 lagging:
#LVChannelsHistory={}
#for timestamp in sorted(TkStatus.PSUChannelHistory.keys()):
#    LVOFF=[psuchannel for psuchannel in TkStatus.getPSUChannelsOff(timestamp) if (psuchannel.endswith('0') or psuchannel.endswith('1'))]
#    for channel in LVOFF:
LVChannelHistory={}
HVChannelHistory={}
for interval in QueryResults.keys():
    for row in QueryResults[interval]:
        if row['dpname'].endswith('0') or row['dpname'].endswith('1'):
            if row['dpname'][:-1] not in LVChannelHistory.keys():
                LVChannelHistory.update({row['dpname'][:-1]:[(row['change_date'],row['actual_status'])]})
            else:
                LVChannelHistory[row['dpname'][:-1]].append((row['change_date'],row['actual_status']))
        else:
            if row['dpname'] not in HVChannelHistory.keys():
                HVChannelHistory.update({row['dpname']:[(row['change_date'],row['actual_status'])]})
            else:
                HVChannelHistory[row['dpname']].append((row['change_date'],row['actual_status']))
        
        #if row['change_date']==datetime.datetime(2010, 8, 28, 22, 52, 8, 994000):
        #    print row
        #if row['change_date']==datetime.datetime(2010, 8, 28, 22, 51, 8, 994000):
        #    print row
        if row['change_date']>datetime.datetime(2010, 8, 28, 22, 44) and row['change_date']<datetime.datetime(2010, 8, 28, 23, 5,37):
            print row    
            
        

for row in sorted(HVChannelHistory['cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_6/branchController02/easyCrate3/easyBoard15/channel002']):
    print row[0],row[1]
for row in sorted(HVChannelHistory['cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_6/branchController02/easyCrate3/easyBoard15/channel003']):
    print row[0],row[1]
print TkStatus.PSUChannelHistory[sorted(TkStatus.PSUChannelHistory.keys())[sorted(TkStatus.PSUChannelHistory.keys()).index(datetime.datetime(2010, 8, 28, 22, 52, 8, 994000))-1]]
a=TkStatus.getIOV(datetime.datetime(2010, 8, 28, 22, 51, 16),TkStatus.PSUChannelHistory)

#PLOTTTING!
import array, time
TOBminus_1_3_1_4_HV1_Status=HVChannelHistory['cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_6/branchController02/easyCrate3/easyBoard15/channel002']
#Select time range via:
TOBminus_1_3_1_4_HV1_Status=[item for item in HVChannelHistory['cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_6/branchController02/easyCrate3/easyBoard15/channel002'] if item[0]>datetime.datetime(2010,8,28,22) and item[0]<datetime.datetime(2010,8,29,2)]
TOBTimestamps=[]
TOBHVStatus=[]
for item in TOBminus_1_3_1_4_HV1_Status:
    timestamp=int(time.mktime(item[0].timetuple()))
    if item==TOBminus_1_3_1_4_HV1_Status[0]: #First item does not need duplication
        TOBTimestamps.append(timestamp)
        HVStatus=int(item[1]==1)
        TOBHVStatus.append(HVStatus)
    else: #Add twice each timestamp except the first one (to make a histogram looking graph)
        TOBTimestamps.append(timestamp)
        TOBHVStatus.append(HVStatus) #Input previous value with new timestamp
        HVStatus=int(item[1]==1)
        TOBTimestamps.append(timestamp)
        TOBHVStatus.append(HVStatus) #Input new value with new timestamp


#TOBTimestamps=map(lambda x: int(time.mktime(x[0].timetuple())),TOBminus_1_3_1_4_HV1_Status)
#TOBHVStatus=map(lambda y: int(y[1]==1),TOBminus_1_3_1_4_HV1_Status)
##print "FIRST the original arrays:"
###Duplication of timestamps...
##for timestamp in [item[0] for item in TOBminus_1_3_1_4_HV1_Status]:
##    print timestamp,TOBHVStatus[TOBTimestamps.index(timestamp)]
##    TOBTimestamps.insert(TOBTimestamps.index(timestamp)+1,timestamp)
##    TOBHVStatus.insert(TOBTimestamps.index(timestamp)+1,TOBHVStatus[TOBTimestamps.index(timestamp)-1])
##print 'NOW "duplicated" arrays'
##for timestamp in TOBTimestamps:
##    print timestamp, TOBHVStatus[TOBTimestamps.index(timestamp)]
##        
TOBTimestampsArray=array.array('i',TOBTimestamps)
TOBHVStatusArray=array.array('i',TOBHVStatus)

TkStatus.plotGraphSeconds(TOBTimestampsArray,TOBHVStatusArray,GraphTitle="PSUChannelGraph",YTitle="Channel HV Status",GraphFilename="PSUChannel.gif")
TOBminus_1_3_1_4_HV1_Status=HVChannelHistory['cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_6/branchController02/easyCrate3/easyBoard15/channel003']
#Select time range via:
TOBminus_1_3_1_4_HV1_Status=[item for item in HVChannelHistory['cms_trk_dcs_04:CAEN/CMS_TRACKER_SY1527_6/branchController02/easyCrate3/easyBoard15/channel003'] if item[0]>datetime.datetime(2010,8,28,22) and item[0]<datetime.datetime(2010,8,29,2)]
TOBTimestamps=[]
TOBHVStatus=[]
for item in TOBminus_1_3_1_4_HV1_Status:
    timestamp=int(time.mktime(item[0].timetuple()))
    if item==TOBminus_1_3_1_4_HV1_Status[0]: #First item does not need duplication
        TOBTimestamps.append(timestamp)
        HVStatus=int(item[1]==1)
        TOBHVStatus.append(HVStatus)
    else: #Add twice each timestamp except the first one (to make a histogram looking graph)
        TOBTimestamps.append(timestamp)
        TOBHVStatus.append(HVStatus) #Input previous value with new timestamp
        HVStatus=int(item[1]==1)
        TOBTimestamps.append(timestamp)
        TOBHVStatus.append(HVStatus) #Input new value with new timestamp


#TOBTimestamps=map(lambda x: int(time.mktime(x[0].timetuple())),TOBminus_1_3_1_4_HV1_Status)
#TOBHVStatus=map(lambda y: int(y[1]==1),TOBminus_1_3_1_4_HV1_Status)
##print "FIRST the original arrays:"
###Duplication of timestamps...
##for timestamp in [item[0] for item in TOBminus_1_3_1_4_HV1_Status]:
##    print timestamp,TOBHVStatus[TOBTimestamps.index(timestamp)]
##    TOBTimestamps.insert(TOBTimestamps.index(timestamp)+1,timestamp)
##    TOBHVStatus.insert(TOBTimestamps.index(timestamp)+1,TOBHVStatus[TOBTimestamps.index(timestamp)-1])
##print 'NOW "duplicated" arrays'
##for timestamp in TOBTimestamps:
##    print timestamp, TOBHVStatus[TOBTimestamps.index(timestamp)]
##        
TOBTimestampsArray=array.array('i',TOBTimestamps)
TOBHVStatusArray=array.array('i',TOBHVStatus)

TkStatus.plotGraphSeconds(TOBTimestampsArray,TOBHVStatusArray,GraphTitle="PSUChannelGraph",YTitle="Channel HV Status",GraphFilename="PSUChannelHV2.gif")
def plotPSUChannelvsTime(self,TimeArray,ValuesArray,GraphTitle="PSUChannelGraph",YTitle="Channel HV Status",GraphFilename="PSUChannel.gif"):
    """
    Function that given an array with timestamps (massaged to introduce a second timestamp for each value to produce an histogram-looking plot) and a corresponding array with values to be plotted, a title, a Y axis title and a plot filename, produces with pyROOT a time plot and saves it locally.
    The function can be used for cumulative plots (number of channels with HV/LV OFF/ON vs time) or for individual (single detID HV/LV status vs time) plots.
    """
    import ROOT
    canvas=ROOT.TCanvas()
    graph=ROOT.TGraph(len(TimeArray),TimeArray,ValuesArray)
    graph.GetXaxis().SetTimeDisplay(1)
    graph.GetXaxis().SetLabelOffset(0.02)
    #Set the time format for the X Axis labels depending on the total time interval of the plot!
    TotalTimeIntervalSecs=TimeArray[-1]-TimeArray[0]
    if TotalTimeIntervalSecs <= 120: #When zooming into less than 2 mins total interval report seconds too
        graph.GetXaxis().SetTimeFormat("#splitline{   %d/%m}{%H:%M:%S}")
    elif 120 < TotalTimeIntervalSecs < 6400: #When zooming into less than 2 hrs total interval report minutes too
        graph.GetXaxis().SetTimeFormat("#splitline{%d/%m}{%H:%M}")
    else: #When plotting more than 2 hrs only report the date and hour of day
        graph.GetXaxis().SetTimeFormat("#splitline{%d/%m}{  %H}")
    graph.GetYaxis().SetTitle(YTitle)
    graph.GetYaxis().SetTitleOffset(1.4)
    graph.SetTitle(GraphTitle)
    graph.Draw("APL")
    canvas.SaveAs(GraphFilename)
    print "Saved graph as %s"%GraphFilename
    return


ReducedIOVsTimestampsTEST=TkStatus.getReducedIOVs(datetime.datetime(2010, 8, 27, 12, 0),datetime.datetime(2010, 8, 29, 17, 45),TkStatus.PSUChannelHistory,2,90)
#Print out for debugging the reduced timestamps with their index (to see when a sequence is present) and the number of LV/HV channels OFF for the given timestamp in the TkStatus object!
print "Dumping ReducedIOVsTimestamps and the corresponding timestamp index and number of HV/LV channels off:"
for timestamp in ReducedIOVsTimestampsTEST:
    TkStatus.debug=False
    print timestamp,sorted(TkStatus.PSUChannelHistory.keys()).index(timestamp),len(TkStatus.getDetIDsHVOff(timestamp)[0]), len(TkStatus.getDetIDsLVOff(timestamp)[0])
ValidationReducedIOVsHistoryTEST=ReducedIOVsHistory(ReducedIOVsTimestampsTEST)
print "Dumping ValidationReducedIOVsHistory contents:"
for timestamp in sorted(ValidationReducedIOVsHistoryTEST.keys()):
    print timestamp, len(ValidationReducedIOVsHistoryTEST[timestamp][0]),len(ValidationReducedIOVsHistoryTEST[timestamp][1])#,sorted(O2OReducedIOVs.keys())[i],len(O2OReducedIOVs[sorted(O2OReducedIOVs.keys())[i]][0]),len(O2OR
