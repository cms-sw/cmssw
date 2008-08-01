#!/usr/bin/env python

'''
Class access the RunSummary web page for a given list of runs, and stores a list of valid runs (i.e., runs meeting user-specified criteria) in that range.  For valid HCAL runs, we simply require that the HCAL was present in the readout of the runs.
'''

import sys
import string
# Url Libraries needed for accessing/parsing html results
import urllib
import urllib2


# class parseResults stores web info for each run, and parses it to search for all run-related info (run #, start, end time, components, etc.)

class parseResult:
    '''
    Takes output from html search, and tries to identify whether HCAL was in the run.
    '''

    def __init__(self,text,debug=False):
        '''
        This is all stuff for parsing the html results
        For Hcal, all we care about is that:
        a) self.run is a valid integer
        b) self.events is a non-zero integer
        c) self.components contains "HCAL"
        However, we fill the class with all the information provided with the expectation that we may want to use more info in our sorting later (such as requiring the B field to be above a certain value, or requiring a specific trigger).
         
        A valid list of results (text) will contain 12 lines:  <blank line>, Run, Sequence, Booking Time, Key, Start Time, End Time, Triggers, Events, B Field, Components, and a </TR> row end.
        '''

        if (debug):
            print self.__init__.__doc__
            
        self.isvalid=False # determines whether run is valid or not (just tests whether text info provided can be properly parsed)
        self.isvalidHcal=False # Boolean describing whether run is valid for Hcal
        self.debug=debug
        
        self.events=None
        self.components=None
        self.run=None
        # values below aren't used yet for determining Hcal validity, but they could be
        self.sequence=None
        self.bookingtime=None
        self.key=None
        self.starttime=None
        self.endtime=None
        self.triggers=None
        self.bfield=None
        
        text=string.replace(text,"&nbsp;","") # get rid of html space format
        self.text=string.split(text,"\n")[1:] # ignore initial blank line

        # Should really use regular expressions for parsing at some point
        try:
            self.run=self.text[0]
            self.run=string.split(self.run,"</A>")[0]
            self.run=string.split(self.run,">")[2]
            self.run=string.atoi(self.run)
        except:
            if (self.debug):
                print "Could not determine run info from %s"%self.text[0]
            return

        try:
            self.sequence = self.text[1]
            self.sequence = string.split(self.sequence,"</TD>")[0]
            self.sequence = string.split(self.sequence,"<TD>")[1]
        except:
            if (self.debug):
                print "Could not determine sequence from %s"%self.text[1]
            return

        try:
            self.bookingtime = self.text[2]
            self.bookingtime = string.split(self.bookingtime,"</TD>")[0]
            self.bookingtime = string.split(self.bookingtime,"<TD>")[1]
        except:
            if (self.debug):
                print "Could not determine booking time from %s"%self.text[2]
            return
        
        try:
            self.key = self.text[3]
            self.key = string.split(self.key,"</TD>")[0]
            self.key = string.split(self.key,"<TD>")[1]
        except:
            if (self.debug):
                print "Could not determine key from %s"%self.text[3]
            return
        
        try:
            self.starttime = self.text[4]
            self.starttime = string.split(self.starttime,"</TD>")[0]
            self.starttime = string.split(self.starttime,"<TD>")[1]
        except:
            if (self.debug):
                print "Could not determine start time from %s"%self.text[4]
            return
        
        try:
            self.endtime = self.text[5]
            self.endtime = string.split(self.endtime,"</TD>")[0]
            self.endtime = string.split(self.endtime,"<TD>")[1]
        except:
            if (self.debug):
                print "Could not determine end time from %s"%self.text[5]
            return
        
        try:
            self.triggers = self.text[6]
            self.triggers = string.split(self.triggers,"</TD>")[0]
            self.triggers = string.split(self.triggers,">")[1]
        except:
            if (self.debug):
                print "Could not determine triggers from %s"%self.text[6]
            return
        
        try:
            self.events = self.text[7]
            self.events = string.split(self.events,"</TD>")[0]
            self.events = string.split(self.events,">")[1]
            # Make sure read integer value 
            if (self.events<>"null"):
                self.events=string.atoi(self.events)
        except:
            if (self.debug):
                print "Could not determine # of events from %s"%self.text[7]
            return
        
        try:
            self.bfield = self.text[8]
            self.bfield = string.split(self.bfield,"</TD>")[0]
            self.bfield = string.split(self.bfield,">")[1]
            if (self.bfield<>"null"):
                self.bfield=string.atof(self.bfield)
        except:
            if (self.debug):
                print "Could not determine B field from %s"%self.text[8]
            return
        
        try:
            self.components = self.text[9]
            self.components = string.split(self.components,"</TD>")[0]
            self.components = string.split(self.components,">")[1]
        except:
            if (self.debug):
                print "Could not determine components from %s"%self.text[9]
            return

        self.isvalid=True # able to read all event info

        # Some good runs have 'events' listed as null -- don't use events as a test of goodness?
        if (self.components<>None and string.find(self.components,"HCAL")>-1):
            self.isvalidHcal=True
        return

    def printParse(self):
        ''' printParse method prints result of parsing input string.\n\n '''
        if (self.debug):
            print self.printParse.__doc__
            
        print "Run # = ", self. run
        print "\tsequence = ",self.sequence
        print "\tbooking time = ",self.bookingtime
        print "\tkey = ",self.key
        print "\tstart time = ",self.starttime
        print "\tend time = ",self.endtime
        print "\ttriggers = ",self.triggers
        print "\tevents = ",self.events
        print "\tB field = ",self.bfield
        print "\tcomponents = ",self.components
        print "Is info valid? ",self.isvalid
        print "\n\n"
        return


class goodHCALList:
    '''
    goodHCALList class stores list of runs in range, and list of runs containing HCAL.
    '''

    def __init__(self, beginRun=43293, endRun=43493,debug=False):
        self.beginRun=beginRun
        self.endRun=endRun
        self.debug=debug
        if (self.debug):
            print "Initializing goodHCALList class"
            print "Looking for runs between %i - %i"%(self.beginRun, self.endRun)
        self.values={"RUN_BEGIN":self.beginRun,
                     "RUN_END":self.endRun}
        self.allruns=[]
        self.hcalruns=[]
        # Check for HCAL runs in range (beginRun-endRun)
        self.CheckHtml()
        return

    def CheckHtml(self):
        '''CheckHtml method searches http://cmsmon.cern.ch/cmsdb/servlet/RunSummary for runs in specified range, and store runs containing HCAL in a separate list.'''
        if (self.debug):
            print self.CheckHtml.__doc__

        # Run Summary Page URL
        myurl="http://cmsmon.cern.ch/cmsdb/servlet/RunSummary"

        # encode values as URL data, and post them to run summary page
        data=urllib.urlencode(self.values)
        req=urllib2.Request(myurl,data)


        # Get result of sending data to URL
        response=urllib2.urlopen(req)
        thepage=response.read()
        
        # Split result by <TR> (since result is a table of runs, if runs found)
        thepage=string.split(thepage,"<TR>")

        for i in thepage:
            temp = parseResult(i,self.debug)
            if (self.debug):
                temp.printParse()
            if temp.isvalid:
                self.allruns.append(temp.run)
                if temp.isvalidHcal:
                    self.hcalruns.append(temp.run)

        self.allruns.sort()
        self.hcalruns.sort()
        return

    def printRuns(self):
        ''' prints all runs found in user-specified range, and indicates if HCAL was present in each run. '''
        if (self.debug):
            print self.printRuns.__doc__

        print "\n%20s%20s"%("Run found","Includes HCAL")
        for i in self.allruns:
            if i in self.hcalruns:
                text="%20s%20s"%(`i`,"YES")
            else:
                text="%20s"%(`i`)
            print text
        return


######################################################

if __name__=="__main__":
    
    values={"RUN_BEGIN": 43293,
            "RUN_END": 43493}

    debug=False
    
    # allow user to input start, end run values on command line
    if len(sys.argv)>1:
        try:
            values["RUN_BEGIN"]=string.atoi(sys.argv[1])
        except:
            print "Could not recognize starting run number '%s' as an integer"%sys.argv[1]
            print "Continuing using default starting run number %i"%values["RUN_BEGIN"]

    if len(sys.argv)>2:
        try:
            values["RUN_END"]=string.atoi(sys.argv[2])
        except:
            print "Could not recognize ending run number '%s' as an integer"%sys.argv[2]
            print "Continuing using default ending run number %i"%values["RUN_END"]

    if len(sys.argv)>3:
        debug=True

    # Get list of good HCAL runs
    x=goodHCALList(values["RUN_BEGIN"],values["RUN_END"],debug)
    x.printRuns() # print out list
