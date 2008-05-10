#!/usr/bin/env python


import sys
import os # used for getting directory, checking if $CMSSW is set up

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

import python_dbs # this sends/receives messages to/from DBS
import string # use to parse DBS input/output
import cPickle # used for saving files

#############################################################################

class dbsAccessor:
    '''
    dbsAccessor:  Class that stores values which are used when
    accessing DBS web page.
    Values stored as IntVars, StringVars, etc. so that they can be
    easily utilized by the main DQMDBSgui.
    '''

    def __init__(self,basepath=os.curdir,debug=False,pclName=".dbsDefaults.cPickle"):
        '''
        dbsAccessor tries to read its values from a cPickle file.
        If the file does not exist, values will be initialized as defaults.
        IMPORTANT -- this class uses Tkinter StringVar, etc. variables,
        so this class will not properly initialize unless a Tk instance has
        been created first.
        '''


        
        self.basepath=basepath
        self.class_debug=debug
        self.pclName=pclName

        if (self.class_debug):
            print self.__init__.__doc__

        try:
            self.host=StringVar()
            self.port=IntVar()
            self.dbsInst=StringVar()
            self.searchStringFile=StringVar()
            self.searchStringDataset=StringVar()
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
        ** pydbsAccessor.setDefaults() **
        Sets defaults for dbsAccessor.  This is used when
        defaults cannot be read from cPickle file.
        '''

        if (self.class_debug):
            print self.setDefaults.__doc__
        
        self.searchResult=None
        self.host.set("cmsweb.cern.ch/dbs_discovery/")
        self.port.set(443)
        self.dbsInst.set("cms_dbs_prod_global")
        self.searchStringFile.set("*/Global*/A/*RAW/*.root")
        self.searchStringDataset.set("")
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

    def Print(self):
        x="host=%s\nport=%s\ndbsInst=%s\nsearchStringFile=\%s\nsearchStringDataset=%s\npage=%s\nlimit=%s\nxml=%s\ncase=%s\ndetails=%s\ndebug=%s\nbeginRun=%s\nendRun=%s\n"%(self.host.get(),self.port.get(),self.dbsInst.get(),self.searchStringFile.get(),self.searchStringDataset.get(),self.page.get(),self.limit.get(),self.xml.get(),self.case.get(),self.details.get(),self.debug.get(),self.beginRun.get(),self.endRun.get())
        if (self.class_debug):
            print x
        return x

    def getDefaultsFromPickle(self):
        '''
        ** pydbsAccessor.getDefaultsFromPickle() **
        Try to read default values of dbsAccessor from .dbsDefaults.cPickle.
        If unsuccessful, defaults will be initialized from "setDefaults"
        function.
        '''

        if (self.class_debug):
            print self.getDefaultsFromPickle.__doc__
        
        if os.path.isfile(os.path.join(self.basepath,self.pclName)):
            try:
                pcl=open(os.path.join(self.basepath,self.pclName),'rb')
                self.host.set(cPickle.load(pcl))
                self.port.set(cPickle.load(pcl))
                self.dbsInst.set(cPickle.load(pcl))
                self.searchStringFile.set(cPickle.load(pcl))
                self.searchStringDataset.set(cPickle.load(pcl))
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
        ** pydbsAccessor.writeDefaultsToPickle() **
        Writes default dbsAccessor values to "dbsDefaults.cPickle"
        '''

        if (self.class_debug):
            print self.writeDefaultsToPickle.__doc__
        
        try:
            pcl=open(os.path.join(self.basepath,self.pclName),'wb')
            cPickle.dump(self.host.get(),pcl)
            cPickle.dump(self.port.get(),pcl)
            cPickle.dump(self.dbsInst.get(),pcl)
            cPickle.dump(self.searchStringFile.get(),pcl)
            cPickle.dump(self.searchStringDataset.get(),pcl)
            cPickle.dump(self.page.get(),pcl)
            cPickle.dump(self.limit.get(),pcl)
            cPickle.dump(self.xml.get(),pcl)
            cPickle.dump(self.case.get(),pcl)
            cPickle.dump(self.details.get(),pcl)
            cPickle.dump(self.debug.get(),pcl)
            cPickle.dump(self.beginRun.get(),pcl)
            cPickle.dump(self.endRun.get(),pcl)
            pcl.close()
            os.system("chmod a+rw %s"%(os.path.join(self.basepath,self.pclName)))
            if self.class_debug:
                print "Wrote output to %s"%os.path.join(self.basepath,self.pclName)
                      
        except:
            print "Could not write file '.dbsDefaults.cPickle'"
        return



    def formParsedString(self):
        '''
        ** pydbsAccessor.formParsedString() **
        Creates a string to send to DBS based on
        user-inputted dataset and file values.
        '''

        if (self.class_debug):
            print self.formParsedString.__doc__
        
        temp=""
        myfile=self.searchStringFile.get()
        if (string.strip(myfile)<>""):
            temp=temp+"file=%s and "%myfile

        mydataset=self.searchStringDataset.get()
        if (string.strip(mydataset)<>""):
            temp=temp+" dataset=%s and "%mydataset

        if (self.class_debug):
            print "parsed string = '%s'"%temp
        return temp


    def searchDBS(self,beginrun=-1, endrun=-1,mytext=None):
        '''
        ** pydbsAccessor.searchDBS(beginrun=-1,endRun=-1,text=None) **

        Searches DBS for files matching specified criteria.
        Criteria is given by user-supplied "mytext" string.
        If no such string is provided, default is "find run
        where file = <dbsAccessor default searchString> and
        run between <default begin run>-<default end run>"

        '''

        if (self.class_debug):
            print self.searchDBS.__doc__

        # If beginrun, endrun specified, use their values in the search
        if (beginrun>-1):
            self.beginRun.set(beginrun)
        if (endrun>-1):
            self.endRun.set(endrun)
        if (mytext==None):
            temp=self.formParsedString()

            mytext="find run where %s run between %i-%i"%(temp,self.beginRun.get(),self.endRun.get())
        if (self.class_debug):  print "mytext = ",mytext

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
        if (self.class_debug):  print "SearchResult = %s"%self.searchResult
        return
    

###################################################################


if __name__=="__main__":
    root=Tk()
    temp=dbsAccessor(debug=1,pclName="pydbsAccessor_test.pcl")
    temp.Print()
    temp.searchDBS()
    #Label(root,text="This is a test window\nfor pydbsAccessor.py").grid(row=0)
    #root.mainloop()
    
