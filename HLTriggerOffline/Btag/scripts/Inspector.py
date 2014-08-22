#____________________________________________________________
#
#  cuy
#
# A very simple way to make plots with ROOT via an XML file
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2008
#
#____________________________________________________________

import sys
import ROOT
from ROOT import *


def getObjectsPaths(self):
     mylist = []
     for key in gDirectory.GetListOfKeys():
        mypath = gDirectory.GetPathStatic()
        self.filterKey(key,mypath,mylist)
        gDirectory.cd(mypath)
    
     return mylist

def filterKey( self, mykey , currentpath, tolist):
     if mykey.IsFolder():
         if not currentpath.endswith('/'):
             currentpath+='/'
         print '   <!-- '+currentpath+' -->'
         topath =  currentpath+mykey.GetName()  
         self.cd(topath)
         for key in gDirectory.GetListOfKeys():
              self.filterKey(key,topath,tolist)
     else:
         tolist.append(gDirectory.GetPathStatic()+'/'+mykey.GetName())
         stripfilename=self.GetName()
         stripfilename=stripfilename.replace(".root","")
         path2TH1=gDirectory.GetPathStatic()+'/'+mykey.GetName()
         path2TH1=path2TH1[path2TH1.find(":")+1:len(path2TH1)]
         print '   <TH1 name=\"'+stripfilename+'_'+mykey.GetName()+'\" source=\"'+path2TH1+'\"/>'

     return
                 
TFile.filterKey = filterKey
TFile.getObjectsPaths = getObjectsPaths




class Inspector:

    def SetFilename(self, value):
	self.Filename = value
    def Verbose(self, value):
	self.Verbose = value

    def createXML(self, value):
	self.XML = value

    def SetTag(self,value):
	self.tag = value
	self.TagOption = True

    def Loop(self,mylist,prevdir,mydir):
		
     afile = TFile(self.Filename)
     afilename = self.Filename
     stripfilename = afilename
#     ROOT.gDirectory=mydir
     try:
       if self.TagOption: stripfilename = self.tag
     except:
       stripfilename = afilename.split('/')[len(afilename.split('/')) -1]
       stripfilename = stripfilename[0:(len(stripfilename)-5)]
#     alist = mydir.GetListOfKeys()
     alist =mylist
     print "alist= ", alist.Print()
     for i in alist:
        aobj = i.ReadObj()
        self.dir2=prevdir
        if aobj.IsA().InheritsFrom("TDirectory"):
         if self.Verbose:
          print ' found directory: '+i.GetName()
         if self.XML:
          print '   <!-- '+i.GetName()+' -->'

#        bdir = ROOT.MakeNullPointer("TDirectory" )
#         bdir = self.dir
#         print bdir 
#		afile.GetObject(i.GetName(),bdir)
#		blist = bdir.GetListOfKeys()
         print "1 myROOT.gDirectory ", ROOT.gDirectory.GetName()
         print "my aobj=", aobj.GetName()
#         self.dir3=self.dir2
         self.dir2+=aobj.GetName()+"/"
         print "dir2=", self.dir2
#         ROOT.gDirectory.cd(aobj.GetName())
#         ROOT.gDirectory.cd(self.dir2)
         mydir.cd(self.dir2)
#         self.dir= ROOT.gDirectory
#         print "2 myROOT.gDirectory ", self.dir.GetName()
         print "2 list ", ROOT.gDirectory.GetListOfKeys().Print()

         
#         self.Loop(ROOT.gDirectory.GetListOfKeys(),self.dir2,mydir)
         self.Loop(mydir.GetListOfKeys(),self.dir2,mydir)
        if aobj.IsA().InheritsFrom(ROOT.TH1.Class()):
#         self.dir2=prevdir
         if self.Verbose:
          print '  --> found TH1: name = '+aobj.GetName() + ' title = '+aobj.GetTitle()
         if self.XML:
          print '   <TH1 name=\"'+stripfilename+'_'+aobj.GetName()+'\" source=\"'+'/'+ self.dir2+aobj.GetName()+'\"/>'
     print prevdir
     ROOT.gDirectory.cd("..")
     print "2 myROOT.gDirectory ", ROOT.gDirectory.GetName()

			
    def GetListObjects(self):
	
     afile = TFile(self.Filename)
     if afile.IsZombie():
	   print " error trying to open file: " + self.Filename
	   sys.exit()
	
     if self.XML:

      print '''
<cuy>

'''	
     print '  <validation type=\"'+afile.GetName()+'\" file=\"'+self.Filename+'\" release=\"x.y.z\">'
     self.dir = ROOT.gDirectory
     self.dir2 = ""
     self.dir3 = ""
#	self.Loop(ROOT.gDirectory.GetListOfKeys(),"",ROOT.gDirectory)
     myobjects=afile.getObjectsPaths()
     if self.XML:
      print '''
  </validation>

</cuy>
'''
	    





    


