from array import array
import FWCore.ParameterSet.Config as cms

# Helper functions
def getPSetDict(thePSet):
   return thePSet.parameters_()

def insertValToPSet(name,val,thePSet):
   setattr(thePSet,name,val)

def insertPSetToPSet(inPSet, outPSet):
   for key,val in getPSetDict(inPSet).iteritems():
      insertValToPSet(key,val,outPSet)

def insertPSetToVPSet(inPSet, outVPSet):
   outVPSet.append(inPSet)


def parseBoolString(theString):
   return theString[0].upper()=='T'

def isGoodEntry(var):
   if (var is None):
      return False
   elif (var == []):
      return False
   else:
      return True


class HipPyOptionParser:
   def __init__(self, strflag, stropt):
      # input file
      self.flag=strflag.lower()
      self.rawopt = stropt
      self.optdict=dict()

      self.datatype=0
      self.CPEtype="tempCPE"
      self.getTrackDefaults()

      self.parseOptions()
      self.interpretOptions()


   def getTrackDefaults(self):
      if self.flag=="mbvertex":
         self.trkcoll="ALCARECOTkAlMinBias"
         self.TBDsel=""
         self.TBDconstraint=""
         self.Bfield="3.8t"
      elif self.flag=="zmumu":
         self.trkcoll="ALCARECOTkAlZMuMu"
         self.TBDconstraint="fullconstr"
         self.Bfield="3.8t"
      elif (self.flag=="y1smumu" or self.flag=="y2smumu" or self.flag=="y3smumu"):
         self.trkcoll="ALCARECOTkAlUpsilonMuMu"
         self.TBDconstraint="fullconstr"
         self.Bfield="3.8t"
      elif self.flag=="cosmics":
         self.trkcoll="ALCARECOTkAlCosmicsCTF0T"
         self.useTrkSplittingInCosmics=False
         self.APVmode="deco"
      elif self.flag=="cdcs":
         self.trkcoll="ALCARECOTkAlCosmicsInCollisions"
         self.useTrkSplittingInCosmics=False
         self.APVmode="deco"


   def parseOptions(self):
      delimiter=' '
      optdelimiter=':'
      optlist=self.rawopt.split(delimiter)
      for sopt in optlist:
         olist=sopt.split(optdelimiter)
         if len(olist)==2:
            theKey=olist[0].lower()
            theVal=olist[1]
            if theKey!="trackcollection": # Do not lower the case of the track collection name
               theVal=theVal.lower()
            self.optdict[theKey]=theVal
         else:
            raise RuntimeError("Option {} has an invalid number of delimiters {}".format(sopt,optdelimiter))


   def interpretOptions(self):
      gttogetpsets=[]
      for key,val in self.optdict.iteritems():
         # Get GT name
         if key=="gt":
            self.GlobalTag = val
         # Get GT toGet PSets
         elif key=="gtspecs":
            vallist=val.split(';')
            for varset in vallist:
               apset = cms.PSet()
               specs = varset.split(',')
               for spec in specs:
                  namespec=spec.split('=')
                  if len(namespec)==2:
                     cmsstrspec = cms.string(namespec[1])
                     insertValToPSet(namespec[0],cmsstrspec,apset)
                  else:
                     raise RuntimeError("GT specification does not ave size==2")
               gttogetpsets.append(apset)
         # Get data type
         elif key=="type":
            try:
               dtype=int(val)
               self.datatype=dtype
            except ValueError:
               print "Data type is not an integer"
         # Get CPE type
         elif key=="cpe" or key=="cpetype":
            self.CPEtype=val
         # Get non-standard track collection name
         elif key=="trackcollection":
            self.trkcoll=val
         ## Options for mMin. bias
         # Get custom track selection for TBD
         elif key=="twobodytrackselection":
            if (val=="zsel" or val=="y1sel"):
               self.TBDsel=val
            else:
               raise ValueError("TBD selection can only be Zsel or Y1sel at this time.")
         ## Get options common in min. bias, Zmumu and Ymumu
         # Get TBD constraint type
         elif key=="twobodyconstraint":
            if (val=="momconstr" or val=="fullconstr"):
               self.TBDconstraint=val
            else:
               raise ValueError("TBD constraint can only be momconstr or fullconstr.")
         ## Options for cosmics
         # Get APV mode
         elif key=="apvmode":
            if (val=="peak" or val=="deco"):
               self.APVmode=val
            else:
               raise ValueError("APV mode can only be peak or deco in cosmics")
         # Get magnetic field value
         elif key=="bfield":
            if (val=="0t" or val=="zerotesla" or val=="3.8t"):
               self.Bfield=val
            else:
               raise ValueError("B field can only be 0T, ZEROTESLA or 3.8T")
         elif key=="usetracksplitting":
            self.useTrkSplittingInCosmics=parseBoolString(val)
         else:
            raise RuntimeError("Option {} is not implemented.".format(key))



      if len(gttogetpsets)>0:
         self.GTtoGet = cms.VPSet()
         for ps in gttogetpsets:
            insertPSetToVPSet(ps,self.GTtoGet)








