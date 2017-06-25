from array import array
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList

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
      self.CPEtype="template"
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
      elif self.flag=="cdcs":
         self.trkcoll="ALCARECOTkAlCosmicsInCollisions"
         self.useTrkSplittingInCosmics=False
      else:
         raise RuntimeError("Flag {} is unimplemented.".format(self.flag))


   def parseOptions(self):
      delimiter=' '
      optdelimiter=':'
      optlist=self.rawopt.split(delimiter)
      for sopt in optlist:
         olist=sopt.split(optdelimiter)
         if len(olist)==2:
            theKey=olist[0].lower()
            theVal=olist[1]
            self.optdict[theKey]=theVal
         else:
            raise RuntimeError("Option {} has an invalid number of delimiters {}".format(sopt,optdelimiter))


   def interpretOptions(self):
      gttogetpsets=[]
      for key,val in self.optdict.iteritems():
         # Get GT name
         if key=="gt":
            autofind=val.find("auto")
            if autofind>-1:
               val=val[0:autofind+4]+":"+val[autofind+4:]
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
                     tmpspec=namespec[1]
                     frontierfind=tmpspec.find("frontier")
                     sqlitefind=tmpspec.find("sqlite_file")
                     if frontierfind>-1 and sqlitefind>-1:
                        raise RuntimeError("Inconsistent setting: Cannot specify frontier and sqlite_file at the same time!")
                     elif frontierfind>-1:
                        tmpspec=tmpspec[0:frontierfind+8]+":"+tmpspec[frontierfind+8:]
                     elif sqlitefind>-1:
                        tmpspec=tmpspec[0:sqlitefind+11]+":"+tmpspec[sqlitefind+11:]
                     elif namespec[0]=="connect":
                        if tmpspec.endswith(".db"):
                           tmpspec = str("sqlite_file:")+tmpspec
                        elif tmpspec.find("//")>-1:
                           tmpspec = str("frontier:")+tmpspec
                        else:
                           tmpspec = str("frontier://")+tmpspec
                     cmsstrspec = cms.string(tmpspec)
                     insertValToPSet(namespec[0],cmsstrspec,apset)
                  else:
                     raise RuntimeError("GT specification does not ave size==2")
               gttogetpsets.append(apset)
         # Get data type
         elif (key=="type" or key=="datatype"):
            try:
               dtype=int(val)
               self.datatype=dtype
            except ValueError:
               print "Data type is not an integer"
         # Get lumi json file
         elif key=="lumilist":
            self.LumiJSON = LumiList.LumiList(filename = val).getVLuminosityBlockRange()
         # Get CPE type
         elif key=="cpe" or key=="cpetype":
            val=val.lower()
            self.CPEtype=val
         # Get non-standard track collection name
         elif key=="trackcollection":
            self.trkcoll=val
         ## Options for mMin. bias
         # Get custom track selection for TBD
         elif (key=="twobodytrackselection" or key=="twobodydecayselection" or key=="tbdselection"):
            val=val.lower()
            if (val=="zsel" or val=="y1sel"):
               self.TBDsel=val
            else:
               raise ValueError("TBD selection can only be Zsel or Y1sel at this time.")
         ## Get options common in min. bias, Zmumu and Ymumu
         # Get TBD constraint type
         elif (key=="twobodytrackconstraint" or key=="twobodydecayconstraint" or key=="tbdconstraint"):
            val=val.lower()
            if ("momconstr" in val or "fullconstr" in val):
               self.TBDconstraint=val
            else:
               raise ValueError("TBD constraint can only be momconstr... or fullconstr...")
         ## Options for cosmics
         # Get APV mode
         elif key=="apvmode":
            val=val.lower()
            if (val=="peak" or val=="deco"):
               self.APVmode=val
            else:
               raise ValueError("APV mode can only be peak or deco in cosmics")
         # Get magnetic field value
         elif key=="bfield":
            val=val.lower()
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








