from array import array
from copy import copy
from copy import deepcopy
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


def matchPSetsByRecord(ps1, ps2):
   if hasattr(ps1,"record") and hasattr(ps2,"record"):
      s1=ps1.record.value()
      s2=ps2.record.value()
      return (s1==s2)
   return False


def mergeVPSets(inVPSet, overrideVPSet, matchrule=None):
   resvpset=overrideVPSet.copy()
   for iop in inVPSet.value():
      nomatch=True
      if matchrule is not None:
         for cps in overrideVPSet.value():
            if matchrule(cps,iop):
               nomatch=False
               break
      if nomatch:
         insertPSetToVPSet(iop,resvpset)
   return resvpset




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

      self.datatype=-1
      self.CPEtype="template"
      self.getTrackDefaults()

      if self.rawopt.lower()!="noopts":
         self.parseOptions()
         self.interpretOptions()


   def getTrackDefaults(self):
      if self.flag=="mbvertex":
         self.trkcoll="ALCARECOTkAlMinBias"
         self.Bfield="3.8t"
      elif self.flag=="zmumu":
         self.trkcoll="ALCARECOTkAlZMuMu"
         self.Bfield="3.8t"
      elif self.flag=="ymumu":
         self.trkcoll="ALCARECOTkAlUpsilonMuMu"
         self.Bfield="3.8t"
      elif self.flag=="jpsimumu":
         self.trkcoll="ALCARECOTkAlJpsiMuMu"
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
               specs = varset.split('|')
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
                     raise RuntimeError("GT specification {} does not have size==2".format(namespec))
               gttogetpsets.append(apset)
         # Get hits to drop or keep
         elif key=="hitfiltercommands":
            vallist=val.split(';')
            for iv in range(0,len(vallist)):
               keepdrop_det_pair=vallist[iv].split('=')
               if len(keepdrop_det_pair)==2:
                  if (keepdrop_det_pair[0]=="keep" or keepdrop_det_pair[0]=="drop"):
                     strcmd = keepdrop_det_pair[0]

                     keepdrop_det_pair[1]=keepdrop_det_pair[1].replace('/',' ') # e.g. 'PIX/2' instead of 'PIX 2'
                     keepdrop_det_pair[1]=keepdrop_det_pair[1].upper()

                     strcmd = strcmd + " " + keepdrop_det_pair[1]
                     if not hasattr(self,"hitfiltercommands"):
                        self.hitfiltercommands=[]
                     self.hitfiltercommands.append(strcmd)
                  else:
                     raise RuntimeError("Keep/drop command {} is not keep or drop.".format(keepdrop_det_pair[0]))
               else:
                  raise RuntimeError("Keep/drop-det. pair {} does not have size==2 or has a command other than keep or drop.".format(vallist[iv]))
         # Get data type
         elif (key=="type" or key=="datatype" or key=="datagroup"):
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
         # Get overall weight. Turns reweighting on
         elif key=="overallweight":
            try:
               fval=float(val)
               self.overallweight=fval
            except ValueError:
               print "Overall weight is not a float"
         # Get uniform eta formula. Turns reweighting on
         elif key=="uniformetaformula":
            self.uniformetaformula=val
         ## Options for mMin. bias
         # Apply vertex constraint
         elif (key=="primaryvertextpye" or key=="pvtype"):
            val=val.lower()
            if (val=="nobs" or val=="withbs"):
               self.PVtype=val
            else:
               raise ValueError("PV type can only receive NoBS or WithBS.")
         elif (key=="primaryvertexconstraint" or key=="pvconstraint"):
            self.applyPVConstraint=parseBoolString(val)
            if not hasattr(self,"PVtype"):
               self.PVtype="nobs"
         # Get custom track selection for TBD
         elif (key=="twobodytrackselection" or key=="twobodydecayselection" or key=="tbdselection"):
            val=val.lower()
            if (val=="zsel" or val=="y1ssel"):
               self.TBDsel=val
            else:
               raise ValueError("TBD selection can only be Zsel or Y1Ssel at this time.")
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


   def doCheckOptions(self,optstocheck):
      # First check option consistencies overall
      if (hasattr(self,"TBDconstraint") and hasattr(self,"applyPVConstraint")):
         raise RuntimeError("Options TBDconstraint and applyPVConstraint cannot coexist.")
      # Force presence of the options passed
      for oc in optstocheck:
         if not hasattr(self,oc):
            raise RuntimeError("Option {} needs to specified in {}.".format(oc, self.flag))


   def checkOptions(self):
      optstocheck=[]
      checkcosmics=(self.flag=="cosmics" or self.flag=="cdcs")
      checkymumuconstr=(self.flag=="ymumu" and hasattr(self, "TBDconstraint"))
      if checkcosmics:
         optstocheck=[
            "Bfield",
            "APVmode",
            "useTrkSplittingInCosmics"
         ]
      if checkymumuconstr:
         optstocheck=[
            "TBDsel"
         ]
      self.doCheckOptions(optstocheck)

