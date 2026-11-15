import ROOT
import sys


"""
Usage 
python createLHEFormatFromROOTFile.py inputfile outputfile pdgId_particle_to_undo_decay1 pdgId_particle_to_undo_decay2 pdgId_particle_to_undo_decay3 ...
"""

args = sys.argv[:]

class HEPPart(object):
  def __init__(self,event, idx):
    """
    Class to organize the description of a particle in the LHE event
    event : whole event information (usually a entry in the input TTree)
    idx : the index of the particle inside the LHE file
    """
    for att in ["pt","eta","phi", "mass","lifetime","pdgId","status","spin","color1", "color2","mother1","mother2","incomingpz"]:
      setattr(self, att, getattr(event, "LHEPart_"+att)[idx])
    self.setP4()
    
  def setP4(self):
    self.p4 = ROOT.TLorentzVector()
    if self.status != -1:
      self.p4.SetPtEtaPhiM(self.pt, self.eta, self.phi, self.mass)
    else:
      self.p4.SetPxPyPzE(0.,0.,self.incomingpz, abs(self.incomingpz))
  def printPart(self):
    """ Just to print it pretty """
    return "       {pdg:d} {status:d}    {mother1:d}    {mother2:d}  {color1:d}  {color2:d} {px:e} {py:e} {pz:e} {energy:e} {mass:e} {time:e} {spin:e}\n".format(pdg=self.pdgId,status=self.status, mother1=self.mother1, mother2=self.mother2, color1=self.color1, color2=self.color2, px=self.p4.Px(), py=self.p4.Py(), pz=self.p4.Pz(), energy=self.p4.E(), mass=self.mass, time=self.lifetime, spin=self.spin)


class LHEPrinter(object):
  def __init__(self,theFile,theTree,outputLHE,undoDecays=[], chunkers=-1, prDict={}):
    """
    theFile : path to input root file with the whole LHEinformation
    theTree : number of ttree inside the file, usually "Events"
    outputLHE: name of output .lhe file
    undoDecays: pdgId of particles whose decays we want to undo (i.e. W that are decayed with madspin, as the reweighting is called before madspin)
    chunkers: process by chunks in case we want to later multithread the jobs. Argument is max size (in events) of the chunks
    prDict  : a dictionary indicating possible mismatchings between the ProcessID in the new gridpack and the old (matching the numbers at generate p p > x y z  @0 in the run card)
    """

    self.prDict = prDict
    self.outputLHE = outputLHE
    self.fil  = ROOT.TFile(theFile,"OPEN")
    self.tree = self.fil.Get(theTree)
    self.undoDecays = undoDecays
    self.baseheader =  "<event nplo=\" {nplo} \" npnlo=\" {npnlo} \">\n"
    self.baseline1 = " {nparts}      {prid} {weight} {scale} {aqed} {aqcd}\n"
    self.baseender = "<rwgt>\n</rwgt>\n</event>\n"
    self.chunkers = chunkers

  def insideLoop(self):
    """ Loop over all events and process the root file into a plain text LHE file"""
    totalEvents = self.tree.GetEntries()
    if self.chunkers == -1 or self.chunkers > totalEvents:
      self.output = open(self.outputLHE,"w")
      self.chunkers = totalEvents + 1
    else:
      self.output = open(self.outputLHE+"chunk0","w")
      chunk = 0

    print "Processing %i events, please wait..."%totalEvents
    iEv = 0
    pEv = 0
    chunk = 0
    for ev in self.tree:
      iEv += 1
      pEv += 1
      if pEv >= self.chunkers:
        pEv = 0
        chunk += 1 
        self.output.close()
        self.output = open(self.outputLHE+"chunk%i"%chunk,"w")
      print "...Event %i/%i"%(iEv, totalEvents)
      self.process(ev)

  def process(self, ev):
    """First produce the global line like <event nplo=" -1 " npnlo=" 1 ">"""
    self.output.write(self.baseheader.format(nplo = ord(str(ev.LHE_NpLO)) if ord(str(ev.LHE_NpLO)) != 255  else  -1, npnlo = ord(str(ev.LHE_NpNLO)) if ord(str(ev.LHE_NpNLO)) != 255  else  -1))

    """Then we need to treat the whole thing to undo the madspin decays, update statuses and rewrite particle order"""
    lhepart = []
    deletedIndexes = []
    for i in range(getattr(ev, "nLHEPart")):
      testPart = HEPPart(ev,i)
      testPart.mother1 = testPart.mother1 - sum([1*(testPart.mother1 > d) for d in deletedIndexes])
      testPart.mother2 = testPart.mother2 - sum([1*(testPart.mother2 > d) for d in deletedIndexes])
      if testPart.mother1 != 0:
        if abs(lhepart[testPart.mother1-1].pdgId) in self.undoDecays: #If from something that decays after weighting just skip it and update particle indexes
          deletedIndexes.append(i)
          continue
      if abs(testPart.pdgId) in self.undoDecays:
        testPart.status = 1 
      lhepart.append(testPart)
    
    """ Now we can compute properly the number of particles at LHE """
    self.output.write(self.baseline1.format(nparts=len(lhepart), prid=self.prDict[str(ord(str(ev.LHE_ProcessID)))], weight=ev.LHEWeight_originalXWGTUP, scale=ev.LHE_Scale,aqed=ev.LHE_AlphaQED,aqcd=ev.LHE_AlphaS))
    """ And save each particle information """
    for part in lhepart:
      self.output.write(part.printPart())   
    self.output.write(self.baseender)

theP = LHEPrinter(args[1],"Events",args[2],undoDecays=[int(i) for i in args[4:]],chunkers=int(args[3]), prDict={str(i):i for i in range(1000)}) #PRDict by default set to not change anything as it is rare to use it 
theP.insideLoop()
