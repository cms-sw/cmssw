#! /usr/bin/env python3
# Sources:
#   https://cms-nanoaod-integration.web.cern.ch/integration/master-106X/mc106Xul18_doc.html#GenPart
#   https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/genparticles_cff.py
#   https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/plugins/LHETablesProducer.cc
from __future__ import print_function # for python3 compatibility
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
import PhysicsTools.NanoAODTools.postprocessing.framework.datamodel as datamodel
datamodel.statusflags['isHardPrompt'] = datamodel.statusflags['isPrompt'] + datamodel.statusflags['fromHardProcess'] 

def hasbit(value,bit):
  """Check if i'th bit is set to 1, i.e. binary of 2^i,
  from the right to the left, starting from position i=0.
  Example: hasbit(GenPart_statusFlags,0) -> isPrompt"""
  return (value & (1 << bit))>0


def getprodchain(part,genparts=None,event=None,decay=-1):
  """Print production chain recursively."""
  chain = "%3s"%(part.pdgId)
  imoth = part.genPartIdxMother
  while imoth>=0:
    if genparts is not None:
      moth = genparts[imoth]
      chain = "%3s -> "%(moth.pdgId)+chain
      imoth = moth.genPartIdxMother
    elif event is not None:
      chain = "%3s -> "%(event.GenPart_pdgId[imoth])+chain
      imoth = event.GenPart_genPartIdxMother[imoth]
  if genparts is not None and decay>0:
    chain = chain[:-3] # remove last particle
    chain += getdecaychain(part,genparts,indent=len(chain),depth=decay-1)
  return chain
  

def getdecaychain(part,genparts,indent=0,depth=999):
  """Print decay chain recursively."""
  chain   = "%3s"%(part.pdgId)
  imoth   = part._index
  ndaus   = 0
  indent_ = len(chain)+indent
  for idau in range(imoth+1,genparts._len): 
    dau = genparts[idau]
    if dau.genPartIdxMother==imoth: # found daughter
      if ndaus>=1:
        chain += '\n'+' '*indent_
      if depth>=2:
        chain += " -> "+getdecaychain(dau,genparts,indent=indent_+4,depth=depth-1)
      else: # stop recursion
        chain += " -> %3s"%(dau.pdgId)
      ndaus += 1
  return chain
  

# DUMPER MODULE
class LHEDumper(Module):
  
  def __init__(self):
    self.nleptonic = 0
    self.ntauonic  = 0
    self.nevents   = 0
  
  def analyze(self,event):
    """Dump gen information for each gen particle in given event."""
    print("\n%s Event %s %s"%('-'*10,event.event,'-'*70))
    self.nevents += 1
    leptonic = False
    tauonic = False
    bosons = [ ]
    taus = [ ]
    particles = Collection(event,'GenPart')
    #particles = Collection(event,'LHEPart')
    print(" \033[4m%7s %7s %7s %7s %7s %7s %7s %7s %8s %9s %10s  \033[0m"%(
      "index","pdgId","moth","mothId","dR","pt","eta","status","prompt","taudecay","last copy"))
    for i, particle in enumerate(particles):
      mothidx  = particle.genPartIdxMother
      eta      = max(-999,min(999,particle.eta))
      prompt   = particle.statusflag('isPrompt') #hasbit(particle.statusFlags,0)
      taudecay = particle.statusflag('isTauDecayProduct') #hasbit(particle.statusFlags,2)
      lastcopy = particle.statusflag('isLastCopy') #hasbit(particle.statusFlags,13)
      #ishardprompt = particle.statusflag('isHardPrompt')
      if 0<=mothidx<particles._len:
        moth    = particles[mothidx]
        mothpid = moth.pdgId
        mothdR  = max(-999,min(999,particle.DeltaR(moth))) #particle.p4().DeltaR(moth.p4())
        print(" %7d %7d %7d %7d %7.2f %7.2f %7.2f %7d %8s %9s %10s"%(
          i,particle.pdgId,mothidx,mothpid,mothdR,particle.pt,eta,particle.status,prompt,taudecay,lastcopy))
      else:
        print(" %7d %7d %7s %7s %7s %7.2f %7.2f %7d %8s %9s %10s"%(
          i,particle.pdgId,"","","",particle.pt,eta,particle.status,prompt,taudecay,lastcopy))
      if lastcopy:
        if abs(particle.pdgId) in [11,13,15]:
          leptonic = True
          if abs(particle.pdgId)==15:
            tauonic = True
            taus.append(particle)
        elif abs(particle.pdgId) in [23,24]:
          bosons.append(particle)
    for boson in bosons: # print production chain
      print("Boson production:")
      print(getprodchain(boson,particles,decay=2))
    for tau in taus: # print decay chain
      print("Tau decay:")
      print(getdecaychain(tau,particles))
    if leptonic:
      self.nleptonic += 1
    if tauonic:
      self.ntauonic += 1
  
  def endJob(self):
    print('\n'+'-'*96)
    if self.nevents>0:
      print("  %-10s %4d / %-4d (%4.1f%%)"%('Tauonic: ',self.ntauonic, self.nevents,100.0*self.ntauonic/self.nevents))
      print("  %-10s %4d / %-4d (%4.1f%%)"%('Leptonic:',self.nleptonic,self.nevents,100.0*self.nleptonic/self.nevents))
    print("%s Done %s\n"%('-'*10,'-'*80))
  

# PROCESS NANOAOD
url = "root://cms-xrd-global.cern.ch/"
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-i', '--infiles', nargs='+')
parser.add_argument('-o', '--outdir', default='.')
parser.add_argument('-n', '--maxevts', type=int, default=20)
args = parser.parse_args()
infiles = args.infiles or [
  url+'/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/525CD279-3344-6043-98B9-2EA8A96623E4.root',
  #url+'/store/mc/RunIISummer20UL18NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/44187D37-0301-3942-A6F7-C723E9F4813D.root',
]
processor = PostProcessor(args.outdir,infiles,noOut=True,modules=[LHEDumper()],maxEntries=args.maxevts)
processor.run()
