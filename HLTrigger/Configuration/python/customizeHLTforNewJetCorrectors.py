import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Mixins import _ParameterTypeBase

_oldToNewProds = {'CaloJetCorrectionProducer':'CorrectedCaloJetProducer','PFJetCorrectionProducer':'CorrectedPFJetProducer'}

def _findModulesToChange(producers, prodsToFind):
  """Based on the type, find the labels for modules we need to change
  """
  modulesToChange = []
  for label,mod in producers.iteritems():
  	if mod.type_() in prodsToFind:
  	  modulesToChange.append(label)
  return modulesToChange

def _findCorrectorsUsedByProducer(process,producer):
  """Starting from a EDProducer, find which old jet correctors are used
  """
  correctors =[]
  for x in producer.correctors:
    _findCorrectorsRecursive(process,x,correctors)
  return correctors

def _findCorrectorsRecursive(process, name, correctors):
  """Find all Correctors used by the corrector 'name'
  """
  m=getattr(process,name)
  if m.type_() == 'JetCorrectionESChain':
    for n in m.correctors:
      _findCorrectorsRecursive(process,n,correctors)
  #we want all the dependent modules first
  correctors.append(name)

def _correctorNameChanger(oldName):
  """Based on the label from the old correct, determine what label to use for the new corrector
  """
  post1 = "CorrectionESProducer"
  if oldName[-1*len(post1):] == post1:
    return oldName[:-1*len(post1)]+"Corrector"
  post2 = "Correction"
  if oldName[-1*len(post2):] == post2:
    return oldName[:-1*len(post2)]+"Corrector"
  return None

def _translateParameters(oldModule, newModule):
  """Appropriately transform the parameters from the old module to the new module.
  This includes changing 'correctors' from a vstring to a VInputTag
  """
  for n in dir(oldModule):
    p = getattr(oldModule,n)
    if isinstance(p,cms._ParameterTypeBase):
      if n == 'appendToDataLabel':
        continue
      if n == "correctors" and isinstance(p,cms.vstring):
        p = cms.VInputTag( (cms.InputTag( _correctorNameChanger(tag) ) for tag in p))
      setattr(newModule,n,p)
  return newModule


def _makeNewCorrectorFromOld(oldCorrector):
  """Based on the old ESProducer corrector, create the equivalent EDProducer
  """
  oldToNewCorrectors_ = {'JetCorrectionESChain':'ChainedJetCorrectorProducer', 'LXXXCorrectionESProducer':'LXXXCorrectorProducer', 'L1FastjetCorrectionESProducer':'L1FastjetCorrectorProducer'}  
  type =  oldToNewCorrectors_[oldCorrector.type_()]
  corrector = cms.EDProducer(type)
  return _translateParameters(oldCorrector,corrector)

def _makeNewProducerFroOld(oldProducer):
  """Based on the old EDProducer which used a corrector from the EventSetup, create the appropriate EDProducer which gets the corrector from the Event.
  """
  type =  _oldToNewProds[oldProducer.type_()]
  newProd = cms.EDProducer(type)
  return _translateParameters(oldProducer,newProd)

def _buildSequenceOfCorrectors(process, correctorNames):
  """Using the dependencies between correctors, construct the appropriate cms.Sequence.
  """
  modSequence = None
  for n in correctorNames:
    mod = getattr(process,n)
    newLabel = _correctorNameChanger(n) 
    newMod = _makeNewCorrectorFromOld(mod)
    setattr(process,newLabel,newMod)
    if not modSequence:
      modSequence = newMod
    else:
      modSequence += newMod
  return cms.Sequence(modSequence)

def customizeHLTforNewJetCorrectors(process):
  modulesToChange = _findModulesToChange(process.producers, [x for x in  _oldToNewProds.iterkeys()])


  oldCorrectorsSequence = set()
  oldCorrectors = set()
  oldCorrectorsToNewSequence = {}
  for m in modulesToChange:
    correctors = _findCorrectorsUsedByProducer(process, getattr(process,m))
    oldCorrectors.update( correctors )
    stringOfOldCorrectorNames = ",".join(correctors)
    if stringOfOldCorrectorNames not in oldCorrectorsSequence:
      seq = _buildSequenceOfCorrectors(process,correctors)
      #need to attach to process since some code looks for labels for all items in a path
      setattr(process,"correctorSeqFor"+_correctorNameChanger(correctors[-1]), seq)
      oldCorrectorsSequence.add(stringOfOldCorrectorNames)
      oldCorrectorsToNewSequence[stringOfOldCorrectorNames] = seq
    
    #replace the old module
    oldModule = getattr(process,m)
    newModule = _makeNewProducerFroOld(oldModule)
    setattr(process,m,newModule)

    #need to insert the new sequence
    seq = oldCorrectorsToNewSequence[stringOfOldCorrectorNames]
    for p in process.paths.itervalues():
      if m in p.moduleNames():
        if not p.replace(newModule,seq+newModule):
          print "failed to replace ",m, "in path ", p.label_(), p

  #now remove the old correctors
  for m in oldCorrectors:
    delattr(process,m)

  return process
