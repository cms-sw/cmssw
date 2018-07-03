import re
import FWCore.ParameterSet.Config as cms

training_vars = {
   'muonMultiplicity': {'default': -1, 'type': 'i'},    
   'trackPPar': {'default': -1, 'max_idx': 3, 'type': 'f'},
   'flightDistance3dSig': {'default': -1, 'max_idx': 1, 'type': 'f'},
   'trackSip2dVal': {'default': -1, 'max_idx': 3, 'type': 'f'},
   'vertexBoostOverSqrtJetPt': {'default': -0.1, 'max_idx': 1, 'type': 'f'}, 
   'trackEtaRel': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'vertexMass': {'default': -0.1, 'max_idx': 1, 'type': 'f'}, 
   'trackDecayLenVal': {'default': -0.1, 'max_idx': 3, 'type': 'f'}, 
   'trackJetPt': {'default': -1, 'type': 'f'}, 
   'neutralHadronMultiplicity': {'default': -1, 'type': 'i'}, 
   'flightDistance3dVal': {'default': -0.1, 'max_idx': 1, 'type': 'f'}, 
   'trackJetDist': {'default': -0.1, 'max_idx': 3, 'type': 'f'}, 
   'leptonSip3d': {'default': -10000, 'max_idx': 3, 'type': 'f'}, 
   'neutralHadronEnergyFraction': {'default': -0.1, 'type': 'f'}, 
   'trackPtRatio': {'default': -0.1, 'max_idx': 3, 'type': 'f'}, 
   'hadronMultiplicity': {'default': -1, 'type': 'i'}, 
   'trackSumJetEtRatio': {'default': -0.1, 'type': 'f'}, 
   'vertexJetDeltaR': {'default': -0.1, 'max_idx': 1, 'type': 'f'}, 
   'leptonRatioRel': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'chargedHadronMultiplicity': {'default': -1, 'type': 'i'}, 
   'jetNTracks': {'default': -0.1, 'type': 'i'}, 
   'trackDeltaR': {'default': -0.1, 'max_idx': 3, 'type': 'f'}, 
   'vertexFitProb': {'default': -1, 'max_idx': 1, 'type': 'f'}, 
   'trackSip3dValAboveCharm': {'default': -1, 'max_idx': 1, 'type': 'f'}, 
   'jetEta': {'default': -3, 'type': 'f'}, 
   'leptonDeltaR': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'hadronPhotonMultiplicity': {'default': -1, 'type': 'i'}, 
   'leptonPtRel': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'flightDistance2dVal': {'default': -0.1, 'max_idx': 1, 'type': 'f'}, 
   'trackSumJetDeltaR': {'default': -0.1, 'type': 'f'}, 
   'photonMultiplicity': {'default': -1, 'type': 'i'}, 
   'chargedHadronEnergyFraction': {'default': -0.1, 'type': 'f'}, 
   'trackSip3dSigAboveQuarterCharm': {'default': -999, 'max_idx': 1, 'type': 'f'}, 
   'vertexLeptonCategory': {'default': -1, 'type': 'i'}, 
   'massVertexEnergyFraction': {'default': -0.1, 'max_idx': 1, 'type': 'f'}, 
   'trackSip2dSig': {'default': -100, 'max_idx': 3, 'type': 'f'}, 
   'flightDistance2dSig': {'default': -1, 'max_idx': 1, 'type': 'f'}, 
   'jetPt': {'default': -1, 'type': 'f'}, 
   'totalMultiplicity': {'default': -1, 'type': 'i'}, 
   'trackSip2dValAboveCharm': {'default': -1, 'max_idx': 1, 'type': 'f'}, 
   'electronEnergyFraction': {'default': -0.1, 'type': 'f'}, 
   'jetNSecondaryVertices': {'default': 0, 'type': 'i'}, 
   'trackSip2dSigAboveCharm': {'default': -999, 'max_idx': 1, 'type': 'f'}, 
   'vertexCategory': {'default': -1, 'type': 'i'}, 
   'vertexEnergyRatio': {'default': -10, 'max_idx': 1, 'type': 'f'}, 
   'photonEnergyFraction': {'default': -0.1, 'type': 'f'}, 
   'flavour': {'default': -1, 'type': 'i'}, 
   'muonEnergyFraction': {'default': -0.1, 'type': 'f'}, 
   'vertexNTracks': {'default': 0, 'max_idx': 1, 'type': 'i'}, 
   'trackSip2dSigAboveQuarterCharm': {'default': -999, 'max_idx': 1, 'type': 'f'}, 
   'trackSip3dVal': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'leptonRatio': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'trackPtRel': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'leptonEtaRel': {'default': -1, 'max_idx': 3, 'type': 'f'}, 
   'trackPParRatio': {'default': 1.1, 'max_idx': 3, 'type': 'f'}, 
   'trackSip3dSig': {'default': -100, 'max_idx': 3, 'type': 'f'}, 
   'trackSip3dSigAboveCharm': {'default': -999, 'max_idx': 1, 'type': 'f'}, 
   'electronMultiplicity': {'default': -1, 'type': 'i'}
}

#
# This could be a python class, but given it only used to convert the previous dict
# to CMSSW format I think is overkill
#
varname_regex_=re.compile(r'(?P<name>[a-zA-Z0-9]+)(:?_(?P<idx>\d+))?$')
def var_match(varname):
   '''matches the name used in the MVA training to 
   get the TaggingVariableName and index'''
   match = varname_regex_.match(varname)
   if not match:
      raise ValueError(
         'Variable name {0} does not match '
         'the default regular expression'.format(varname)
         )
   return match

def get_var_name(varname):
   'returns the TaggingVariableName of a MVA Name'
   match = var_match(varname)
   name = match.group('name')
   if name not in training_vars:
      raise ValueError(
         'Variable name {0}, matched to name {1}, '
         'is not among the known trainig variables.'.format(
            varname, name)
         )
   return name

def get_var_default(varname):
   'returns the default value used in the traing'
   name = get_var_name(varname)
   return training_vars[name]['default']

def get_var_idx(varname):
   'returns the index in case of vectorial TaggingVariableName'
   match = var_match(varname)
   idx   = match.group('idx')
   return int(idx) if idx else None

def get_var_pset(mvaname):
   'returns the cms.PSet to be used by CharmTaggerESProducer'
   pset = cms.PSet(      
      name = cms.string(mvaname),
      taggingVarName = cms.string(get_var_name(mvaname)),
      default = cms.double(get_var_default(mvaname))
      )
   idx = get_var_idx(mvaname)
   if idx is not None:
      pset.idx = cms.int32(idx)
   return pset

if __name__ == '__main__':
   assert(varname_regex_.match('leptonEtaRel_10').groupdict() == {'name': 'leptonEtaRel', 'idx': '10'})
   assert(varname_regex_.match('leptonEtaRel_1').groupdict() == {'name': 'leptonEtaRel', 'idx': '1'})
   assert(varname_regex_.match('leptonEtaRel').groupdict() == {'name': 'leptonEtaRel', 'idx': None})
   assert(varname_regex_.match('lepton_EtaRel') == None)
   
   assert(get_var_default('leptonEtaRel_10') == training_vars['leptonEtaRel']['default'])
   assert(get_var_default('electronMultiplicity') == training_vars['electronMultiplicity']['default'])
   assert(get_var_idx('leptonEtaRel_10') == 10)
   assert(get_var_idx('leptonEtaRel_3') == 3)
   assert(get_var_idx('FOOBAR') == None)
   
