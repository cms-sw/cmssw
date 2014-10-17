
# the ROOT TLorentzVector 
from ROOT import TLorentzVector

# the standard LorentzVector, used in CMSSW
from ROOT import gSystem
gSystem.Load("libDataFormatsRecoCandidate.so")
from ROOT import reco
LorentzVector = reco.LeafCandidate.LorentzVector

#COLIN need to add mathcore stuff
