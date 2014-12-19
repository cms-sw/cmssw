import math
import fnmatch

from PhysicsTools.Heppy.physicsutils.TauDecayModes import tauDecayModes

def printOut(objects):
    if len(objects)==0:
        return ''
    return '\n'.join( map( type(objects[0]).__str__, objects) )

from PhysicsTools.Heppy.physicsobjects.PhysicsObject import PhysicsObject
from PhysicsTools.Heppy.physicsobjects.TriggerObject import TriggerObject
from PhysicsTools.Heppy.physicsobjects.Jet import Jet, GenJet
from PhysicsTools.Heppy.physicsobjects.Lepton import Lepton
from PhysicsTools.Heppy.physicsobjects.Photon import Photon
from PhysicsTools.Heppy.physicsobjects.Muon import Muon
# COLIN need to import MVA ID recipe. 
# from PhysicsTools.Heppy.physicsobjects.Electron import Electron
from PhysicsTools.Heppy.physicsobjects.Tau import Tau, isTau
from PhysicsTools.Heppy.physicsobjects.GenParticle import GenParticle 

