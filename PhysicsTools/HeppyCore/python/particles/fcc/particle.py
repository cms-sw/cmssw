from PhysicsTools.HeppyCore.particles.particle import Particle as BaseParticle
from vertex import Vertex
from pod import POD
from ROOT import TLorentzVector
from PhysicsTools.HeppyCore.papas.data.identifier import Identifier
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger
import copy

class Particle(BaseParticle, POD):
    
    def __init__(self, fccobj):
        super(Particle, self).__init__(fccobj)
        self.uniqueid=Identifier.make_id(Identifier.PFOBJECTTYPE.PARTICLE)
        self._charge = fccobj.Core().Charge
        self._pid = fccobj.Core().Type
        self._status = fccobj.Core().Status
        if hasattr(fccobj, 'StartVertex'):
            start = fccobj.StartVertex()
            self._start_vertex = Vertex(start) if start.isAvailable() \
                                 else None 
            end = fccobj.EndVertex()
            self._end_vertex = Vertex(end) if end.isAvailable() \
                               else None 
        self._tlv = TLorentzVector()
        p4 = fccobj.Core().P4
        self._tlv.SetXYZM(p4.Px, p4.Py, p4.Pz, p4.Mass)
        #write(str('Made Pythia {}').format(self))
        
    def __deepcopy__(self, memodict={}):
        newone = type(self).__new__(type(self))
        for attr, val in self.__dict__.iteritems():
            if attr not in ['fccobj', '_start_vertex', '_end_vertex']:
                setattr(newone, attr, copy.deepcopy(val, memodict))
        return newone

    def __str__(self):
        mainstr =  super(Particle, self).__str__()
        idstr = '{pretty:6}:{id}'.format(
            pretty = Identifier.pretty(self.uniqueid),
            id = self.uniqueid)
        fields = mainstr.split(':')
        fields.insert(1, idstr)
        return ':'.join(fields)     

