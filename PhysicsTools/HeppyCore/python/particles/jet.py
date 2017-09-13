import math
from p4 import P4

def group_pdgid(ptc):
    pdgid = abs(ptc.pdgid())
    if pdgid>100:
        if ptc.q():
            return 211
        else:
            return 130
    else:
        return pdgid

class JetComponent(list):

    def __init__(self, pdgid):
        super(JetComponent, self).__init__()
        self._e = 0
        self._pt = 0
        self._num = 0
        self._pdgid = pdgid

    def pdgid(self):
        return self._pdgid
        
    def e(self):
        return self._e

    def pt(self):
        return self._pt

    def num(self):
        return self._num
    
    def append(self, ptc):
        pdgid = group_pdgid(ptc)
        if self._pdgid is None:
            self._pdgid = pdgid
        elif pdgid!=self._pdgid:
            raise ValueError('cannot add particles of different type to a component')
        super(JetComponent, self).append(ptc)
        self._e += ptc.e()
        self._pt += ptc.pt()
        self._num += 1

    def __str__(self):
        header = '\t\tpdgid={pdgid}, n={num:d}, e={e:3.1f}, pt={pt:3.1f}'.format(
            pdgid = self.pdgid(),
            num = self.num(),
            e = self.e(),
            pt = self.pt()
        )
        ptcs = []
        for ptc in self:
            ptcs.append('\t\t\t{particle}'.format(particle=str(ptc)))
        result = [header]
        result.extend(ptcs)
        return '\n'.join(result)
        
 
class JetConstituents(dict):

    def __init__(self):
        super(JetConstituents, self).__init__()
        all_pdgids = [211, 22, 130, 11, 13, 
                      1, 2 #HF had and em 
                      ]
        for pdgid in all_pdgids:
            self[pdgid] = JetComponent(pdgid)

    def validate(self, jet_energy, tolerance = 1e-2):
        '''Calls pdb if total component energy != jet energy'''
        tote = sum([comp.e() for comp in self.values()]) 
        if abs(jet_energy-tote)>tolerance: 
            import pdb; pdb.set_trace()
    
    def append(self, ptc):
        pdgid = group_pdgid(ptc)
        try:
            self[pdgid].append(ptc)
        except KeyError:
            import pdb; pdb.set_trace()

    def sort(self):
        for ptcs in self.values():
            ptcs.sort(key = lambda ptc: ptc.e(), reverse=True)

    def __str__(self):
        return '\n'.join(map(str, self.values()))


class JetTags(dict):

    def summary(self):
        tagstrs = []
        for name, val in sorted(self.iteritems()):
            valstr = '..'
            if hasattr(val, 'summary'):
                valstr = val.summary()
            elif isinstance(val, int):
                valstr = '{val:d}'.format(val=val)
            else:
                try:
                    valstr = '{val:2.1f}'.format(val=val)
                except:
                    pass
            tagstr = '{name}:{val}'.format(name=name, val=valstr)
            tagstrs.append(tagstr)
        return ', '.join(tagstrs)
            
    
class Jet(P4):

    def __init__(self, *args, **kwargs):
        super(Jet, self).__init__(*args, **kwargs)
        self.constituents = None
        self.tags = JetTags()

    def pdgid(self):
        return 0

    def q(self):
        return 0
    
    def __str__(self):
        tmp = '{className} : {p4}, tags={tags}'
        return tmp.format(
            className = self.__class__.__name__,
            p4 = super(Jet, self).__str__(),
            tags = self.tags.summary()
            )
    
    def __repr__(self):
        return str(self)
