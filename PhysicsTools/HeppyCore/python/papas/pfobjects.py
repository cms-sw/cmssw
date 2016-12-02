import math
from PhysicsTools.HeppyCore.particles.tlv.particle import Particle as BaseParticle
from PhysicsTools.HeppyCore.utils.deltar import deltaR
from PhysicsTools.HeppyCore.papas.data.identifier import Identifier

#add angular size needs to be fixed since at the moment the angluar size is set by the first element
#in a merged cluster. If the merged cluster is formed in a different order then the angular size will be different

class PFObject(object):
    '''Base class for all particle flow objects (tracks, clusters, etc).
    Particle flow objects of different types can be linked together
    forming graphs called "blocks".

    attributes:
    linked : list of PFObjects linked to this one
    locked : already used in the particle flow algorithm
    block_label : label of the block the PFObject belongs to. The block label is a unique identifier for the block.
    '''

    def __init__(self, pfobjecttype=Identifier.PFOBJECTTYPE.NONE):
   #def __init__(self):
        super(PFObject, self).__init__()

        self.linked = []
        self.locked = False
        self.block_label = None
        self.uniqueid = Identifier.make_id(pfobjecttype)

    def accept(self, visitor):
        '''Called by visitors, such as FloodFill. See pfalgo.floodfill'''
        notseen = visitor.visit(self)
        if notseen:
            for elem in self.linked:
                elem.accept(visitor)

    def __repr__(self):
        return str(self)

    def info(self):
        return ""

    def __str__(self):
        return '{classname}: {pretty:6}:{id}: {info}'.format(
            classname=self.__class__.__name__,
            pretty=Identifier.pretty(self.uniqueid),
            id=self.uniqueid,
            info=self.info())


class Cluster(PFObject):
    '''
    TODO:
    - not sure  max_energy plays well with SmearedClusters
    - investigate the possibility to have only one class.
     so: put mother in Cluster
     define the identifier outside?
    or stay as it is, but do not do any work in the child SmearedCluster and MergedCluster classes
    '''

    #TODO: not sure this plays well with SmearedClusters
    max_energy = 0.

    def __init__(self, energy, position, size_m, layer='ecal_in', particle=None):

        #may be better to have one PFOBJECTTYPE.CLUSTER type and also use the layer...
        if layer == 'ecal_in':
            super(Cluster, self).__init__(Identifier.PFOBJECTTYPE.ECALCLUSTER)
        elif layer == 'hcal_in':
            super(Cluster, self).__init__(Identifier.PFOBJECTTYPE.HCALCLUSTER)
        else :
            assert False
        self.position = position
        self.set_energy(energy)
        self.set_size(float(size_m))
        self.layer = layer
        self.particle = particle
        self.subclusters = [self]
        # self.absorbed = []

    def set_size(self, value):
        self._size = value
        try:
            self._angularsize = math.atan(self._size / self.position.Mag())
        except:
            import pdb; pdb.set_trace()

    def size(self):
        return self._size

    def angular_size(self):
        #angular_size is only properly specified for single (unmerged) clusters
        return self._angularsize

    def is_inside_clusters(self, other):
        '''TODO: no need for two versions of this method, see below.
        one should have a single overlap method that always works, whether or not there are any
        subclusters.
        '''
        #see if two clusters overlap (allowing for merged clusters which contain subclusters)
        #we have a link if any of the subclusters overlap
        #the distance is the distance betewen the weighted centres of each (merged) cluster

        dist = deltaR(self.position.Theta(),
                      self.position.Phi(),
                      other.position.Theta(),
                      other.position.Phi())

        for c in self.subclusters:
            for o in  other.subclusters:
                is_link, innerdist = c.is_inside_cluster(o)
                if is_link:
                    return True, dist
        return False, dist


    def is_inside_cluster(self, other):
        '''TODO change name to "overlaps" ? '''
        #now we have original unmerged clusters so we can compare directly to see if they overlap
        dR = deltaR(self.position.Theta(),
                    self.position.Phi(),
                    other.position.Theta(),
                    other.position.Phi())
        link_ok = dR < self.angular_size() + other.angular_size()
        return link_ok, dR


    def is_inside(self, point):
        """check if the point lies within the "size" circle of each of the subclusters"""
        subdist = []
        for subc in self.subclusters:
            dist = (subc.position - point).Mag()
            if dist < subc.size():
                subdist.append(dist)
        if len(subdist):
            return True, min(subdist)

        subdists = [(subc.position - point).Mag() for subc in self.subclusters]
        dist = min(subdists)
        return False, dist

        #subdists = [ (subc.position - point).Mag() for subc in self.subclusters ]
        #dist = min(subdists)
        #if dist < self.size():
            #return True, dist
        #else:
            #return False, dist

    def __iadd__(self, other):
        if other.layer != self.layer:
            raise ValueError('can only add a cluster from the same layer')
        position = self.position * self.energy + other.position * other.energy
        energy = self.energy + other.energy
        denom = 1/energy
        position *= denom
        self.position = position
        self.energy = energy
        assert len(other.subclusters) == 1
        self.subclusters.extend(other.subclusters)

        #todo recalculate the angular size
        return self

    def set_energy(self, energy):
        energy = float(energy)
        self.energy = energy
        if energy > self.__class__.max_energy:
            self.__class__.max_energy = energy
        self.pt = energy * self.position.Unit().Perp()

    # fancy but I prefer the other solution
    # def __setattr__(self, name, value):
    #     if name == 'energy':
    #         self.pt = value * self.position.Unit().Perp()
    #     self.__dict__[name] = value
    def info(self):
        subclusterstr = str('sub(')
        for s in self.subclusters:
            subclusterstr += str('{:}, '.format(Identifier.pretty(s.uniqueid)))
        subclusterstr += ")"
        return '{energy:7.2f} {theta:5.2f} {phi:5.2f} {sub}'.format(
            energy=self.energy,
            theta=math.pi/2. - self.position.Theta(),
            phi=self.position.Phi(),
            sub=subclusterstr
        )

class SmearedCluster(Cluster):
    def __init__(self, mother, *args, **kwargs):
        self.mother = mother
        super(SmearedCluster, self).__init__(*args, **kwargs)

class MergedCluster(Cluster):
    '''The MergedCluster is used to hold a cluster that has been merged from other clusters '''

    def __init__(self, mother):
        self.mother = mother
        super(MergedCluster, self).__init__(mother.energy, mother.position, mother._size, mother.layer, mother.particle)
        self.subclusters = [mother]

    def __iadd__(self, other):
        '''TODO: why not using iadd from base class'''
        if other.layer != self.layer:
            raise ValueError('can only add a cluster from the same layer')
        position = self.position * self.energy + other.position * other.energy
        energy = self.energy + other.energy
        denom = 1/energy
        position *= denom
        self.position = position
        self.energy = energy
        self.subclusters.extend([other])

        return self

class Track(PFObject):
    '''Determines the trajectory in space and time of a particle (charged or neutral).

    attributes:
    - p3 : momentum in 3D space (px, py, pz)
    - charge : particle charge
    - path : contains the trajectory parameters and points
    '''

    def __init__(self, p3, charge, path, particle=None):
        super(Track, self).__init__(Identifier.PFOBJECTTYPE.TRACK)
        self.p3 = p3
        self.pt = p3.Perp()
        self.energy = p3.Mag()  #TODO clarify energy and momentum
        self.charge = charge
        self.path = path
        self.particle = particle
        self.layer = 'tracker'

    def info(self):
        return '{e:7.2f} {pt:7.2f} {theta:5.2f} {phi:5.2f}'.format(
            pt=self.pt,
            e=self.energy,
            theta=math.pi/2. - self.p3.Theta(),
            phi=self.p3.Phi()
        )


class SmearedTrack(Track):
    def __init__(self, mother, *args, **kwargs):
        self.mother = mother
        self.path = mother.path
        super(SmearedTrack, self).__init__(*args, **kwargs)

class Particle(BaseParticle):
    def __init__(self, tlv, vertex, charge,
                 pdgid=None,
                 ParticleType=Identifier.PFOBJECTTYPE.PARTICLE):
        super(Particle, self).__init__(pdgid, charge, tlv)
        self.uniqueid = Identifier.make_id(ParticleType)
        self.vertex = vertex
        self.path = None
        self.clusters = dict()
        self.track = None # Alice Experiment to match cpp debug Track(self.p3(), self.q(), self.path)
        self.clusters_smeared = dict()
        self.track_smeared = None

    def __getattr__(self, name):
        if name == 'points':
            # if self.path is None:
            #     import pdb; pdb.set_trace()
            return self.path.points
        else:
            raise AttributeError

    def is_em(self):
        kind = abs(self.pdgid())
        return kind == 11 or kind == 22

    def set_path(self, path, option=None):
        if option == 'w' or self.path is None:
            self.path = path
            if self.q():
                self.track = Track(self.p3(), self.q(), self.path)

    def __str__(self):
        mainstr = super(Particle, self).__str__()
        idstr = '{pretty:6}:{id}'.format(
            pretty=Identifier.pretty(self.uniqueid),
            id=self.uniqueid)
        fields = mainstr.split(':')
        fields.insert(1, idstr)
        return ':'.join(fields)

if __name__ == '__main__':
    from ROOT import TVector3
    cluster = Cluster(10., TVector3(1, 0, 0), 1)  #alice made this use default layer
    print cluster.pt
    cluster.set_energy(5.)
    print cluster.pt
