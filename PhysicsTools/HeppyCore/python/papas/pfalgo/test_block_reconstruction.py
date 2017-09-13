import unittest
from PhysicsTools.HeppyCore.papas.graphtools.DAG import Node, BreadthFirstSearchIterative,DAGFloodFill
from PhysicsTools.HeppyCore.papas.data.identifier import Identifier
from PhysicsTools.HeppyCore.papas.graphtools.edge import Edge
from PhysicsTools.HeppyCore.papas.pfalgo.pfblockbuilder import PFBlockBuilder
#from PhysicsTools.HeppyCore.papas.pfalgo.pfblockbuilder import BlockSplitter
from PhysicsTools.HeppyCore.papas.pfalgo.pfblock import PFBlock as realPFBlock


class Cluster(object):
    ''' Simple Cluster class for test case
        Contains a long uniqueid (created via Identifier class), a short id (used for distance) and a layer (ecal/hcal)
    '''
    def __init__(self, id, layer):
        ''' id is unique integer from 101-199 for ecal cluster
              unique integer from 201-299 for hcal cluster
              layer is ecal/hcal
        '''
        if (layer == 'ecal_in'):
            self.uniqueid = Identifier.make_id(Identifier.PFOBJECTTYPE.ECALCLUSTER)
        elif (layer == 'hcal_in'):
            self.uniqueid = Identifier.make_id(Identifier.PFOBJECTTYPE.HCALCLUSTER)
        else:
            assert false
        self.layer = layer
        self.id = id
        self.energy=0

    def __repr__(self):
        return "cluster:" +  str(self.id) + " :" + str(self.uniqueid)
        
class Track(object):
    ''' Simple Track class for test case
        Contains a long uniqueid (created via Identifier class), a short id (used for distance) and a layer (tracker)
    '''
    def __init__(self, id):
        ''' id is unique integer from 1-99
        '''
        self.uniqueid = Identifier.make_id(Identifier.PFOBJECTTYPE.TRACK)
        self.id = id
        self.layer = 'tracker'
        self.energy=0
        
    def __repr__(self):
        return "track:"+  str(self.id) + " :"+  str(self.uniqueid)
        
class Particle(object):
    ''' Simple Particle class for test case
        Contains a long uniqueid (created via Identifier class), a short id (used for distance) and a ppdgid
    '''
    def __init__(self, id, pdgid):
        ''' id is unique integer from 301-399
            pdgid is particle id eg 22 for photon
        '''
        self.uniqueid = Identifier.make_id(Identifier.PFOBJECTTYPE.PARTICLE)
        #print "particle: ",self.uniqueid," ",id
        self.pdgid = pdgid
        self.id = id
        #self.type = PFType.PARTICLE
    
    def __repr__(self):
        return "particle:"+  str(self.id) + " :"+  str(self.uniqueid)        
class ReconstructedParticle(Particle):
    ''' Simple Particle class for test case
        Contains a long uniqueid (created via Identifier class), a short id (used for distance) and a ppdgid
    '''
    def __init__(self, id,pdgid):
        ''' id is unique integer from 601-699
            pdgid is particle id eg 22 for photon
        '''
        self.uniqueid = Identifier.make_id(Identifier.PFOBJECTTYPE.RECPARTICLE)
        self.pdgid = pdgid
        self.id = id
        
    def __repr__(self):
        return "reconstructed particle:"+  str(self.id) + " :"+  str(self.uniqueid)    


            
class Event(object):
    ''' Simple Event class for test case
        Used to contains the tracks, clusters, particles
        and also nodes describing history (which particle gave rise to which track)
        and nodes describing links/distances between elements
    '''
    def __init__(self, distance):
        self.sim_particles = dict() #simulated particles
        self.reconstructed_particles = dict() #reconstructed particles
        self.ecal_clusters = dict() 
        self.hcal_clusters = dict()
        self.tracks = dict()         #tracks 
        self.history_nodes = dict()  #Nodes used in simulation/reconstruction (contain uniqueid)
        self.nodes = dict()          #Contains links/ distances between nodes
        self.blocks = dict()         #Blocks to be made for use in reconstuction
        self.ruler = distance
        #self.get_object=GetObject(self)
    
    def get_object(self, uniqueid):
        ''' given a uniqueid return the underlying obejct
        '''
        type = Identifier.get_type(uniqueid)
        if type == Identifier.PFOBJECTTYPE.TRACK:
            return self.tracks[uniqueid]       
        elif type == Identifier.PFOBJECTTYPE.ECALCLUSTER:      
            return self.ecal_clusters[uniqueid] 
        elif type == Identifier.PFOBJECTTYPE.HCALCLUSTER:            
            return self.hcal_clusters[uniqueid]            
        elif type == Identifier.PFOBJECTTYPE.PARTICLE:
            return self.sim_particles[uniqueid]   
        elif type == Identifier.PFOBJECTTYPE.RECPARTICLE:
            return self.reconstructed_particles[uniqueid]               
        elif type == Identifier.PFOBJECTTYPE.BLOCK:
            return self.blocks[uniqueid]               
        else:
            assert(False)   

class Simulator(object):
    ''' Simplified simulator for testing
        The simulator sets up  two charged hadrons and a photon
        the clusters/tracks/particles contain a short id where
        #    1- 99 is  a track
        #  101-199 is an ecal cluster
        #  201-299 is an hcal cluster
        #  301-399 is  a simulated particle
        #  401-499 is  a block
        #  601-699 is  a reconstructed particle 
        the short ids are used to construct the distances between elements
        The elements also have a long unique id which is created via
        an Identifier class
    '''
    def __init__(self,event):
        self.event = event
       
        #add some clusters /tracks/ particles
        self.add_ecal_cluster(101)
        self.add_hcal_cluster(202)
        self.add_ecal_cluster(102)
        self.add_ecal_cluster(103)
        self.add_track(1)
        self.add_track(2)
        self.add_particle(301,211) #charged hadron
        self.add_particle(302,211) #charged hadron
        self.add_particle(303,22)  #photon
        #define links between clusters/tracks and particles
        self.add_link(self.UID(301),self.UID(101))
        self.add_link(self.UID(301),self.UID(1))
        self.add_link(self.UID(302),self.UID(2))
        self.add_link(self.UID(302),self.UID(102))
        self.add_link(self.UID(302),self.UID(202))
        self.add_link(self.UID(303),self.UID(103))
              
    def add_ecal_cluster(self, id):
        clust = Cluster(id,'ecal_in')# make a cluster
        uniqueid = clust.uniqueid 
        self.event.ecal_clusters[uniqueid] = clust     # add into the collection of clusters
        self.event.history_nodes[uniqueid] = Node(uniqueid)  #add into the collection of History Nodes
                 
    def add_hcal_cluster(self, id):
        clust = Cluster(id,'hcal_in')
        uniqueid = clust.uniqueid 
        self.event.hcal_clusters[uniqueid] = clust
        self.event.history_nodes[uniqueid] = Node(uniqueid)
        
    def add_track(self, id):
        track = Track(id)
        uniqueid = track.uniqueid 
        self.event.tracks[uniqueid] =  track
        self.event.history_nodes[uniqueid] =  Node(uniqueid) 
        
    def add_particle(self, id, pdgid):
        particle = Particle(id,pdgid)
        uniqueid = particle.uniqueid
        self.event.sim_particles[uniqueid] =  particle
        self.event.history_nodes[uniqueid] =  Node(uniqueid) 
    
    def UID(self, id): #Takes the test case short id and find the unique id
        ''' id is the short id of the element
            this returns the corresponding long unique id
        '''
        for h in self.event.history_nodes :
            obj = self.event.get_object(h)
            if hasattr(obj, "id"):
                if obj.id  == id :
                    return obj.uniqueid 
        return 0
     
    def short_id(self, uniqueid): #Takes the unique id and finds corresponding short id
        ''' uniqueid is the long unique id of the element
            this returns the corresponding short integer id
        '''
        for h in self.event.history_nodes :
            obj = self.event.get_object(h)
            if hasattr(obj, "id"):
                if obj.uniqueid  == uniqueid :
                    return obj.id 
        return 0               

    def add_link(self, uniqueid1, uniqueid2):
        ''' create a parent child link in the history nodes between two elements
            uniqueid1, uniqueid2 are the elements unique ids
        '''
        self.event.history_nodes[uniqueid1].add_child(self.event.history_nodes[uniqueid2])
        

    
class Reconstructor(object):
    ''' Simplified reconstructor class for testing
    '''
    def __init__(self, event):
        self.event  =  event 
        self.particlecounter  =  600 #used to create reconstructed particle short ids
        self.reconstruct_particles()
    
    def add_nodes(self, nodedict,values):
        for e1 in values :
            nodedict[e1.uniqueid] =  Node(e1.uniqueid)                   
    
    def reconstruct_particles (self):
        for block in self.event.blocks.itervalues():
            self.make_particles_from_block (block)    
    
    def make_particles_from_block(self, block):
        ''' Take a block and use simple rules to construct particles 
        '''
        #take a block and find its parents (clusters and tracks)
        parents = block.element_uniqueids
        
        if  (len(parents) == 1) & (Identifier.is_ecal(parents[0])):
            #print "make photon"
            self.make_photon(parents)
            
        elif ( (len(parents) == 2)  & (block.count_ecal() == 1 ) & (block.count_tracks() == 1)):
            #print "make hadron" 
            self.make_hadron(parents)
            
        elif  ((len(parents) == 3)  & (block.count_ecal() == 1) & (block.count_tracks() == 1) & (block.count_hcal() == 1)):
                #print "make hadron and photon"
                #probably not right but illustrates splitting of parents for more than one particle
                hparents = [] # will contain parents for the Hadron which gets everything except the 
                              #hcal which is used for the photom
                for elem in parents:
                    if (Identifier.is_hcal(elem)):
                        self.make_photon({elem})
                    else :
                        hparents.append(elem)    
                self.make_hadron(hparents)
    
        else :
            pass
            #print "particle TODO"  
         
    def make_photon(self, parents):
        return self.add_particle(self.new_id(), 22,parents)

    def make_hadron(self, parents):
        return self.add_particle(self.new_id(), 211,parents)

    def add_particle(self, id, pdgid, parents):
        ''' creates a new particle and then updates the 
            event to include the new node and its parental links
            pdgid = is the particle type id eg 22 for photon
            parents = list of the unique ids (from Identifier class) for the elements from 
                       which the particle has been reconstructed
        '''
        particle = ReconstructedParticle(id,pdgid)
        self.event.reconstructed_particles[particle.uniqueid] =  particle
        #Now create the history node and links
        particle_node = Node(particle.uniqueid)
        self.event.history_nodes[particle.uniqueid] =  particle_node
        for parent in parents :
            self.event.history_nodes[parent].add_child(particle_node)
        
    def new_id(self):
        #new short id for the next reconstucted particle
        id = self.particlecounter
        self.particlecounter += 1
        return id


#class PFType(Enum):
    #NONE  =  0
    #TRACK  =  1
    #ECAL  =  2
    #HCAL  =  4
    #PARTICLE = 8
    #REConstructedPARTICLE = 16
    
#class PFDISTANCEtype(Enum):
    #NONE  =  0
    #TRACK_TRACK  =  2
    #TRACK_ECAL  =  3
    #ECAL_ECAL  =  4
    #TRACK_HCAL = 5
    #ECAL_HCAL  =  6
    #HCAL_HCAL = 8
           
class DistanceItem(object):
    '''Concrete distance calculator using an integer id to determine distance
    ''' 
    def __call__(self, ele1, ele2):
        '''ele1, ele2 two elements to find distance between
            returns a tuple: 
          Link_type set to None for this test
          True/False depending on the validity of the link
          float      the link distance
        '''
        distance = abs(ele1.id%100 -ele2.id%100)
        return  None, distance == 0, distance        

#will be the ruler in the event class
distance  =  DistanceItem()
         
        
class TestBlockReconstruction(unittest.TestCase):
    
        
    def test_1(self):
        
        event  =  Event(distance)
        sim  =  Simulator(event)
        event=sim.event
        
        pfblocker = PFBlockBuilder( event, distance, event.history_nodes)
        
        event.blocks = pfblocker.blocks
        #event.history_nodes = pfblocker.history_nodes
        
        
        ##test block splitting
        #blockids = []
        #unlink=[]
        #for b in event.blocks.itervalues():
            #ids=b.element_uniqueids
            #if len(ids)==3 :
                #print ids[0], ids[2]
                #unlink.append(b.edges[Edge.make_key(ids[0], ids[2])])
                #unlink.append(b.edges[Edge.make_key(ids[0], ids[1])])
                #print unlink
                #splitter=BlockSplitter(b,unlink,event.history_nodes)
                #print splitter.blocks
        
        #blocksplitter=BlockSplitter()
       
        rec  =  Reconstructor(event)

        
        # What is connected to HCAL 202 node?
        #  (1) via history_nodes
        #  (2) via reconstructed node links
        #  (3) Give me all blocks with  one track:
        #  (4) Give me all simulation particles attached to each reconstructed particle
        nodeid = 202
        nodeuid = sim.UID(nodeid)
        
        #(1) what is connected to the the HCAL CLUSTER
        ids = []
        BFS  =  BreadthFirstSearchIterative(event.history_nodes[nodeuid],"undirected")
        for n in BFS.result :
            ids.append(n.get_value())
         
        #1b WHAT BLOCK Does it belong to   
        x = None
        for id in ids:
            if Identifier.is_block(id) and event.blocks[id].short_name()== "E1H1T1":
                x =  event.blocks[id]     
                
        #1c #check that the block contains the expected list of suspects    
        pids = [] 
        for n in x.element_uniqueids:
            pids.append(n)              
        ids  = sorted(pids)
        expected_ids = sorted([sim.UID(2), sim.UID(102),sim.UID(202)])
        self.assertEqual(ids,expected_ids )
    
        #(2) use edge nodes to see what is connected
        ids = []
        BFS  =  BreadthFirstSearchIterative(pfblocker.nodes[nodeuid],"undirected")
        for n in BFS.result :
            ids.append(n.get_value())
        expected_ids = sorted([sim.UID(2), sim.UID(102),sim.UID(202)])   
        self.assertEqual(sorted(ids), expected_ids)

        #(3) Give me all simulation particles attached to each reconstructed particle
        for rp in event.reconstructed_particles :
            ids=[]
            BFS  =  BreadthFirstSearchIterative(event.history_nodes[rp],"parents")    
            for n in BFS.result :
                z=n.get_value()
        
if __name__  ==  '__main__':        
    unittest.main()
