from PhysicsTools.HeppyCore.papas.data.identifier import Identifier
from PhysicsTools.HeppyCore.papas.graphtools.DAG import Node
from PhysicsTools.HeppyCore.papas.pfalgo.pfblocksplitter import BlockSplitter
from PhysicsTools.HeppyCore.papas.pdt import particle_data
from PhysicsTools.HeppyCore.papas.path import StraightLine, Helix
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger
from PhysicsTools.HeppyCore.papas.pfobjects import Particle
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger

from ROOT import TVector3, TLorentzVector
import math
import pprint

#Discuss with colin self.locked vs ecal.locked
#209 in reconstruct_block extra ecals to be added in
#remove sort


class PFReconstructor(object):
    ''' The reconstructor takes an event containing blocks of elements
        and attempts to reconstruct particles
        The following strategy is used (to be checked with Colin)
        single elements:
             track -> charged hadron
             hcal  -> neutral hadron
             ecal  -> photon
        connected elements:
              has more than one hcal
                -> each hcal is treated using rules below
              has an hcal with one or more connected tracks
                -> add up all connected track energies, turn each track into a charged hadron
                -> add up all ecal energies connected to the above tracks
                -> if excess = hcal energy + ecal energies - track energies > 0
                       and excess < ecal energies
                           then turn the excess into an photon
                -> if excess > 0 and excess > ecal energies
                          make a neutral hadron with excess- ecal energies
                          make photon with ecal energies
              has hcal but no track (nb by design there will be no attached ecals because hcal ecal links have been removed
                    so this will equate to single hcal:- that two hcals should not occur as a single block
                    because if they are close enough to be linked then they should already have been merged)
                -> make a neutral hadron 
              
              has track(s) 
                -> each track is turned into a charged hadron
              has track(s) and  ecal(s)
                -> the tracks are turned into charged hadrons, the ecals are marked as locked but energy is not checked 
                and no photons are made
                TODO handle case where there is more energy in ecals than in the track and make some photons
              has only ecals 
                -> this should not occur because ecals that are close enough to be linked should already have been merged
        
             
         If history_nodes are provided then the particles are linked into the exisiting history
         
         Contains:
            blocks: the dictionary of blocks to be reconstructed { blockid; block }
            unused: list of unused elements
            particles: list of constructed particles
            history_nodes: optional, desribes links between elements, blocks, particles
         Example usage:
         
              reconstructed = PFReconstructor(event)
              event.reconstructed_particles= sorted( reconstructed.particles,
                            key = lambda ptc: ptc.e(), reverse=True)
              event.history_nodes=reconstructed.history_nodes
        ''' 
    
    def __init__(self,  detector, logger):
        self.detector = detector
        self.log = logger
    #self.reconstruct(links)


    def reconstruct(self, event,  blocksname, historyname):
        '''arguments event: should contain blocks and optionally history_nodes'''
        self.blocks = getattr(event,  blocksname)
        self.unused = []
        self.particles = dict()
        
        
        # history nodes will be used to connect reconstructed particles into the history
        # its optional at the moment
        if hasattr(event, historyname):
            self.history_nodes = event.history_nodes
        else : 
            self.history_nodes = None
        
        # simplify the blocks by editing the links so that each track will end up linked to at most one hcal
        # then recalculate the blocks
        splitblocks=dict() 
        
        for block in self._sorted_block_keys(): #put big interesting blocks first
            #print "block: ", len(self.blocks[block]),  self.blocks[block].short_name();
            newblocks=self.simplify_blocks(self.blocks[block], self.history_nodes)
            if newblocks != None:
                splitblocks.update( newblocks)      
        if len(splitblocks):
            self.blocks.update(splitblocks)
        
            
        #reconstruct each of the resulting blocks        
        for b in self._sorted_block_keys():  #put big interesting blocks first
            block=self.blocks[b]
            if block.is_active: # when blocks are split the original gets deactivated                
                #ALICE debugging
                #if len(block.element_uniqueids)<6:
                #    continue
                pdebugger.info('Processing {}'.format(block))
                self.reconstruct_block(block)                
                self.unused.extend( [id for id in block.element_uniqueids if not self.locked[id]])
                
        #check if anything is unused
        if len(self.unused):
            self.log.warning(str(self.unused))
        self.log.info("Particles:")
        self.log.info(str(self))        
            
      
    def _sorted_block_keys(self) :
        #sort blocks (1) by number of elements (2) by mix of ecal, hcal , tracks (the shortname will look like "H1T2" for a block
        #Alice temporary to match cpp
        #return sorted(self.blocks.keys(), key=lambda k: (len(self.blocks[k].element_uniqueids), self.blocks[k].short_name()),reverse =True)
        #newsort
        return sorted(self.blocks.keys());
            
    def simplify_blocks(self, block, history_nodes=None):
        
        ''' Block: a block which contains list of element ids and set of edges that connect them
            history_nodes: optional dictionary of Nodes with element identifiers in each node
        
        returns None or a dictionary of new split blocks
            
        The goal is to remove, if needed, some links from the block so that each track links to 
        at most one hcal within a block. In some cases this may separate a block into smaller
        blocks (splitblocks). The BlockSplitter is used to return the new smaller blocks.
         If history_nodes are provided then the history will be updated. Split blocks will 
         have the tracks and cluster elements as parents, and also the original block as a parent
        '''
        
        ids=block.element_uniqueids
        
        
        if len(ids)<=1 :    #no links to remove
            return  None    
        
        # work out any links that need to be removed    
        #   - for tracks unink all hcals except the closest hcal
        #   - for ecals unlink hcals
        to_unlink = []        
        for id in ids :
            if Identifier.is_track(id):
                linked = block.linked_edges(id,"hcal_track") # NB already sorted from small to large distance
                if linked!=None and len(linked)>1 :
                    first_hcal = True
                    for elem in linked:
                        if first_hcal:
                            first_dist=elem.distance
                            first_hcal = False
                        else:
                            if (elem.distance==first_dist):
                                pass
                            to_unlink.append(elem)
        
        #if there is something to unlink then use the BlockSplitter        
        splitblocks=None        
        if len(to_unlink):
            splitblocks= BlockSplitter(block, to_unlink, history_nodes).blocks
        
        return splitblocks
            
    def reconstruct_block(self, block):
        ''' see class description for summary of reconstruction approach
        '''
        particles = dict()
        ids = block.element_uniqueids
        #ids =  sorted( ids,  key = lambda id: Identifier.type_short_code ) 
        self.locked = dict()
        for id in ids:
            self.locked[id] = False
        
        self.debugprint = False
        if (self.debugprint  and len(block.element_uniqueids)> 4):
            print  block
            
       
        if len(ids) == 1: #TODO WARNING!!! LOTS OF MISSING CASES
            id = ids[0]
            
            if Identifier.is_ecal(id):
                self.insert_particle(block, self.reconstruct_cluster(block.pfevent.ecal_clusters[id],"ecal_in"))
                
            elif Identifier.is_hcal(id):
                self.insert_particle(block, self.reconstruct_cluster(block.pfevent.hcal_clusters[id],"hcal_in"))
                
            elif Identifier.is_track(id):
                self.insert_particle(block, self.reconstruct_track(block.pfevent.tracks[id]))
                # ask Colin about energy balance - what happened to the associated clusters that one would expect?
        else: #TODO
            for id in sorted(ids) : #newsort
                if Identifier.is_hcal(id):
                    self.reconstruct_hcal(block,id)
            for id in sorted(ids) : #newsort
                if Identifier.is_track(id) and not self.locked[id]:
                # unused tracks, so not linked to HCAL
                # reconstructing charged hadrons.
                # ELECTRONS TO BE DEALT WITH.
                    self.insert_particle(block, self.reconstruct_track(block.pfevent.tracks[id]))
                    
                    # tracks possibly linked to ecal->locking cluster
                    for idlink in block.linked_ids(id,"ecal_track"):
                        #ask colin what happened to possible photons here:
                        self.locked[idlink] = True
                        #TODO add in extra photonsbut decide where they should go?
                        
                        
            # #TODO deal with ecal-ecal
            # ecals = [elem for elem in group if elem.layer=='ecal_in'
            #          and not elem.locked]
            # for ecal in ecals:
            #     linked_layers = [linked.layer for linked in ecal.linked]
            #     # assert('tracker' not in linked_layers) #TODO electrons
            #     self.log.warning( 'DEAL WITH ELECTRONS!' ) 
            #     particles.append(self.reconstruct_cluster(ecal, 'ecal_in'))
            #TODO deal with track-ecal
          
    
       
    def insert_particle(self, block, newparticle):
            ''' The new particle will be inserted into the history_nodes (if present).
                A new node for the particle will be created if needed.
                It will have as its parents the block and all the elements of the block.
                '''        
            #Note that although it may be possible to specify more closely that the particle comes from
            #some parts of the block, there are frequently ambiguities and so for now the particle is
            #linked to everything in the block
            if (newparticle) :
                newid = newparticle.uniqueid
                self.particles[newid] = newparticle            
                
                #check if history nodes exists
                if (self.history_nodes == None):
                    return
                
                #find the node for the block        
                blocknode = self.history_nodes[block.uniqueid]
                
                #find or make a node for the particle            
                if newid  in self.history_nodes :
                    pnode = self.history_nodes[newid]
                else :
                    pnode = Node(newid)
                    self.history_nodes[newid] = pnode
                
                #link particle to the block            
                blocknode.add_child(pnode)
                #link particle to block elements
                for element_id in block.element_uniqueids:
                    self.history_nodes[element_id].add_child(pnode)    
    

    def neutral_hadron_energy_resolution(self, energy, eta):
        '''Currently returns the hcal resolution of the detector in use.
        That's a generic solution, but CMS is doing the following
        (implementation in commented code)
        http://cmslxr.fnal.gov/source/RecoParticleFlow/PFProducer/src/PFAlgo.cc#3350 
        '''
        resolution = self.detector.elements['hcal'].energy_resolution(energy, eta)
        return resolution
## energy = max(hcal.energy, 1.)
## eta = hcal.position.Eta()
##        stoch, const = 1.02, 0.065
##        if abs(hcal.position.Eta())>1.48:
##            stoch, const = 1.2, 0.028
##        resol = math.sqrt(stoch**2/energy + const**2)
##        return resol

    def nsigma_hcal(self, cluster):
        '''Currently returns 2.
        CMS is doing the following (implementation in commented code)
        http://cmslxr.fnal.gov/source/RecoParticleFlow/PFProducer/src/PFAlgo.cc#3365 
        '''
        return 2
## return 1. + math.exp(-cluster.energy/100.)
    
      
        
    def reconstruct_hcal(self, block, hcalid):
        '''
           block: element ids and edges 
           hcalid: id of the hcal being processed her
        
           has hcal and has a track
                -> add up all connected tracks, turn each track into a charged hadron
                -> add up all ecal energies
                -> if track energies is greater than hcal energy then turn the missing energies into an ecal (photon)
                      NB this links the photon to the hcal rather than the ecals
                -> if track energies are less than hcal then make a neutral hadron with rest of hcal energy and turn all ecals into photons
              has hcal but no track (nb by design there will be no attached ecals because hcal ecal links have been removed)
                -> make a neutral hadron
              has hcals
                -> each hcal is treated using rules above
        '''
        
        # hcal used to make ecal_in has a couple of possible issues
        tracks = []
        ecals = []
        hcal =block.pfevent.hcal_clusters[hcalid]
        
        assert(len(block.linked_ids(hcalid, "hcal_hcal"))==0  )

        #trackids =  block.linked_ids(hcalid, "hcal_track")
        #alice temporarily disabled
        #trackids =    block.sort_distance_energy(hcalid, trackids )
#newsort        
        trackids = block.linked_ids(hcalid, "hcal_track")  #sorted within block
        for trackid in trackids:
            tracks.append(block.pfevent.tracks[trackid])
            for ecalid in block.linked_ids(trackid, "ecal_track"): #new sort
                # the ecals get all grouped together for all tracks in the block
                # Maybe we want to link ecals to their closest track etc?
                # this might help with history work
                # ask colin.
                if not self.locked[ecalid]:
                    ecals.append(block.pfevent.ecal_clusters[ecalid])
                    self.locked[ecalid]  = True
                # hcal should be the only remaining linked hcal cluster (closest one)
                #thcals = [th for th in elem.linked if th.layer=='hcal_in']
                #assert(thcals[0]==hcal)
        self.log.info( hcal )
        self.log.info( '\tT {tracks}'.format(tracks=tracks) )
        self.log.info( '\tE {ecals}'.format(ecals=ecals) )
        hcal_energy = hcal.energy
        if len(tracks):
            ecal_energy = sum(ecal.energy for ecal in ecals)
            track_energy = sum(track.energy for track in tracks)
            for track in tracks:
                #make a charged hadron
                self.insert_particle(block, self.reconstruct_track( track))
                
            delta_e_rel = (hcal_energy + ecal_energy) / track_energy - 1.
            # WARNING
            # calo_eres = self.detector.elements['hcal'].energy_resolution(track_energy)
            # calo_eres = self.neutral_hadron_energy_resolution(hcal)
            calo_eres = self.neutral_hadron_energy_resolution(track_energy,
                                                              hcal.position.Eta())
            self.log.info( 'dE/p, res = {derel}, {res} '.format(
                derel = delta_e_rel,
                res = calo_eres ))
            # if False:
            if delta_e_rel > self.nsigma_hcal(hcal) * calo_eres: # approx means hcal energy + ecal energies > track energies
                
                excess = delta_e_rel * track_energy # energy in excess of track energies
                #print( 'excess = {excess:5.2f}, ecal_E = {ecal_e:5.2f}, diff = {diff:5.2f}'.format(
                #    excess=excess, ecal_e = ecal_energy, diff=excess-ecal_energy))
                if excess <= ecal_energy: # approx means hcal energy > track energies 
                    # Make a photon from the ecal energy
                    # We make only one photon using only the combined ecal energies
                    self.insert_particle(block, self.reconstruct_cluster(hcal, 'ecal_in',excess))
                    
                else: # approx means that hcal energy>track energies so we must have a neutral hadron
                    #excess-ecal_energy is approximately hcal energy  - track energies
                    self.insert_particle(block, self.reconstruct_cluster(hcal, 'hcal_in',
                                                        excess-ecal_energy))
                    if ecal_energy:
                        #make a photon from the remaining ecal energies
                        #again history is confusingbecause hcal is used to provide direction
                        #be better to make several smaller photons one per ecal?
                        self.insert_particle(block, self.reconstruct_cluster(hcal, 'ecal_in',
                                                                  ecal_energy))

        else: # case where there are no tracks make a neutral hadron for each hcal
              # note that hcal-ecal links have been removed so hcal should only be linked to 
              # other hcals
                 
            self.insert_particle(block,  self.reconstruct_cluster(hcal, 'hcal_in'))
            
        self.locked[hcalid] = True
        
                
    def reconstruct_cluster(self, cluster, layer, energy = None, vertex = None):
        '''construct a photon if it is an ecal
           construct a neutral hadron if it is an hcal
        '''        
        if vertex is None:
            vertex = TVector3()
        pdg_id = None
        if layer=='ecal_in':
            pdg_id = 22 #photon
        elif layer=='hcal_in':
            pdg_id = 130 #K0
        else:
            raise ValueError('layer must be equal to ecal_in or hcal_in')
        assert(pdg_id)
        mass, charge = particle_data[pdg_id]
        if energy is None:
            energy = cluster.energy
        if energy < mass: 
            return None 
        if (mass==0):
            momentum= energy #avoid sqrt for zero mass
        else:
            momentum = math.sqrt(energy**2 - mass**2)
        p3 = cluster.position.Unit() * momentum
        p4 = TLorentzVector(p3.Px(), p3.Py(), p3.Pz(), energy) #mass is not accurate here
        particle = Particle(p4, vertex, charge, pdg_id, Identifier.PFOBJECTTYPE.RECPARTICLE)
        path = StraightLine(p4, vertex)
        path.points[layer] = cluster.position #alice: this may be a bit strange because we can make a photon with a path where the point is actually that of the hcal?
                                            # nb this only is problem if the cluster and the assigned layer are different
        particle.set_path(path)
        particle.clusters[layer] = cluster  # not sure about this either when hcal is used to make an ecal cluster?
        self.locked[cluster.uniqueid] = True #just OK but not nice if hcal used to make ecal.
        pdebugger.info(str('Made {} from {}'.format(particle,  cluster)))
        return particle
        
    def reconstruct_track(self, track, clusters = None): # cluster argument does not ever seem to be used at present
        '''construct a charged hadron from the track
        '''
        vertex = track.path.points['vertex']
        pdg_id = 211 * track.charge
        mass, charge = particle_data[pdg_id]
        p4 = TLorentzVector()
        p4.SetVectM(track.p3, mass)
        particle = Particle(p4, vertex, charge, pdg_id, Identifier.PFOBJECTTYPE.RECPARTICLE)
        particle.set_path(track.path)
        particle.clusters = clusters
        self.locked[track.uniqueid] = True
        pdebugger.info(str('Made {} from {}'.format(particle,  track)))
        return particle


    def __str__(self):
        theStr = ['New Rec Particles:']
        theStr.extend( map(str, self.particles.itervalues()))
        theStr.append('Unused:')
        if len(self.unused)==0:
            theStr.append('None')
        else:
            theStr.extend( map(str, self.unused))
        return '\n'.join( theStr )
