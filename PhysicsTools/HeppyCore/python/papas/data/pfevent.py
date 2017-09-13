from PhysicsTools.HeppyCore.papas.data.identifier import Identifier



class PFEvent(object):
    '''PFEvent is used to  allow addition of a function get_object to an Event class
       get_object() allows a cluster or track to be found from its id
       May want to merge this with the history class
       
       attributes:
          tracks is a dictionary : {id1:track1, id2:track2, ...}
          ecal is a dictionary : {id1:ecal1, id2:ecal2, ...}
          hcal is a dictionary : {id1:hcal1, id2:hcal2, ...}
          blocks = optional dictionary of blocks : {id1:block1, id2:block2, ...}
       
       usage: 
          pfevent=PFEvent(event, self.tracksname,  self.ecalsname,  self.hcalsname,  self.blocksname) 
          obj1 = pfevent.get_object(id1)
    ''' 
    def __init__(self, event,  tracksname = 'tracks', ecalsname = 'ecal_clusters',  hcalsname = 'hcal_clusters',  blocksname = 'blocks',
                 sim_particlesname = "None",  rec_particlesname = "reconstructed_particles"):    
        '''arguments
             event: must contain
                  tracks dictionary : {id1:track1, id2:track2, ...}
                  ecal dictionary : {id1:ecal1, id2:ecal2, ...}
                  hcal dictionary : {id1:hcal1, id2:hcal2, ...}
                  
                  and these must be names according to ecalsname etc
                  blocks, sim_particles and rec_particles are optional
                  '''            
        self.tracks = getattr(event, tracksname)
        self.ecal_clusters = getattr(event, ecalsname)
        self.hcal_clusters = getattr(event, hcalsname)
        
        self.blocks = []
        if hasattr(event, blocksname):
            self.blocks =  getattr(event, blocksname)
        if hasattr(event,sim_particlesname): 
            self.sim_particles= getattr(event, sim_particlesname)
        if hasattr(event,rec_particlesname): #todo think about naming
            self.reconstructed_particles= getattr(event, rec_particlesname)                       
    
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


