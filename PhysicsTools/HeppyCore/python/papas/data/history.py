from PhysicsTools.HeppyCore.papas.graphtools.DAG import Node, BreadthFirstSearchIterative
from PhysicsTools.HeppyCore.papas.data.identifier import Identifier

class History(object):
    '''   
       Object to assist with printing and reconstructing histories
       may need to be merged with pfevent
       only just started ...
    '''    
    def __init__(self, history_nodes, pfevent):
        #this information information needed to be able to unravel information based on a unique identifier
        self.history_nodes = history_nodes
        self.pfevent = pfevent
        
    def summary_of_linked_elems(self, id):
    
        #find everything that is linked to this id
        #and write a summary of what is found
        #the BFS search returns a list of the ids that are  connected to the id of interest
        BFS = BreadthFirstSearchIterative(self.history_nodes[id], "undirected")
       
        #collate the string descriptions
        track_descrips = []
        ecal_descrips = []
        hcal_descrips = []
        #sim_particle_descrips = []
        rec_particle_descrips = []
        block_descrips = []
        
        
        for n in BFS.result :
            z = n.get_value()
            obj = self.pfevent.get_object(z)
            descrip = obj.__str__()
           # if (Identifier.is_particle(z)):
            #    sim_particle_descrips.append(descrip)
            if (Identifier.is_block(z)):
                block_descrips.append(descrip)            
            elif (Identifier.is_track(z)):
                track_descrips.append(descrip)         
            elif (Identifier.is_ecal(z)):
                ecal_descrips.append(descrip)  
            elif (Identifier.is_hcal(z)):
                hcal_descrips.append(descrip)         
            elif (Identifier.is_rec_particle(z)):
                rec_particle_descrips.append(descrip)               
        
       
        print "history connected to node:", id
        print "block", block_descrips
       
        print "       tracks", track_descrips
        print "        ecals", ecal_descrips
        print "        hcals", hcal_descrips
        print "rec particles", rec_particle_descrips
        
       