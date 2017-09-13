from PhysicsTools.HeppyCore.papas.graphtools.DAG import Node, DAGFloodFill
from PhysicsTools.HeppyCore.papas.pfalgo.pfblock import PFBlock
from PhysicsTools.HeppyCore.papas.graphtools.graphbuilder import GraphBuilder
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger

#todo remove pfevent from this class once we have written a helper class to print the block and its elements

        
class BlockBuilder(GraphBuilder):
    ''' BlockBuilder takes a set of identifiers and a dict of associated edges which have distance and link info
        It uses the distances between elements to construct a set of blocks
        Each element will end up in one (and only one block)
        Blocks retain information of the elements and the distances between elements
        The blocks can then be used for future particle reconstruction
        The ids must be unique and are expected to come from the Identifier class
        
        attributes:
        
        ids   : list of unique identifiers eg of tracks, clusters etc
        edges : dict of edges which contains all edges between the ids (and maybe more)
                an edge records the distance between two ids
        history_nodes : dictionary of nodes that describe which elements are parents of which blocks 
                        if a history_nodes tree is passed in then 
                        the additional history will be added into the exisiting history 
        pfevent : the particle flow event object which is needed so that the underlying object can 
                be retrieved
        nodes : a set of nodes corresponding to the unique ids which is used to construct a graph
                and thus find distinct blocks
        blocks: the resulting blocks
    
        
        Usage example:

            builder = BlockBuilder(ids, edges, history_nodes, pfevent)
            for b in builder.blocks.itervalues() :
                print b
    '''
    def __init__(self, ids, edges, history_nodes = None, pfevent = None):
        '''
        ids   : list of unique identifiers eg of tracks, clusters etc
        edges : dict of edges which contains all edges between the ids (and maybe more)
                an edge records the distance/link between two ids
        history_nodes : optional dictionary of nodes that describe which elements are parents of which blocks 
                        if a history_nodes tree is passed in then 
                        the additional history will be added into the exisiting history 
        pfevent : particle flow event object  needed so that the underlying object can 
                be retrieved
       
        '''
        
        #given a unique id this can return the underying object
        self.pfevent = pfevent
        self.history_nodes = history_nodes
        
        super(BlockBuilder, self).__init__(ids, edges)       

        # build the blocks of connected nodes
        self.blocks = dict()
        self._make_blocks()        
        
        
    
        
    def _make_blocks (self) :
        ''' uses the DAGfloodfill algorithm in connection with the BlockBuilder nodes
            to work out which elements are connected
            Each set of connected elements will be used to make a new PFBlock
        ''' 
        for subgraph in self.subgraphs:
            #make the block
            block = PFBlock(subgraph,  self.edges, self.pfevent)        
            pdebugger.info("Made {}".format(block))
            #put the block in the dict of blocks            
            self.blocks[block.uniqueid] = block        
            
            #make a node for the block and add into the history Nodes
            if (self.history_nodes != None):
                blocknode = Node(block.uniqueid)
                self.history_nodes[block.uniqueid] = blocknode
                #now add in the links between the block elements and the block into the history_nodes
                for elemid in block.element_uniqueids:
                    self.history_nodes[elemid].add_child(blocknode)
        
                     
    def __str__(self):
        descrip = "{ "
        #for block in self.blocks.iteritems():
        for block in   sorted(self.blocks, key = lambda k: (len(self.blocks[k].element_uniqueids), self.blocks[k].short_name()),reverse =True):            
            descrip = descrip + self.blocks[block].__str__()
           
        descrip = descrip + "}\n"
        return descrip  
    
    def __repr__(self):
        return self.__str__()     