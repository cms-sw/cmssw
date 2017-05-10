#todo remove pfevent from this class once we have written a helper class to print the block and its elements
from DAG import Node, DAGFloodFill
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger
from PhysicsTools.HeppyCore.papas.data.identifier import Identifier
import collections
#todo remove pfevent from this class once we have written a helper class to print the block and its elements


class GraphBuilder(object):
    ''' GraphBuilder takes a set of identifiers and a dict of associated edges which have distance and link info
        It uses the distances between elements to construct a set of subgraphs
        Each element will end up in one (and only one) subgraph
        
        attributes:

        ids   : list of unique identifiers eg of tracks, clusters etc
        edges : dict of edges which contains all edges between the ids (and maybe more)
                an edge records the distance between two ids
        nodes : a set of nodes corresponding to the unique ids which is used to construct a graph
                and thus find distinct blocks
        subgraphs : a list of subgraphs, each subgraph is a list of connected ids

        Usage example:
            graph = GraphBuilder(ids, edges)
            
    '''
    def __init__(self, ids, edges):
        '''
        ids   : list of unique identifiers eg of tracks, clusters etc
        edges : dict of edges which contains all edges between the ids (and maybe more)
                an edge records the distance/link between two ids
        '''
        self.ids = ids
        self.edges = edges

        # build the block nodes (separate graph which will use distances between items to determine links)
        self.nodes = dict((idt, Node(idt)) for idt in ids)
        for edge in edges.itervalues():
            #add linkage info into the nodes dictionary
            if  edge.linked: #this is actually an undirected link - OK for undirected searches 
                self.nodes[edge.id1].add_child(self.nodes[edge.id2])

        # build the subgraphs of connected nodes
        self.subgraphs = []
        for subgraphlist in DAGFloodFill(self.nodes).blocks: # change to subgraphs
            element_ids = [] 
            # NB the nodes that are found by FloodFill are the Nodes describing links between items
            # we want the ids of these nodes
            for node in subgraphlist:
                element_ids.append(node.get_value())        
            self.subgraphs.append(sorted(element_ids)) #newsort

    def __str__(self):
        descrip = "{ "
        
        for subgraph in  self.subgraphs:
            descrip =  descrip +  " ("
            for elemid in  subgraph:
                descrip = descrip + str(elem) +  " "
            descrip =  descrip +  " )"    
        descrip = descrip + "}\n"
        return descrip  

    def __repr__(self):
        return self.__str__()     