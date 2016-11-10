import itertools
from PhysicsTools.HeppyCore.papas.graphtools.graphbuilder import GraphBuilder
from PhysicsTools.HeppyCore.papas.graphtools.edge import Edge
from PhysicsTools.HeppyCore.papas.graphtools.DAG import Node
from PhysicsTools.HeppyCore.papas.pfobjects import MergedCluster
from PhysicsTools.HeppyCore.papas.data.identifier import Identifier
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger

class MergedClusterBuilder(GraphBuilder):
    ''' MergingBlockBuilder takes particle flow elements of one cluster type eg ecal_in
        and uses the distances between elements to construct a set of blocks ( of connected clusters)
        The blocks will contain overlapping clusters and then be used to merge the clusters
        
        attributes:
             merged - the dictionary of merged clusters
        
        Usage example:
             (will return the merged clusters to the event)
            event.ecal_clusters =  MergingBlockBuilder(event.ecal_clusters, ruler).merged
            
    '''
    def __init__(self, clusters, ruler, history_nodes = None):
        '''
        clusters a dictionary : {id1:ecal1, id2:ecal2, ...}
        ruler is something that measures distance between two objects eg track and hcal
            (see Distance class for example)
            it should take the two objects as arguments and return a tuple
            of the form
                link_type = 'ecal_ecal', 'ecal_track' etc
                is_link = true/false
                distance = float
        hist_nodes is an optional dictionary of Nodes : { id:Node1, id: Node2 etc}
            it could for example contain the simulation history nodes
            A Node contains the id of an item (cluster, track, particle etc)
            and says what it is linked to (its parents and children)
            if hist_nodes is provided it will be added to with the new block information
            If hist_nodes is not provided one will be created, it will contain nodes
            corresponding to each of the tracks, ecal etc and also for the blocks that
            are created by the event block builder.
        '''
        self.clusters = clusters
        
        # the merged clusters will be stored here
        self.merged = dict()

        # collate ids of clusters
        uniqueids = list(clusters.keys())         
             
        #make the edges match cpp by using the same approach as cpp
        edges = dict()
        for obj1 in  clusters.values():
            for obj2 in  clusters.values():
                if obj1.uniqueid < obj2.uniqueid :
                    link_type, is_linked, distance = ruler(obj1, obj2)
                    edge = Edge(obj1.uniqueid, obj2.uniqueid, is_linked, distance)
                    #the edge object is added into the edges dictionary
                    edges[edge.key] = edge

        #make the subgraphs of clusters
        super(MergedClusterBuilder, self).__init__(uniqueids, edges)
        
        #make sure we use the original history and update it as needed
        self.history_nodes = history_nodes

        self._make_merged_clusters()

    def _make_merged_clusters(self):
        #carry out the merging of linked clusters
        for subgraphids in self.subgraphs:
            subgraphids.sort()
            first = None
            supercluster =None
            snode = None
            for elemid in subgraphids :
                if not first:
                    first = elemid
                    supercluster = MergedCluster(self.clusters[elemid])
                    self.merged[supercluster.uniqueid] = supercluster;
                    if (self.history_nodes) :
                        snode = Node(supercluster.uniqueid)
                        self.history_nodes[supercluster.uniqueid] = snode 
                else:
                    thing = self.clusters[elemid]
                    supercluster += thing
                if (self.history_nodes) :
                    self.history_nodes[elemid].add_child(snode)
                pdebugger.info('Merged Cluster from {}\n'.format(self.clusters[elemid]))
            pdebugger.info(str('Made {}\n'.format(supercluster)))
