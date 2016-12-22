import itertools
from PhysicsTools.HeppyCore.papas.graphtools.edge import Edge
from PhysicsTools.HeppyCore.papas.data.identifier import Identifier

#todo remove pfevent from this class once we have written a helper class to print the block and its elements

class PFBlock(object):

    ''' A Particle Flow Block stores a set of element ids that are connected to each other
     together with the edge data (distances) for each possible edge combination

     attributes:

     uniqueid : the block's unique id generated from Identifier class
     element_uniqueids : list of uniqueids of its elements
     pfevent : contains the tracks and clusters and a get_object method to allow access to the
               underlying objects given their uniqueid
     edges : Dictionary of all the edge cominations in the block dict{edgekey : Edge}
             use  get_edge(id1,id2) to find an edge
     is_active : bool true/false, set to false if the block is subsequently subdivided

     Usage:
            block = PFBlock(element_ids,  edges, pfevent)
            for uid in block.element_uniqueids:
                 print pfevent.get_object(uid).__str__() + "\n"
     '''

    temp_block_count = 0 #sequential numbering of blocks, not essential but helpful for debugging

    def __init__(self, element_ids, edges, pfevent):
        '''
            element_ids:  list of the uniqueids of the elements to go in this block [id1,id2,...]
            edges: is a dictionary of edges, it must contain at least all needed edges.
                   It is not a problem if it contains
                   additional edges as only the ones needed will be extracted
            pfevent: allows access to the underlying elements given a uniqueid
                     must provide a get_object function
        '''
        #make a uniqueid for this block
        self.uniqueid = Identifier.make_id(Identifier.PFOBJECTTYPE.BLOCK)
        self.is_active = True # if a block is subsequently split it will be deactivated

        #allow access to the underlying objects
        self.pfevent = pfevent

        #comment out energy sorting  for now as not available C++
        sortby = lambda x: Identifier.type_short_code(x)
        self.element_uniqueids = sorted(element_ids, key=sortby)
        #sequential numbering of blocks, not essential but helpful for debugging
        self.block_count = PFBlock.temp_block_count
        PFBlock.temp_block_count += 1

        #extract the relevant parts of the complete set of edges and store this within the block
        self.edges = dict()
        for id1, id2 in itertools.combinations(self.element_uniqueids, 2):
            key = Edge.make_key(id1, id2)
            self.edges[key] = edges[key]


    def count_ecal(self):
        ''' Counts how many ecal cluster ids are in the block '''
        count = 0
        for elem in self.element_uniqueids:
            count += Identifier.is_ecal(elem)
        return count

    def count_tracks(self):
        ''' Counts how many track ids are in the block '''
        count = 0
        for elem in self.element_uniqueids:
            count += Identifier.is_track(elem)
        return count

    def count_hcal(self):
        ''' Counts how many hcal cluster ids are in the block '''
        count = 0
        for elem in self.element_uniqueids:
            count += Identifier.is_hcal(elem)
        return count

    def __len__(self) :
        return len(self.element_uniqueids)

    def linked_edges(self, uniqueid, edgetype=None) :
        '''
        Returns list of all edges of a given edge type that are connected to a given id.
        The list is sorted in order of increasing distance

        Arguments:
        uniqueid : is the id of item of interest
        edgetype : is an optional type of edge. If specified only links of the given edgetype will be returned
        '''
        linked_edges = []
        for edge in self.edges.itervalues():
            if edge.linked and (edge.id1 == uniqueid or edge.id2 == uniqueid):
                if edgetype is None or ((edgetype != None) and (edge.edge_type == edgetype)):
                    linked_edges.append(edge)


        #this is a bit yucky and temporary solution as need to make sure the order returned is consistent
        # maybe should live outside of this class
        linked_edges.sort(key=lambda x: (x.distance is None, x.distance))
        return linked_edges

    def linked_ids(self, uniqueid, edgetype=None) :
        ''' Returns list of all linked ids of a given edge type that are connected to a given id -
            sorted in order of increasing distance'''
        linked_ids = []
        linked_edges = []
        linked_edges = self.linked_edges(uniqueid, edgetype)
        if len(linked_edges):
            for edge in linked_edges:
                if edge.id1 == uniqueid:
                    linked_ids.append(edge.id2)
                else:
                    linked_ids.append(edge.id1)
        return sorted(linked_ids)

    def sort_distance_energy(self, uniqueid, otherids):
        ''' returns a list of the otherids sorted by distance to uniqueid and by decreasing energies

            eg if uniqueid is an hcal
               and other ids are  track1 energy = 18, dist to hcal = 0.1
                                  track2 energy = 9,  dist to hcal = 0
                                  track3 energy = 4,  dist to hcal = 0
            this will return {track2, track3, track1}
            '''
        #this is ""needed"" for particle reconstruction
        #this is a bit yucky and may only be a  temporary work around
        # maybe should live outside of this class
        return sorted(otherids, key=lambda x: (self.get_edge(x, uniqueid).distance is None,
                                               self.get_edge(x, uniqueid).distance,
                                               -self.pfevent.get_object(x).energy))

    def elements_string(self):
        ''' Construct a string description of each of the elements in a block:-
        The elements are given a short name E/H/T according to ecal/hcal/track
        and then sequential numbering starting from 0, this naming is also used to index the
        matrix of distances. The full unique id is also given.
        For example:-
        elements: {
        E0:1104134446736:SmearedCluster : ecal_in       0.57  0.33 -2.78
        H1:2203643940048:SmearedCluster : hcal_in       6.78  0.35 -2.86
        T2:3303155568016:SmearedTrack   :    5.23    4.92  0.34 -2.63
        }
        '''
        count = 0
        elemdetails = "\n      elements: {\n"
        for uid in self.element_uniqueids:
            elemdetails += "      {shortname}{count}:{strdescrip}\n".format(
                shortname=Identifier.type_short_code(uid),
                count=count,
                strdescrip=self.pfevent.get_object(uid).__str__())
            count = count + 1
        return elemdetails + "      }\n"

    def short_elements_string(self):
        ''' Construct a string description of each of the elements in a block.

        The elements are given a short name E/H/T according to ecal/hcal/track
        and then sequential numbering starting from 0, this naming is also used to index the
        matrix of distances. The full unique id is also given.
        For example:-
        elements: {
        E0:1104134446736:SmearedCluster : ecal_in       0.57  0.33 -2.78
        H1:2203643940048:SmearedCluster : hcal_in       6.78  0.35 -2.86
        T2:3303155568016:SmearedTrack   :    5.23    4.92  0.34 -2.63
        }
        '''

        count = 0
        elemdetails = "    elements:\n"
        for uid in self.element_uniqueids:
            elemdetails += "{shortname:>7}{count} = {strdescrip:9} ({id})\n".format(
                shortname=Identifier.type_short_code(uid),
                count=count,
                strdescrip=Identifier.pretty(uid),
                id=uid)
            count = count + 1
        return elemdetails

    def short_name(self):
        ''' constructs a short summary name for blocks allowing sorting based on contents
            eg 'E1H1T2' for a block with 1 ecal, 1 hcal, 2 tracks
        '''
        shortname = ""
        if self.count_ecal():
            shortname = shortname + "E" + str(self.count_ecal())
        if self.count_hcal():
            shortname = shortname + "H" + str(self.count_hcal())
        if self.count_tracks():
            shortname = shortname + "T" + str(self.count_tracks())
        return shortname

    def edge_matrix_string(self):
        ''' produces a string containing the the lower part of the matrix of distances between elements
        elements are ordered as ECAL(E), HCAL(H), Track(T)
        for example:-

        distances:
                  E0       H1       T2       T3
         E0       .
         H1  0.0267        .
         T2  0.0000   0.0000        .
         T3  0.0287   0.0825      ---        .
         '''

        # make the header line for the matrix
        count = 0
        matrixstr = "\n"
        if len(self.element_uniqueids) > 1:
            matrixstr = "    distances:\n        "
            for e1 in self.element_uniqueids :
                # will produce short id of form E2 H3, T4 etc in tidy format
                elemstr = Identifier.type_short_code(e1) + str(count)
                matrixstr += "{:>8}".format(elemstr)
                count += 1
            matrixstr += "\n"

            #for each element find distances to all other items that are in the lower part of the matrix
            countrow = 0
            for e1 in self.element_uniqueids : # this will be the rows
                countcol = 0
                rowstr = ""
                #make short name for the row element eg E3, H5 etc
                rowname = Identifier.type_short_code(e1) +str(countrow)
                for e2 in self.element_uniqueids:  # these will be the columns
                    countcol += 1
                    if e1 == e2:
                        rowstr += "       ."
                        break
                    elif self.get_edge(e1, e2).distance is None:
                        rowstr += "     ---"
                    elif not self.get_edge(e1, e2).linked:
                        rowstr += "     xxx"
                    else :
                        rowstr += "{:8.4f}".format(self.get_edge(e1, e2).distance)
                matrixstr += "{:>8}".format(rowname) + rowstr + "\n"
                countrow += 1
        return matrixstr

    def get_edge(self, id1, id2):
        ''' Find the edge corresponding to e1 e2
            Note that make_key deals with whether it is get_edge(e1, e2) or
                                                        get_edge(e2, e1) (either order gives same result)
            '''
        return self.edges[Edge.make_key(id1, id2)]

    def __str__(self):
        ''' Block description which includes list of elements and a matrix of distances
        Example:
        block: E1H1T1       id=  39 :uid= 6601693505424: ecals = 1 hcals = 1 tracks = 1
            elements: {
            E0:1104134446736:SmearedCluster : ecal_in       0.57  0.33 -2.78
            H1:2203643940048:SmearedCluster : hcal_in       6.78  0.35 -2.86
            T2:3303155568016:SmearedTrack   :    5.23    4.92  0.34 -2.63
            }
            distances:
                        E0       H1       T2
               E0       .
               H1  0.0796        .
               T2  0.0210   0.0000        .
            }
        '''
        description = self.__repr__() + "\n"
        description += self.short_elements_string()
        description += self.edge_matrix_string()
        return description

    def __repr__(self):
        ''' Short Block description
        '''
        if self.is_active:
            description = "block:"
        else:
            description = "deactivated block:"
        description += str('{shortname:8} :{prettyid:6}: ecals = {count_ecal} hcals = {count_hcal} tracks = {count_tracks}'.format(
            shortname=self.short_name(),
            prettyid=Identifier.pretty(self.uniqueid),
            count_ecal=self.count_ecal(),
            count_hcal=self.count_hcal(),
            count_tracks=self.count_tracks())
                           )
        return description

