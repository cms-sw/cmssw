from PhysicsTools.HeppyCore.papas.data.identifier import Identifier
class Edge(object): 
    '''An Edge stores end node ids, distance between the nodes, and whether they are linked
       
       attributes:
       
       id1 : element1 uniqueid generated from Identifier class
       id2 : element2 uniqueid generated from Identifier class
       key : unique key value created from id1 and id2 (order of id1 and id2 is not important) 
       distance: distance between two elements
       is_linked : boolean T/F
       edge_type : "hcal_track" "ecal_track" etc
    '''
    
    def __init__(self, id1, id2, is_linked, distance): 
        ''' The Edge knows the ids of its ends, the distance between the two ends and whether or not they are linked 
           id1 : element1 uniqueid generated from Identifier class
           id2 : element2 uniqueid generated from Identifier class
           is_linked : boolean T/F
           distance: distance between two elements
        '''
        self.id1 = id1
        self.id2 = id2
        self.distance = distance
        self.linked = is_linked
        self.edge_type = self._edge_type()
            
        #for reconstruction we do not use ecal-hcal links (may need to be moved if we use these edges for merging)
        if self.edge_type == "ecal_hcal":
            self.is_linked = False
        self.key = Edge.make_key(id1,id2)
    
    def _edge_type(self):
        ''' produces an edge_type string eg "ecal_track"
            the order of id1 an id2 does not matter, 
            eg for one track and one ecal the type will always be "ecal_track" (and never be a "track_ecal")         
        '''
        #consider creating an ENUM instead for the edge_type
        shortid1=Identifier.type_short_code(self.id1);
        shortid2=Identifier.type_short_code(self.id2);
        if shortid1 == shortid2:
            if shortid1 == "h":
                return "hcal_hcal"
            elif shortid1 == "e":
                return "ecal_ecal"
            elif shortid1 == "t":
                return "track_track"           
        elif (shortid1=="h" and shortid2=="t" or shortid1=="t" and shortid2=="h"):
            return "hcal_track"
        elif (shortid1=="e" and shortid2=="t" or shortid1=="t" and shortid2=="e"):
            return "ecal_track"  
        elif (shortid1=="e" and shortid2=="h" or shortid1=="h" and shortid2=="e"):
            return "ecal_hcal"  
        
        return "unknown"

    def __str__(self):
        ''' String descriptor of the edge
             for example:
             Edge: 3303164520272<->3303164436240 = No distance (link = False) 
        '''
        if self.distance==None:
            descrip = 'Edge: {id1:d}<->{id2:d} = No distance (link = {linked}) '.format(id1=self.id1,id2=self.id2,linked=self.linked)
        else :
            descrip = 'Edge: {id1}<->{id2} = {dist:8.4f} (link = {linked}) '.format(id1=self.id1,id2=self.id2,dist=self.distance,linked=self.linked)            
        return descrip
    
    def __repr__(self):
        return self.__str__()      
    
    @staticmethod 
    def make_key(id1,id2):
        '''method to create a key based on two ids that can then be used to retrieve a specific edge
        ''' 
        return hash(tuple(sorted([id1,id2])))

  