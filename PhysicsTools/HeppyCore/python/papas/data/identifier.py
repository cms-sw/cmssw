# simplified class to provide a unique identifier for each object
# could also add more information into the identifier as needed
from itertools import count
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger

class Identifier(long):
    '''the Identififier is a uniqueid that contains encoded information about an element
           for example, given an indentifier, we can determine that the element is an ecal_cluster
           and retrieve the cluster from a cluster dict.
    
        The Identifier class consists of a set of static methods that can be used
        to create and to dissect identifiers.
        
        The first(rightmost) 40 bits are used to contain the python unique objectid of the item
        The bits to the left of this contain the objecttype eg ECALCLUSTER etc
        
        usage:
           self.uniqueid = Identifier.make_id(Identifier.PFOBJECTTYPE.BLOCK) 
           if Identifier.is_track(self.uniqueid):
                ....
           
        '''    

    _id = count(1)


    class PFOBJECTTYPE:
        NONE = 0
        ECALCLUSTER = 1 #simplistic initial implementation (may need an enum for layer to be added)
        HCALCLUSTER = 2
        TRACK = 3
        PARTICLE = 4
        RECPARTICLE = 5
        BLOCK = 6
    
    @classmethod    
    def make_id(cls, type):
        x = cls._id.next()
        value = type <<32
        id = value | x
        assert (Identifier.get_unique_id(id) == x )
        assert (Identifier.get_type(id)==type)
        return id
   
    @staticmethod      
    def get_unique_id( ident):
        return ident & 0b11111111111111111111111111111111
    
    @staticmethod  
    def get_type ( ident):
        return ident >> 32
    
    @staticmethod  
    def is_ecal ( ident):
        return Identifier.get_type(ident)  == Identifier.PFOBJECTTYPE.ECALCLUSTER  

    @staticmethod  
    def is_hcal ( ident):
        return Identifier.get_type(ident)  == Identifier.PFOBJECTTYPE.HCALCLUSTER  

    @staticmethod  
    def is_track ( ident):
        return Identifier.get_type(ident)  == Identifier.PFOBJECTTYPE.TRACK 
    
    @staticmethod  
    def is_block ( ident):
        return Identifier.get_type(ident)  == Identifier.PFOBJECTTYPE.BLOCK     
    
    @staticmethod  
    def is_rec_particle ( ident):
        return Identifier.get_type(ident)  == Identifier.PFOBJECTTYPE.RECPARTICLE 
    
    @staticmethod  
    def is_particle ( ident):
        return Identifier.get_type(ident)  == Identifier.PFOBJECTTYPE.PARTICLE     

    @staticmethod
    def type_short_code(ident):
        ''' Returns code
              e = ecal
              h = hcal
              t = track
              p = particle
              r = reconstructed particle (this will soon go)
              b = block
        '''
        typelist=".ehtprb..." #the enum value (0 to 8) will index into this and return E is it is ECAL etc
        return typelist[Identifier.get_type(ident)]

    @staticmethod
    def pretty(ident):
        return Identifier.type_short_code(ident) + str(Identifier.get_unique_id(ident))
    
    @classmethod
    def reset(cls):
        cls._id=count(1)
        pdebugger.info("reset ID")
        return
