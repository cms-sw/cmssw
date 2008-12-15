class RelativeObject(object):
    """ Object that holds relation information """
    def __init__(self,mother=None,options=[]):
        """ constructor """
        self.mothers=[]
        self.daughters=[]
        self.flatrelations=[]
        self.sinks=[]
        self.sources=[]
        self.options=options
        if mother!=None:
            self.addMother(mother)

    def addMother(self,mother):
        """ Set mother object """
        self.mothers+=[mother]
        if mother!=None:
            mother.daughters+=[self]
    
    def addFlat(self,flat):
        """ Set mother object """
        self.flatrelations+=[flat]
        if flat!=None:
            flat.flatrelations+=[self]

    def getFirstMother(self):
        if self.mothers!=[]:
            return self.mothers[0]
        else:
            return None
    
    def getAdditionalMothers(self):
        first=True
        mothers=[]
        for mother in self.mothers:
            if not first:
                mothers+=[mother]
            first=False
        return mothers
    
    def getMothers(self):
        """ get all mothers of an object """
        mothers=[]
        m=self.getFirstMother()
        while m!=None:
            mothers+=[m]
            m=m.getFirstMother()
        return mothers

#    def getNext(self):
#        """ Get next sibling """
#        next=None
#        if self.mother!=None:
#            if self.mother.daughters.index(self)<len(self.mother.daughters)-1:
#                next=self.mother.daughters[self.mother.daughters.index(self)+1]
#        return next
#
#    def setNext(self,next):
#        raise NotImplementedError
#            
#    next=property(getNext,setNext)
#
#    def getPrevious(self):
#        """ Get previous sibling """
#        previous=None
#        if self.mother!=None:
#            if self.mother.daughters.index(self)>0:
#                previous=self.mother.daughters[self.mother.daughters.index(self)-1]
#        return previous
#
#    def setPrevious(self,previous):
#        raise NotImplementedError
#            
#    previous=property(getPrevious,setPrevious)

class UserObject(RelativeObject):
    """ Object that holds user information """
    def __init__(self,name="",mother=None,add_properties=[]):
        """ constructor """
        RelativeObject.__init__(self,mother)
        self.label=name
        self.properties=[]
        self.properties+=[("Label","Object info","")]
        if self.label!="":
            self.properties+=[("Text","label",self.label)]
        self.properties+=add_properties
