class RelativeDataAccessor(object):
    """ This class provides access to the underlying data model.
    """
    
    def daughterRelations(self, object):
        """ Return a list of the daughter relations of an object.
        """
        raise NotImplementedError
    
    def motherRelations(self, object):
        """ Return a list of the mother relations of an object.
        """
        raise NotImplementedError

    def allDaughterRelations(self,object, list=None):
        daughterRelations=[]
        for child in self.daughterRelations(object):
            if list==None or not child in list:
                daughterRelations+=[child]
            for grandchild in self.allDaughterRelations(child,daughterRelations):
                daughterRelations+=[grandchild]
        return daughterRelations

    def allMotherRelations(self,object, list=None):
        motherRelations=[]
        for mother in self.motherRelations(object):
            if list==None or not mother in list:
                motherRelations+=[mother]
            for grandmother in self.allMotherRelations(motherRelations):
                motherRelations+=[grandmother]
        return motherRelations

    def hasRelations(self,object):
        """ Return if object has relations.
        """
        return True
