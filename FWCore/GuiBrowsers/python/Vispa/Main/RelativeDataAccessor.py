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

    def allDaughterRelations(self,object):
        daughterRelations=list(self.daughterRelations(object))
        for child in self.daughterRelations(object):
            daughterRelations+=list(self.allDaughterRelations(child)) 
        return tuple(daughterRelations)

    def allMotherRelations(self,object):
        motherRelations=list(self.motherRelations(object))
        for child in self.motherRelations(object):
            motherRelations+=list(self.allMotherRelations(child)) 
        return tuple(motherRelations)
