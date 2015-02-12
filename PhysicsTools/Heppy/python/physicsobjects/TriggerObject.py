from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *
import fnmatch

class TriggerObject( PhysicsObject):
    '''With a nice printout, and functions to investigate the path and filters in the trigger object.'''

    def hasPath( self, path ):
        '''Returns true if this trigger object was used in path
        (path can contain a wildcard).
        '''
        selNames = fnmatch.filter( self.getSelectionNames(), path )
        if len(selNames)>0:
            return True
        else:
            return False

    def __str__(self):
        base = super(TriggerObject, self).__str__()
        specific = []
        theStrs = [base]
        for name in self.getSelectionNames():
            hasSel = self.getSelection( name )
            if hasSel:
                specific.append( ''.join(['\t', name]) )
        if len(specific)>0:
            specific.insert(0,'Paths:')
            theStrs.extend( specific )
        return '\n'.join(theStrs)

