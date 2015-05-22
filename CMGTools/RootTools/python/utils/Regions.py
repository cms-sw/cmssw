import itertools


class RegionsBase( object ):
    '''Not necessary?'''
    def __init__(self):
        self.regions = {}
        
    def test(self, event):
        return None


class Regions1D( object ):
    '''Handles a set of regions along 1 dimension (region_name, min, max).
    ROOTOOLS
    The regions should be non-overlapping.'''

    def __init__(self, complete=False):
        self.regions = {}
        self.overlapChecked = False
        self.complete = complete

    def regionNames(self):
        names = self.regions.values()
        if not self.complete:
            names.append('None')
        return names

    def addRegion(self, name, min, max):
        self.regions[(min, max)] = name

    def _checkOverlap(self):
        self.overlapChecked = True
        overlap = False
        for region in self.regions.keys():
            min, max = region
            for other in self.regions.keys():
                omin, omax = other 
                if min < omin and omin < max:
                    return True
                if min < omax and omax < max:
                    return True
                
    def test(self, var):
        '''Returns the name of the region containing var, and None if such
        a region cannot be found.

        The algorithm might not be the fastest one around, but profiling
        said we do not care. 
        '''
        if not self.overlapChecked:
            overlap = self._checkOverlap()
            if overlap:
                raise ValueError( 'Please define non-overlapping regions' )
        last = None
        for region, name in sorted(self.regions.iteritems() ):
            min, max = region
            if min > var:
                break
            else:
                last = (min, max, name)
        # last is now the last range for which min < var
        if last is None:
            if self.complete:
                raise ValueError('you declared this region as complete, it is not.')
            return 'None'
        min, max, name = last
        if var < max:
            return name
        else:
            if self.complete:
                raise ValueError('you declared this region as complete, it is not.')
            return 'None'

    def __str__(self):
        tmp = '\n'.join( map(str, sorted(self.regions.iteritems() )) )
        return tmp



        
if __name__ == '__main__':

    reg = Regions1D()
    reg.addRegion( 'low_mT', 0, 50)
    reg.addRegion( 'high_mT', 80, float('+inf') )

    print reg.test(20)
    print reg.test(60)
    print reg.test(100124)
