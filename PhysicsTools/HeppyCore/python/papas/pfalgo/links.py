import itertools
import pprint
from floodfill import FloodFill

class Element(object):
    '''Basic interface for a particle flow element.
    Your class should expose the same attributes
    '''
    def __init__(self):
        self.layer = None
        self.linked = []
        self.locked = False
        self.block_label = None

    def accept(self, visitor):
        '''Called by visitors, such as FloodFill.'''
        notseen = visitor.visit(self)
        if notseen:
            for elem in self.linked:
                elem.accept(visitor)

                
class Distance(object):
    '''Basic distance functor interface.
    You should provide such a functor (or a function), able to deal 
    with any pair of elements you have.
    ''' 
    def __call__(self, ele1, ele2):
        '''Should return True if the link is valid, 
        together with a link property object (maybe only the link distance).
        '''
        link_type = 'dummy'
        dist12 = 0.
        return link_type, True, dist12
    
    
    
class Links(dict):

    def __init__(self, elements, distance):
        self.elements = elements
        for ele in elements:
            ele.linked = []
        for ele1, ele2 in itertools.combinations(elements, 2):
            link_type, link_ok, dist = distance(ele1, ele2)
            if link_ok: 
                self.add(ele1, ele2, dist)
        floodfill = FloodFill(elements)
        #print floodfill
        self.groups = floodfill.groups
        self.group_label = floodfill.label
        for elem in elements:
            self.sort_links(elem)

    def subgroups(self, groupid):
        floodfill = FloodFill(self.groups[groupid], self.group_label)
        self.group_label = floodfill.label
        return floodfill.groups
        # if len(floodfill.groups)>1:
        #     del self.groups[groupid]
        #     self.groups.extend(floodfill.groups)
    
    def dist_linked(self, elem):
        '''returns [(dist, linked_elem1), ...]
        for all elements linked to elem.'''
        dist_linked = []
        for linked_elem in elem.linked:
            dist = self.info(elem, linked_elem)
            dist_linked.append( (dist, linked_elem) )
        return dist_linked
            
    def sort_links(self, elem):
        '''sort links in elem according to link distance.
        TODO unittest
        '''
        dist_linked = []
        for linked_elem in elem.linked:
            dist = self.info(elem, linked_elem)
            dist_linked.append( (dist, linked_elem) )
        sorted_links = [linked_elem for dist, linked_elem in sorted(dist_linked)]
        elem.linked = sorted_links
            
    def key(self, elem1, elem2):
        '''Build the dictionary key for the pair elem1 and elem2.'''
        return tuple(sorted([elem1, elem2]))
    
    def add(self, elem1, elem2, link_info):
        '''Link two elements.
        TODO: call that link.
        '''
        key = self.key(elem1, elem2)
        elem1.linked.append(elem2)
        elem2.linked.append(elem1)
        self[key] = link_info

    def unlink(self, elem1, elem2):
        '''Unlink two elements.'''
        key = self.key(elem1, elem2)
        elem1.linked.remove(elem2)
        elem2.linked.remove(elem1)
        del self[key]
        
    def info(self, elem1, elem2):
        '''Return link information between two elements. 
        None if the link does not exist.'''
        key = self.key(elem1, elem2)
        return self.get(key, None)
     
    def __str__(self):
        lines = []
        for key, val in self.iteritems():
            ele1, ele2 = key
            lines.append("{ele1:50} {ele2:50} dist = {val:5.4f}".format(ele1=ele1,
                                                                 ele2=ele2,
                                                                 val=val))
        '\n Groups:\n'.join(lines)        
        for gid, group in self.groups.iteritems():
                    groupinfo = ', '.join(map(str, group))
                    lines.append('group {gid:5} {ginfo}'.format(gid=gid, ginfo=groupinfo))
                   
        return '\n'.join(lines)




