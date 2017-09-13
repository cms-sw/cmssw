

class Node(object):
    '''Basic interface for nodes traversed by the floodfill algo.
    Floodill will give a label to each node corresponding to the 
    disconnected subgraph it corresponds to. 

    The linked attribute is a list that should contain the elements y-linkes.
    '''
    def __init__(self):
        self.linked = []
        self.block_label = None

    def accept(self, visitor):
        '''Called by visitors, such as FloodFill.'''
        notseen = visitor.visit(self)
        if notseen:
            for elem in self.linked:
                elem.accept(visitor)


class FloodFill(object):
    '''The flood fill algorithm finds all disconnected subgraphs in 
    a list of nodes. 
    
    The block_label of each node is set to an integer corresponding to the 
    disconnected subgraph it corresponds to. 
    
    The results can be accessed through the nodes themselves, 
    or through the groups attribute, which has the following form: 
      {0: [list of elements in subgraph0], 1: [list of elements in subgraph 1], ...}
    '''
    
    def __init__(self, elements, first_label=0):
        '''Perform the search for disconnected subgraphs on a list of elements 
        matching the interface given in this module.'''
        self.label = first_label
        self.visited = dict()
        self.groups = dict()
        for elem in elements:
            if self.visited.get(elem, False):
                continue
            elem.accept(self)
            # print 'incrementing', elem, self.label
            self.label += 1

    def visit(self, element):
        '''visit one element.'''
        if self.visited.get(element, False):
            return False
        else:
            # print 'visiting', element, self.label
            element.block_label = self.label
            self.groups.setdefault(element.block_label, []).append(element)
            self.visited[element] = True
            return True

    def __str__(self):
        lines = []
        for gid, group in self.groups.iteritems():
            groupinfo = ', '.join(map(str, group))
            lines.append('{gid:5} {ginfo}'.format(gid=gid, ginfo=groupinfo))
        return '\n'.join(lines)
            
