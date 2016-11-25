import copy

class GenBrowser(object):
    """Browser for gen particle history."""
    
    def __init__(self, particles, vertices):
        """
        parameters: 
        - particles: a list of gen particles 
        
        the particles must have a start_vertex and an end_vertex 
        attribute, set to None if the vertex doesn't exist. 
    
        After calling this constructor, two lists are added to each 
        particle: 
        - daughters: list of direct daugthers
        - mothers: list of direct mothers  
        """
        self.vertices = dict()
        for v in vertices:
            self.vertices[v] = v
        self.particles = particles
        for ptc in particles:
            ptc.daughters = []
            ptc.mothers = []
            start = ptc.start_vertex()
            if start:
                vertex = self.vertices.get(start, None)
                if vertex:
                    vertex.outgoing.append(ptc)
                else:
                    raise ValueError('vertex not found!')
            end = ptc.end_vertex()
            if end:
                vertex = self.vertices.get(end, None)
                if vertex:
                    vertex.incoming.append(ptc)
                else:
                    raise ValueError('vertex not found!')

        # now the lists of incoming and outgoing particles is
        # complete for each vertex
        # setting the list of daughters and mothers for each particle
        for vtx in self.vertices:
            # print vtx, id(vtx),'-'*50
            # print 'incoming'
            for ptc in vtx.incoming:
                # print ptc
                ptc.daughters = vtx.outgoing
            # print 'outgoing'
            for ptc in vtx.outgoing:
                # print ptc
                ptc.mothers = vtx.incoming
                
    def ancestors(self, particle):
        """Returns the list of ancestors for a given particle, 
        that is mothers, grandmothers, etc."""
        result = []
        for mother in particle.mothers:
            result.append(mother)
            result.extend(self.ancestors(mother))
        return result

    def descendants(self, particle):
        """Returns the list of descendants for a given particle, 
        that is daughters, granddaughters, etc."""
        result = []
        for daughter in particle.daughters:
            result.append(daughter)
            result.extend(self.descendants(daughter))
        return result




