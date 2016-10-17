import unittest
from genbrowser import GenBrowser

class Particle(object):

    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end
        # self.mothers = []
        # self.daughters = []

    def start_vertex(self):
        return self.start

    def end_vertex(self):
        return self.end

    def __str__(self):
        return 'particle {i}: \tstart {s}, \tend {e}'.format(
            i=self.id,
            s=self.start.id if self.start else None,
            e=self.end.id if self.end else None
            )
        
    
class Vertex(object):

    def __init__(self, id):
        self.id = id
        self.outgoing = []
        self.incoming = []

    def __str__(self):
        outg = map(str, self.outgoing)
        inc = map(str, self.incoming)
        result = ['vertex {i}'.format(i=self.id), 'incoming']
        result += inc
        result.append('outgoing')
        result += outg
        return '\n'.join(result)
        
        

class TestGenBrowser(unittest.TestCase):

    def test_1(self):
        vs = map(Vertex, range(2))
        ps = [
            Particle(0, None, vs[0]),
            Particle(1, vs[0], None),
            Particle(2, vs[0],vs[1]),
            Particle(3, vs[1],None),
            Particle(4, vs[1],None),
            Particle(5, None, vs[0])
        ]
        browser = GenBrowser(ps, vs)
        self.assertItemsEqual( vs[0].incoming, [ps[0], ps[5]] )
        self.assertItemsEqual( vs[0].outgoing, ps[1:3] )
        self.assertItemsEqual( vs[1].incoming, [ps[2]] )
        self.assertItemsEqual( vs[1].outgoing, ps[3:5] )
        self.assertItemsEqual( ps[0].daughters, ps[1:3] )
        self.assertItemsEqual( ps[0].mothers, [] )
        self.assertItemsEqual( ps[1].daughters, [] )
        self.assertItemsEqual( ps[1].mothers, [ps[0], ps[5]] )
        self.assertItemsEqual( ps[2].daughters, ps[3:5] )
        self.assertItemsEqual( ps[2].mothers, [ps[0], ps[5]] )
        self.assertItemsEqual( ps[3].daughters, [] )
        self.assertItemsEqual( ps[3].mothers, [ps[2]] )
        self.assertItemsEqual( ps[4].daughters, [] )
        self.assertItemsEqual( ps[4].mothers, [ps[2]] )
        self.assertItemsEqual( browser.ancestors(ps[4]), [ps[2], ps[0], ps[5]]) 
        self.assertItemsEqual( browser.descendants(ps[0]), ps[1:5]) 



if __name__ == '__main__':
    unittest.main()

