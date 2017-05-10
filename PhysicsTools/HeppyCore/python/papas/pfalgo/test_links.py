import unittest
from links import Links, Element

def distance(ele1, ele2):
    dist = abs(ele1-ele2)
    return 'link_type', dist<3., dist


class TestElement(Element):
    def __init__(self, val):
        self.val = val
        super(TestElement, self).__init__()

    def __repr__(self):
        return str(self.val)
        
    def __sub__(self, other):
        return self.val - other.val

class TestLinks(unittest.TestCase):

    def test_link_1(self):
        elements = map(TestElement, range(10))
        links = Links(elements, distance)
        distances = links.values()
        self.assertTrue( max(distances)==2 )
        self.assertEqual(elements[0].linked, [elements[1], elements[2]])
        self.assertEqual(links.info(elements[2], elements[4]), 2)
        self.assertIsNone(links.info(elements[2], elements[5]), None)
        
if __name__ == '__main__':
    unittest.main()



