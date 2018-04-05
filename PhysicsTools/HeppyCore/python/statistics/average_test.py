import unittest

from average import Average

class AverageTestCase(unittest.TestCase):

    def test_ave_unw(self):
        c = Average('TestAve')
        c.add( 1, 1 )
        c.add( 2, 1 )
        ave, unc = c.average()
        self.assertEqual(ave, 1.5)

    def test_ave_wei(self):
        c = Average('TestAve')
        c.add( 0, 1 )
        c.add( 1, 3 )
        ave, unc = c.average()
        self.assertEqual(ave, 0.75)

    def test_ave_add(self):
        c1 = Average('c1')
        c1.add(0,1)
        c2 = Average('c2')
        c2.add(1,3)
        c3 = c1 + c2
        ave, unc = c3.average()
        self.assertEqual(ave, 0.75)

if __name__ == '__main__':
    unittest.main()
