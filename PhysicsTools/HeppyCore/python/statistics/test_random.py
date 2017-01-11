import unittest
import logging
logging.getLogger().setLevel(logging.ERROR)

import rrandom as random

class TestRandom(unittest.TestCase):

    def test_seed(self):

        #unseeded
        r0 = random.uniform(0, 1)
        r1 = random.expovariate(3000)
        r2 = random.gauss(1,3)

        #seed
        random.seed(0xdeadbeef)
        a0 = random.uniform(0, 1)
        a1 = random.expovariate(3)
        a2 = random.gauss(1,3)

        #reseed
        random.seed(0xdeadbeef)
        b0 = random.uniform(0, 1)
        b1 = random.expovariate(3)
        b2 = random.gauss(1,3)

        #unseeded should be different to seeded
        self.assertFalse(a0==r0)
        self.assertFalse(a1==r1)
        self.assertFalse(a2==r2)

        #reseeded should be same as seeded
        self.assertEqual(a0,b0)
        self.assertEqual(a1,b1)
        self.assertEqual(a2,b2)


if __name__ == '__main__':

    unittest.main()
