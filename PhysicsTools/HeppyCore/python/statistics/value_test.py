import unittest
import math
from value import Value 

class ValueTestCase(unittest.TestCase):

    def test(self):
        val1 = Value(1.,0.02)
        val2 = Value(2.,0.02)
        val3 = val1 / val2
        # should test the value and the error after each operation.
        # I'll write the tests when I have some time 

    def test_equal(self):
        val1 = Value(1.,0.02)
        val2 = Value(1.,0.02)
        self.assertEqual(val1, val2)

if __name__ == '__main__':
    unittest.main()
