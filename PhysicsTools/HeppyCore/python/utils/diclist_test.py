import unittest

from diclist import diclist

class DiclistTestCase(unittest.TestCase):

    def test_string_key(self):
        dl = diclist()
        dl.add('a', 1)
        dl.add('b', 2)
        dl.add('c', 3)
        self.assertEqual([1,2,3], [value for value in dl] ) 
        self.assertEqual(dl['c'], 3)

    def test_bad_int_key(self):
        dl = diclist()
        self.assertRaises(ValueError, dl.add, 1, 'a')
        self.assertRaises(ValueError, dl.add, 1L, 'a')
   
    def test_float_key(self):
        dl = diclist()
        dl.add(1., 'a')
        dl.add(2., 'b')
        self.assertRaises(IndexError, dl.__getitem__, 2)
        self.assertEqual(dl[2.], 'b')


if __name__ == '__main__':
    unittest.main()
