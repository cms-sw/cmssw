import unittest
import os 
import shutil

from counter import Counter

class CounterTestCase(unittest.TestCase):

    def test_simple(self):
        c = Counter('Test')
        c.register('a')
        c.register('b')
        c.inc('a')
        self.assertEqual(c['a'], ['a', 1])
        self.assertEqual(c['b'], ['b', 0])
        c.inc('a')
        self.assertEqual(c['a'], ['a', 2])

    def test_add(self):
        c = Counter('Test')
        c.register('a')
        c.register('b')
        c.inc('a')
        d = Counter('Test')
        d.register('a')
        d.register('b')
        d.inc('a')
        d.inc('b')
        d += c 
        self.assertEqual(d['a'], ['a', 2])
        self.assertEqual(d['b'], ['b', 1])
     
    def test_bad_add(self):
        c = Counter('Test')
        c.register('a')
        c.register('b')
        c.inc('a')
        d = Counter('Test')
        d.register('b')
        self.assertRaises(ValueError, d.__iadd__, c)
  
    def test_write(self):
        c = Counter('Test')
        c.register('a')
        c.register('b')
        c.inc('a')
        dirname = 'test_dir'
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)
        c.write(dirname)
        shutil.rmtree(dirname)
        
        




if __name__ == '__main__':
    unittest.main()
