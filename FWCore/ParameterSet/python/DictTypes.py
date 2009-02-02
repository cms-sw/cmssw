# helper classes for sorted and fixed dicts
class SortedKeysDict(dict):
    """a dict preserving order of keys"""
    # specialised __repr__ missing.
    def __init__(self,*args,**kw):
        dict.__init__(self,*args,**kw)
        self.list = list()
        if len(args) == 1:
            if not hasattr(args[0],'iterkeys'):
                s = set()
                #must protect against adding the same key multiple times
                for x,y in iter(args[0]):
                    if x not in s:
                        self.list.append(x)
                        s.add(x)
            else:
                self.list = list(args[0].iterkeys())
            return
        self.list = list(super(SortedKeysDict,self).iterkeys())

    def __repr__(self):
        meat = ', '.join([ '%s: %s' % (repr(key), repr(val)) for key,val in self.iteritems() ])
        return '{' + meat + '}'

    def __iter__(self):
        for key in self.list:
            yield key
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if not key in self.list:
            self.list.append(key)
    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.list.remove(key)
    def items(self):
        return [ dict.__getitem__(self, key) for key in self.list]
    def iteritems(self):
        for key in self.list:
            yield key, dict.__getitem__(self, key)
    def iterkeys(self):
        for key in self.list:
            yield key
    def itervalues(self):
        for key in self.list:
            yield dict.__getitem__(self,key)
    def keys(self):
        return self.list
    def values(self):
        return [ dict.__getitems__(self, key) for key in self.list]


class SortedAndFixedKeysDict(SortedKeysDict):
    """a sorted dictionary with fixed/frozen keys"""
    def _blocked_attribute(obj):
        raise AttributeError, "A SortedAndFixedKeysDict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)
    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute
    def __new__(cls, *args, **kw):
        new = SortedKeysDict.__new__(cls)
        SortedKeysDict.__init__(new, *args, **kw)
        return new
    def __init__(self, *args, **kw):
        pass
    def __repr__(self):
        return "SortedAndFixedKeysDict(%s)" % SortedKeysDict.__repr__(self)


#helper based on code from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/414283
class FixedKeysDict(dict):
    def _blocked_attribute(obj):
        raise AttributeError, "A FixedKeysDict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute
    def __new__(cls, *args, **kw):
        new = dict.__new__(cls)
        dict.__init__(new, *args, **kw)
        return new
    def __init__(self, *args, **kw):
        pass
    def __repr__(self):
        return "FixedKeysDict(%s)" % dict.__repr__(self)


if __name__=="__main__":
    import unittest
    class TestDictTypes(unittest.TestCase):
        def testFixedKeysDict(self):
            import operator
            d = FixedKeysDict({'a':1, 'b':[3]})
            self.assertEqual(d['a'],1)
            self.assertEqual(d['b'],[3])
            self.assertRaises(AttributeError,operator.setitem,*(d,'a',2))
            d['b'].append(2)
            self.assertEqual(d['b'],[3,2])
        
        def testSortedKeysDict(self):
            sd = SortedKeysDict()
            sd['a']=1
            sd['b']=2
            sd['c']=3
            sd['d']=4
            count =1
            for key in sd.iterkeys():
                self.assertEqual(count,sd[key])
                count +=1
            sd2 = SortedKeysDict(sd)
            count =1
            for key in sd2.iterkeys():
                self.assertEqual(count,sd2[key])
                count +=1
            sd3 = SortedKeysDict([('a',1),('b',2),('c',3),('d',4)])
            count =1
            for key in sd3.iterkeys():
                self.assertEqual(count,sd3[key])
                count +=1
            self.assertEqual(count-1,len(sd3))
            sd3 = SortedKeysDict(a=1,b=2,c=3,d=4)
            count =1
            for key in sd3.iterkeys():
                count +=1
            self.assertEqual(count-1,len(sd3))
            sd['d']=5
            self.assertEqual(5,sd['d'])
            
        def testSortedAndFixedKeysDict(self):
            import operator
            sd = SortedAndFixedKeysDict({'a':1, 'b':[3]})
            self.assertEqual(sd['a'],1)
            self.assertEqual(sd['b'],[3])
            self.assertRaises(AttributeError,operator.setitem,*(sd,'a',2))
            sd = SortedAndFixedKeysDict([('a',1), ('b',2),('a',3)])
            self.assertEqual(['a','b'], [x for x in iter(sd)])
    unittest.main()
