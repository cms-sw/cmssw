# decorator for adding iterators to container like objects
# NOTE: EventBranch._readData has to be taken care of at another place!

#import cmserror

def addIterator(obj):
    """function for adding iterators to objects""" 
    if not hasattr(obj, "__iter__"):
        if hasattr(obj, "size"):
            obj.__iter__ = iteratorForSizedObjects
        else:
            try:
              begin, end = _findIterators(obj)
            except:
              return obj  
            if not hasattr(obj, "_begin") and hasattr(obj, "_end"):
                obj._begin = begin
                obj._end = end
                obj.__iter__ = iteratorForBeginEnd
        #else:
        #    obj.__iter__ = EmptyIterator
    return obj


def iteratorForSizedObjects(self):
    """dynamically added iterator"""
    entries = container.size()
    for entry in xrange(entries):
        yield obj[entry]
  
        
def iteratorForBeginEnd(self):
    """dynamically added iterator"""
    it = self._begin
    while (it != self.end):
        yield begin.__deref__()  #*b
        begin.__preinc__()       #++b


def emptyIterator(self):
    """empty iterator"""
    raise cmserror("Automatic iterator search failed for %s. Either it is no iterable or it has multiple iterator possibilites. Please use loop(begin, end) instead." %obj )


# automatic detection of iterators.      
def _findIterators(obj):
    objDict = obj.__dict__
    _beginNames = [name for name in objDict.keys() if "begin" in 
name.lower()]
    _endNames = [name for name in objDict.keys() if "end" in name.lower()]
    if len(_beginNames)==1 and len(_endNames)== 1 and _beginNames[0].lower().replace("begin","") == _endNames[0].lower().replace("end",""):  
        return objDict[_beginNames[0]], objDict[_endNames[0]]
    else:
        return False
        
        
        
##########################
if __name__ == "__main__":

    import unittest
    class TestIterators(unittest.TestCase):
    
        def testFindIterators(self):
            class A(object):
                pass
            a = A()
            a.BeGin_foo = 1
            a.EnD_foo = 100
            self.assertEqual(_findIterators(a),(1,100))
            a.begin_bar = 1
            a.end_bar = 100
            self.failIf(_findIterators(a))
                  
        def testAddIterator(self):
            # test for size types
            class A(object):
                size = 3
            a = A()
            a = addIterator(a)
            self.assert_(hasattr(a, "__iter__"))
            # test if __iter__ already there
            class B(object):
                def __iter__(self):
                    return True
            b = B()
            b = addIterator(b)
            self.assert_(b.__iter__())
            
        
    unittest.main()


