class Options(dict):

    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)
        self.readKeys = set()

    def __getitem__(self, key):
        self.readKeys.add(key)
        return self.get(key,None)

    def _unreadKeys(self):
        """return unused keys"""
        return set([a for a in self if a not in self.readKeys])
    unreadKeys = property(_unreadKeys)


##########################
if __name__ == "__main__":
    import unittest
    class TestOptions(unittest.TestCase):
        def testOptions(self):
            a = Options()
            a["A"] = 3
            a["B"] = 3
            a["A"]
            self.assertEqual(a.readKeys, set(["A"]))
            self.assertEqual(a.unreadKeys, set(["B"]))                        

    unittest.main()
