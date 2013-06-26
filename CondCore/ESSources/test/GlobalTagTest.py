from CondCore.ESSources.GlobalTag import *
import unittest

class TestGlobalTag(unittest.TestCase):
    def setUp(self):
        self.GT1 = GlobalTag("PRA_61_V1::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V1.db")
        self.GT2 = GlobalTag("PRE_61_V2::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V2.db")
        self.GT3 = GlobalTag("PRB_61_V3::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V3.db")
        self.GT4 = GlobalTag("PRE_61_V4::All", "sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/PRE_61_V4.db")
        self.Alias = GlobalTag("MAINGT")
        self.AliasWithConnectionString = GlobalTag("MAINGT", "frontier://FrontierProd/CMS_COND_31X_GLOBALTAG")

    def test_or(self):
        """ Test concatenation of different GT components """
        self.globalTag = self.GT1 | self.GT2 | self.GT3
        self.assertTrue( self.globalTag.gt() == self.GT1.gt()+"|"+self.GT2.gt()+"|"+self.GT3.gt() )
        self.assertTrue( self.globalTag.connect() == self.GT1.connect()+"|"+self.GT2.connect()+"|"+self.GT3.connect() )
        self.assertTrue( self.globalTag.pfnPrefix() == self.GT1.pfnPrefix()+"|"+self.GT2.pfnPrefix()+"|"+self.GT3.pfnPrefix() )
        self.assertTrue( self.globalTag.pfnPostfix() == self.GT1.pfnPostfix()+"|"+self.GT2.pfnPostfix()+"|"+self.GT3.pfnPostfix() )

    def test_orException(self):
        """ Test exception when trying to concatenate the same component type """
        try:
            self.GT1 | self.GT2 | self.GT4
        except GlobalTagBuilderException:
            self.assertTrue( True )
        else:
            self.assertTrue( False )

    def test_add(self):
        """ Test replacement of an existing component type """
        self.globalTag = (self.GT1 | self.GT2 | self.GT3) + self.GT4
        self.assertTrue( self.globalTag.gt() == self.GT1.gt()+"|"+self.GT4.gt()+"|"+self.GT3.gt() )
        self.assertTrue( self.globalTag.connect() == self.GT1.connect()+"|"+self.GT4.connect()+"|"+self.GT3.connect() )
        self.assertTrue( self.globalTag.pfnPrefix() == self.GT1.pfnPrefix()+"|"+self.GT4.pfnPrefix()+"|"+self.GT3.pfnPrefix() )
        self.assertTrue( self.globalTag.pfnPostfix() == self.GT1.pfnPostfix()+"|"+self.GT4.pfnPostfix()+"|"+self.GT3.pfnPostfix() )

    def test_addException(self):
        """ Test exception when trying to replace a non existent component type """
        try:
            (self.GT1 | self.GT3) + self.GT4
        except GlobalTagBuilderException:
            self.assertTrue( True )
        else:
            self.assertTrue( False )

    def test_alias(self):
        """ Test the aliases """
        self.globalTag = (self.Alias)
        self.globalTag = (self.AliasWithConnectionString)

if __name__ == '__main__':
    unittest.main()
