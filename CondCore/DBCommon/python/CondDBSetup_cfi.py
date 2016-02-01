print " ################################################################### "
print " # WARNING: this module is deprecated.                             # "  
print " # Please use CondCore.CondDB.CondDB_cfi.py                        # "
print " ################################################################### "

from CondCore.CondDB.CondDB_cfi import *
CondDBSetup = CondDB.clone()
CondDBSetup.__delattr__('connect')

