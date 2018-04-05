print " ##################################################################### "
print " # WARNING: the module CondCore.DBCommon.CondDBSetup is deprecated.  # "
print " # Please import CondCore.CondDB.CondDB_cfi                          # "
print " ##################################################################### "

from CondCore.CondDB.CondDB_cfi import *
CondDBSetup = CondDB.clone()
CondDBSetup.__delattr__('connect')

