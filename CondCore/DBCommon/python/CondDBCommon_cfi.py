print " ##################################################################### "  
print " # WARNING: the module CondCore.DBCommon.CondDBCommon is deprecated. # "                                                                              
print " # Please import CondCore.CondDB.CondDB_cfi                          # "                                                                             
print " ##################################################################### "

from CondCore.CondDB.CondDB_cfi import *
CondDBCommon = CondDB.clone()

