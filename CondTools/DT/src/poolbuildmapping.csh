#!/bin/csh
pool_build_object_relational_mapping -f mapping-template-DTReadOutMapping-default.xml -d CondFormatsDTObjectsCapabilities -c sqlite_file:testfile.db -p pass -u user
pool_build_object_relational_mapping -f mapping-template-DTT0-default.xml -d CondFormatsDTObjectsCapabilities -c sqlite_file:testfile.db -p pass -u user
pool_build_object_relational_mapping -f mapping-template-DTTtrig-default.xml -d CondFormatsDTObjectsCapabilities -c sqlite_file:testfile.db -p pass -u user
pool_build_object_relational_mapping -f mapping-template-DTMtime-default.xml -d CondFormatsDTObjectsCapabilities -c sqlite_file:testfile.db -p pass -u user
