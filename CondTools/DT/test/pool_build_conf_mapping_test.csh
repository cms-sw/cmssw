#!/bin/csh
pool_build_object_relational_mapping -f ../xml/DTCCBConfig-mapping-custom.xml -d CondFormatsDTObjectsCapabilities -c sqlite_file:testconf.db -u "user" -p "pass"
pool_build_object_relational_mapping -f ../xml/DTConfigData-mapping-custom.xml -d CondFormatsDTObjectsCapabilities -c sqlite_file:testconf.db -u "user" -p "pass"
pool_build_object_relational_mapping -f ../xml/DTConfigList-mapping-custom.xml -d CondFormatsDTObjectsCapabilities -c sqlite_file:testconf.db -u "user" -p "pass"

