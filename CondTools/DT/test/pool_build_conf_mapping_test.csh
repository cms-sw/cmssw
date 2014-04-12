#!/bin/csh
pool_build_object_relational_mapping -f ../xml/DTCCBConfig-mapping-custom.xml -d CondFormatsDTObjects -c sqlite_file:testconf.db -u "user" -p "pass"

