#!/bin/sh
pool_build_object_relational_mapping \
	-f MVAComputer-mapping-custom_1.0.xml \
	-d CondFormatsBTauObjects \
	-c sqlite_file:FoobarDiscriminator.db \
	-u me -p mypass
