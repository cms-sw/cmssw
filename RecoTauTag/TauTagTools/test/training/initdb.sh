#!/bin/sh

# Generate a local mysql database with the correct schema for
# holding TauMVA objects

eval `scramv1 runtime -sh`

pool_build_object_relational_mapping \
        -f $CMSSW_BASE/src/RecoTauTag/TauTagTools/xml/TauTagMVAComputer-mapping-1.0.xml \
	-d CondFormatsPhysicsToolsObjects \
	-c sqlite_file:$1 \
	-u me -p mypass -info
