#!/bin/sh

# Build the appropriate schema in ORCOFF.  Takes as argument the schema to be used. (ie CMS_COND_BTAU)
# find passwords in /afs/cern.ch/cms/DB/conddb

eval `scramv1 runtime -sh`

pool_build_object_relational_mapping \
        -f $CMSSW_BASE/src/RecoTauTag/TauTagTools/xml/TauTagMVAComputer-mapping-1.0.xml \
	-d CondFormatsPhysicsToolsObjects \
	-c oracle://cms_orcoff_prep/CMS_COND_BTAU \
        -u cms_cond_btau \
        -p WCYE6II08K530GPK \
        -dry 

