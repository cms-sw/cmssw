#!/bin/sh

py=`mktemp --tmpdir=. make_db_XXXXXXXXXX.py`

trap "rm -f \"$py\"" 1 2 3 15

o="./output.db"
v="00"

#eval `scramv1 runtime -sh`
rm -f "$o"
for f in ue_calibrations_pf_mc.txt ue_calibrations_calo_mc.txt ue_calibrations_pf_data.txt ue_calibrations_calo_data.txt; do
    e="unknown"
    d="unknown"
    case "$f" in
	*pf*) e="PF";;
	*calo*) e="Calo";;
    esac
    case "$f" in
	*data*) d="offline";;
	*mc*) d="mc";;
    esac
    cat <<EOF > "$py"
import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')

ivars.register ('outputTag',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")
ivars.outputTag="UETable_${e}_v${v}_${d}"

ivars.register ('inputFile',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")

ivars.register ('outputFile',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")

ivars.inputFile="$f"
ivars.outputFile="$o"

ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMMY')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string("runnumber"),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = "sqlite_file:" + ivars.outputFile

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          timetype = cms.untracked.string("runnumber"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('HeavyIonUERcd'),
                                                                     tag = cms.string(ivars.outputTag)
                                                                     )
                                                            )
                                          )

process.makeUETableDB = cms.EDAnalyzer('UETableProducer',
                                       txtFile = cms.string(ivars.inputFile),
				       jetCorrectorFormat = cms.untracked.bool(True)
                                       )

process.step  = cms.Path(process.makeUETableDB)
EOF
    cmsRun "$py"
done

#rm -f "$py"
