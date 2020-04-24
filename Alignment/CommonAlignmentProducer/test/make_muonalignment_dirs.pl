#!/usr/bin/perl

$baseofall = `pwd`;
chop($baseofall);

$dirname = $ARGV[0];
$initial = "wheeldisk_nocut";
# $initial = "wheeldisk_cut3sig";
# $initial = "wheeldisk_cut5sig";
$filetemplate = "rfio:///?svcclass=cmscaf&path=//castor/cern.ch/cms/store/cmscaf/alca/alignment/CSA07/MuonHIP/AlCaRecoMu/wmunu_100pb-1/%02d.root";
# $filetemplate = "rfio:///?svcclass=cmscaf&path=//castor/cern.ch/cms/store/cmscaf/alca/alignment/CSA07/MuonHIP/AlCaRecoMu/wmunu_100pb-1_cut3sig/%02d.root";
# $filetemplate = "rfio:///?svcclass=cmscaf&path=//castor/cern.ch/cms/store/cmscaf/alca/alignment/CSA07/MuonHIP/AlCaRecoMu/wmunu_100pb-1_cut5sig/%02d.root";
# $filetemplate = "rfio:///castor/cern.ch/user/p/pivarski/tmp2/%02d.root";
# $filetemplate = "rfio:///castor/cern.ch/user/p/pivarski/tmp2_cut3sig/%02d.root";
# $filetemplate = "rfio:///castor/cern.ch/user/p/pivarski/tmp2_cut5sig/%02d.root";

$muonSourceLabel = "ALCARECOMuAlZMuMu";
# $muonSourceLabel = "StraightMuonCutProducer";

$theConstraint = "NONE";
# $theConstraint = "/afs/cern.ch/user/p/pivarski/scratch0/cmssw_dest/constraints/constraint_10";

$last = $initial;
$lastwhere = "$baseofall/$initial";

if (-e $dirname) {
    die "Delete the old one first: rm -rf $dirname";
}

if ($dirname eq "") {
    die "Give this trial a name as an argument: ./makestandard.pl SOMETHING";
}

system("mkdir $dirname");
&common_cff($dirname);
&source_cff($dirname);

open(RUNALL, "> $dirname/runall.sh");
print RUNALL "#!/bin/sh
cd $baseofall
eval `scramv1 run -sh`
";

foreach $pass ("pass1", "pass2_mb2_me12", "pass3_mb3_me21", "pass4_mb4_me13", "pass5_me22_me31", "pass6_me32_me41", "stage3", "stage4") {
    system("mkdir $dirname/$pass");

    if ($pass eq "pass1") {
	$alignParams = "{\"MuonDTChambers,111111,mb1\", \"MuonDTChambers,111111,mb2\", \"MuonDTChambers,111111,mb3\", \"MuonDTChambers,101011,mb4\", \"MuonCSCChambers,110011,me11\", \"MuonCSCChambers,110011,me12\", \"MuonCSCChambers,110011,me13\", \"MuonCSCChambers,110011,me21\", \"MuonCSCChambers,110011,me22\", \"MuonCSCChambers,110011,me31\", \"MuonCSCChambers,110011,me32\", \"MuonCSCChambers,110011,me41\"}";
	$APEs = "";
    }
    elsif ($pass eq "pass2_mb2_me12") {
	$alignParams = "{\"MuonDTChambers,111111,mb2\", \"MuonCSCChambers,110011,me12\"}";
	$APEs = ", {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb1\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me11\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}
";
    }
    elsif ($pass eq "pass3_mb3_me21") {
	$alignParams = "{\"MuonDTChambers,111111,mb3\", \"MuonCSCChambers,110011,me21\"}";
	$APEs = ", {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb1\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb2\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me11\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me12\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}
";
    }
    elsif ($pass eq "pass4_mb4_me13") {
	$alignParams = "{\"MuonDTChambers,101011,mb4\", \"MuonCSCChambers,110011,me13\"}";
	$APEs = ", {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb1\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb2\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb3\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me11\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me12\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me21\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}
";
    }
    elsif ($pass eq "pass5_me22_me31") {
	$alignParams = "{\"MuonCSCChambers,110011,me22\", \"MuonCSCChambers,110011,me31\"}";
	$APEs = ", {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb1\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb2\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb3\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb4\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me11\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me12\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me13\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me21\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}
";
    }
    elsif ($pass eq "pass6_me32_me41") {
	$alignParams = "{\"MuonCSCChambers,110011,me32\", \"MuonCSCChambers,110011,me41\"}";
	$APEs = ", {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb1\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb2\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb3\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb4\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me11\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me12\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me13\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me21\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me22\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me31\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}
";
    }
    elsif ($pass eq "stage3") {
 	$alignParams = "{\"MuonDTChambers,111111,mb1\", \"MuonDTChambers,111111,mb2\", \"MuonDTChambers,111111,mb3\", \"MuonDTChambers,101011,mb4\", \"MuonCSCChambers,110011,me11\", \"MuonCSCChambers,110011,me12\", \"MuonCSCChambers,110011,me13\", \"MuonCSCChambers,110011,me21\", \"MuonCSCChambers,110011,me22\", \"MuonCSCChambers,110011,me31\", \"MuonCSCChambers,110011,me32\", \"MuonCSCChambers,110011,me41\"}";
	$APEs = ", {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111\", \"MuonCSCChambers,111111\"} }
    string function = \"linear\"
    vdouble apeSPar = {0.1, 0.1, 10.}
    vdouble apeRPar = {0.1, 0.1, 10.}
}
";
    }
    elsif ($pass eq "stage4") {
	$alignParams = "{\"MuonDTChambers,111111,mb2\", \"MuonDTChambers,111111,mb3\", \"MuonDTChambers,101011,mb4\", \"MuonCSCChambers,110011,me13\", \"MuonCSCChambers,110011,me21\", \"MuonCSCChambers,110011,me22\", \"MuonCSCChambers,110011,me31\", \"MuonCSCChambers,110011,me32\", \"MuonCSCChambers,110011,me41\"}";
	$APEs = ", {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111\", \"MuonCSCChambers,111111\"} }
    string function = \"linear\"
    vdouble apeSPar = {0.1, 0.1, 10.}
    vdouble apeRPar = {0.1, 0.1, 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonDTChambers,111111,mb1\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me11\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}, {
    PSet Selector = { vstring alignParams = {\"MuonCSCChambers,111111,me12\"} using MuonAlignmentStationSelector }
    string function = \"linear\"
    vdouble apeSPar = {0., 0., 10.}
    vdouble apeRPar = {0., 0., 10.}
}
";
    }

    @iters = ("iter1", "iter2", "iter3", "iter4", "iter5");

    foreach $iter (@iters) {
	system("mkdir $dirname/$pass/$iter");

	print RUNALL "cd $baseofall/$dirname/$pass/$iter\n";

	foreach $N (1..50) {
	    printf(RUNALL "cd job%d
bsub -q cmscaf ./$dirname\_$pass\_$iter\_job%d.sh
cd ..\n", $N, $N);
	}
	foreach $N (1..50) {
	    printf(RUNALL "while (test ! -e job%d/DONE.txt); do sleep 1; done\n", $N);
	}

	system("ln -s $lastwhere.db $dirname/$pass/$iter/$last.db");
	system("ln -s $lastwhere.xml $dirname/$pass/$iter/$last.xml");

	foreach $N (1..50) {
	    $job = sprintf("job%d", $N);
	    system("mkdir $dirname/$pass/$iter/$job");
	    open(FILE, "> $dirname/$pass/$iter/$job/$dirname\_$pass\_$iter\_$job.cfg");

	    print FILE "process align = {
    include \"$dirname/sources/$job.cff\"
    include \"$dirname/common.cff\"

    replace AlignmentProducer.ParameterBuilder.Selector = {
        vstring alignParams = $alignParams
        using MuonAlignmentStationSelector
    }

    replace AlignmentProducer.algoConfig.applyAPE = true
    replace AlignmentProducer.algoConfig.apeParam = {
	{
	    PSet Selector = {vstring alignParams = {\"MuonDTChambers,111111\", \"MuonCSCChambers,111111\"}}
	    string function = \"linear\"
	    vdouble apeSPar = {1000., 1000., 10.}
	    vdouble apeRPar = {1000., 1000., 10.}
	}$APEs
    }

    include \"Alignment/CommonAlignmentMonitor/data/AlignmentMonitorMuonResidualsDefaults.cff\"
    replace AlignmentProducer.monitorConfig = {
	untracked vstring monitors = {\"AlignmentMonitorMuonResiduals\"}
	
	untracked PSet AlignmentMonitorMuonResiduals = {
	    string outfile = \"$dirname\_$pass\_$iter.root\"
	    using AlignmentMonitorMuonResidualsDefaults
	}
    }

    replace PoolDBESSource.connect = \"sqlite_file:$last.db\"
    replace PoolDBESSource.catalog = \"xmlcatalog_file:$last.xml\"

    path p = {recoMuon2recoTrack, TrackRefitter}

";
	    if ($theConstraint ne "NONE") {
		print FILE "
    es_source constraints = PoolDBESSource {
        using CondDBCommon
        untracked uint32 authenticationMethod = 1
        VPSet toGet = {
            {
                string record = \"DTSurveyRcd\"
                string tag = \"DTSurveyRcd\"
            },
            {
                string record = \"DTSurveyErrorExtendedRcd\"
                string tag = \"DTSurveyErrorExtendedRcd\"
            },
            {
                string record = \"CSCSurveyRcd\"
                string tag = \"CSCSurveyRcd\"
            },
            {
                string record = \"CSCSurveyErrorExtendedRcd\"
                string tag = \"CSCSurveyErrorExtendedRcd\"
            }
        }
    }
    replace constraints.connect = \"sqlite_file:$theConstraint.db\"
    replace constraints.catalog = \"xmlcatalog_file:$theConstraint\_catalog.xml\"
    replace AlignmentProducer.useSurvey = true
";
	    }
	    print FILE "
}
";
	    close(FILE);

	    open(FILE, "> $dirname/$pass/$iter/$job/$dirname\_$pass\_$iter\_$job.sh");
            print FILE "#!/bin/sh
cd $baseofall
eval `scramv1 run -sh`
# export STAGE_SVCCLASS=cmscaf
cd $dirname/$pass/$iter/$job

echo \"Start\"
date

echo \"About to rm -rf /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job\"
rm -rf /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job
date

echo \"About to mkdir -p /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job\"
mkdir -p /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job
date

echo \"About to cp $dirname\_$pass\_$iter\_$job.cfg $lastwhere.db $lastwhere.xml /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job/\"
cp $dirname\_$pass\_$iter\_$job.cfg $lastwhere.db $lastwhere.xml /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job/
date

";
	    if ($theConstraint ne "NONE") {
		print FILE "
echo \"About to copy $theConstraint to /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job\"
cp $theConstraint.db /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job/
cp $theConstraint\_catalog.xml /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job/
date
";
	    }

	    print FILE "
echo \"About to cd /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job\"
cd /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job
date

echo \"About to pwd and ls\"
pwd
ls
date

echo \"About to cmsRun\"
cmsRun $dirname\_$pass\_$iter\_$job.cfg  &&  echo DONE > DONE.txt
date

echo \"About to cp *.root *.txt $baseofall/$dirname/$pass/$iter/$job/\"
cp *.root *.txt $baseofall/$dirname/$pass/$iter/$job/
date

echo \"About to cd $baseofall/$dirname/$pass/$iter/$job\"
cd $baseofall/$dirname/$pass/$iter/$job
date

echo \"About to rm -rf /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job\"
rm -rf /pool/lsf/pivarski/$dirname\_$pass\_$iter\_$job
date
";
	    close(FILE);
	    system("chmod +x $dirname/$pass/$iter/$job/$dirname\_$pass\_$iter\_$job.sh");
	} # end loop over jobs

	open(FILE, "> $dirname/$pass/$iter/collect.cfg");
	print FILE "process collect = {
    source = EmptySource {}
    untracked PSet maxEvents = {untracked int32 input = 0}
    include \"$dirname/common.cff\"

    replace AlignmentProducer.algoConfig.collectorActive = true
    replace AlignmentProducer.algoConfig.collectorNJobs = 50
    replace AlignmentProducer.algoConfig.collectorPath = \"./\"

    replace AlignmentProducer.algoConfig.applyAPE = false
    replace AlignmentProducer.algoConfig.apeParam = {
	{
	    PSet Selector = {vstring alignParams = {\"MuonDTChambers,111111\", \"MuonCSCChambers,111111\"}}
	    string function = \"linear\"
	    vdouble apeSPar = {0., 0., 10.}
	    vdouble apeRPar = {0., 0., 10.}
	}
    }

    replace AlignmentProducer.ParameterBuilder.Selector = {
        vstring alignParams = $alignParams
        using MuonAlignmentStationSelector
    }

    replace AlignmentProducer.monitorConfig = {
        untracked vstring monitors = {}
    }

    replace PoolDBESSource.connect = \"sqlite_file:$last.db\"
    replace PoolDBESSource.catalog = \"xmlcatalog_file:$last.xml\"

    service = PoolDBOutputService {
        using CondDBCommon
        untracked uint32 authenticationMethod = 1
        VPSet toPut = {
            {
                string record = \"DTAlignmentRcd\"
                string tag = \"DTAlignmentRcd\"
            },
            {
                string record = \"DTAlignmentErrorExtendedRcd\"
                string tag = \"DTAlignmentErrorExtendedRcd\"
            },
            {
                string record = \"CSCAlignmentRcd\"
                string tag = \"CSCAlignmentRcd\"
            },
            {
                string record = \"CSCAlignmentErrorExtendedRcd\"
                string tag = \"CSCAlignmentErrorExtendedRcd\"
            }
        }
    }
    replace AlignmentProducer.saveToDB = true
    replace PoolDBOutputService.connect = \"sqlite_file:$dirname\_$pass\_$iter.db\"
    replace PoolDBOutputService.catalog = \"xmlcatalog_file:$dirname\_$pass\_$iter.xml\"

    path p = {recoMuon2recoTrack, TrackRefitter}
}
";
	close(FILE);
	print RUNALL "cmsRun collect.cfg\n";

	open(FILE, "> $dirname/$pass/$iter/dbtoxml.cfg");
	print FILE "process dbtoxml = {
    source = EmptySource {}
    untracked PSet maxEvents = {untracked int32 input = 0}
    
    include \"Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi\"
    include \"Geometry/MuonNumbering/data/muonNumberingInitialization.cfi\"
    
    include \"CondCore/DBCommon/data/CondDBCommon.cfi\"
    es_source = PoolDBESSource {
	using CondDBCommon
	untracked uint32 authenticationMethod = 1
	VPSet toGet = {
	    {
		string record = \"DTAlignmentRcd\"
		string tag = \"DTAlignmentRcd\"
	    },
	    {
		string record = \"DTAlignmentErrorExtendedRcd\"
		string tag = \"DTAlignmentErrorExtendedRcd\"
	    },
	    {
		string record = \"CSCAlignmentRcd\"
		string tag = \"CSCAlignmentRcd\"
	    },
	    {
		string record = \"CSCAlignmentErrorExtendedRcd\"
		string tag = \"CSCAlignmentErrorExtendedRcd\"
	    }
	}
    }
    replace PoolDBESSource.connect = \"sqlite_file:$dirname\_$pass\_$iter.db\"
    replace PoolDBESSource.catalog = \"xmlcatalog_file:$dirname\_$pass\_$iter.xml\"

    module MuonGeometryDBConverter = MuonGeometryDBConverter {
	string input = \"db\"
	string dtLabel = \"\"
        string cscLabel = \"\"
        double shiftErr = 1000.
        double angleErr = 6.28

	string output = \"xml\"
        PSet outputXML = {
            string fileName = \"geom_$dirname\_$pass\_$iter.xml\"
	    string relativeto = \"ideal\"
	    bool survey = false
	    bool rawIds = false
	    bool eulerAngles = false

	    untracked bool suppressDTBarrel = true
	    untracked bool suppressDTWheels = true
	    untracked bool suppressDTStations = true
	    untracked bool suppressDTChambers = false
	    untracked bool suppressDTSuperLayers = true
	    untracked bool suppressDTLayers = true
	    untracked bool suppressCSCEndcaps = true
	    untracked bool suppressCSCStations = true
	    untracked bool suppressCSCRings = true
	    untracked bool suppressCSCChambers = false
	    untracked bool suppressCSCLayers = true
        }
    }

    path p = {MuonGeometryDBConverter}
}
";
	close(FILE);
	print RUNALL "cmsRun dbtoxml.cfg\n";
	print RUNALL "../../../Alignment/MuonAlignment/python/geometryXMLtoCSV.py < geom_$dirname\_$pass\_$iter.xml > geom_$dirname\_$pass\_$iter.csv\n";

	&collect_py($dirname, $pass, $iter);
	print RUNALL "python collect.py\n";
	print RUNALL "rm -f job*/$dirname\_$pass\_$iter.root\n\n";

	$last = "$dirname\_$pass\_$iter";
	$lastwhere = "$baseofall/$dirname/$pass/$iter/$last";

    } # end loop over iters
} # end loop over passes

close(RUNALL);
system("chmod +x $dirname/runall.sh");

###############################################################################################

sub common_cff($dirname) {
    open(FILE, "> $dirname/common.cff");

    print FILE "service = MessageLogger {
    untracked vstring destinations = {\"cout\"}
    untracked PSet cout = {
        untracked string threshold = \"ERROR\"
    }
}

include \"MagneticField/Engine/data/volumeBasedMagneticField.cfi\"
include \"Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi\"
include \"Geometry/CommonDetUnit/data/bareGlobalTrackingGeometry.cfi\"
include \"Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi\"
include \"Geometry/MuonNumbering/data/muonNumberingInitialization.cfi\"
include \"Geometry/RPCGeometry/data/rpcGeometry.cfi\"
include \"TrackingTools/TrackRefitter/data/TracksToTrajectories.cff\"
include \"RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilderWithoutRefit.cfi\"
include \"Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi\"
include \"CondCore/DBCommon/data/CondDBCommon.cfi\"

module recoMuon2recoTrack = AlignmentMuonSelector from \"Alignment/CommonAlignmentProducer/data/AlignmentMuonSelector.cfi\"
replace recoMuon2recoTrack.src = $muonSourceLabel
replace recoMuon2recoTrack.filter = false
replace recoMuon2recoTrack.applyBasicCuts = false
replace recoMuon2recoTrack.applyNHighestPt = false
replace recoMuon2recoTrack.applyMultiplicityFilter = false
replace recoMuon2recoTrack.applyMassPairFilter = false

module TrackRefitter = TracksToTrajectories {
    InputTag Tracks = recoMuon2recoTrack:GlobalMuon
    PSet TrackTransformer = {
        string Fitter = \"KFFitterForRefitInsideOut\"
        string Smoother = \"KFSmootherForRefitInsideOut\"
        string Propagator = \"SmartPropagatorAnyOpposite\"
        string TrackerRecHitBuilder = \"WithoutRefit\"
        string MuonRecHitBuilder = \"MuonRecHitBuilder\"
        string RefitDirection = \"insideOut\"
        untracked bool RefitRPCHits = false
    }
}

include \"Alignment/CommonAlignmentProducer/data/AlignmentProducer.cff\"
replace AlignmentProducer.doTracker = false
replace AlignmentProducer.doMuon = true
replace AlignmentProducer.tjTkAssociationMapTag = TrackRefitter
replace AlignmentProducer.algoConfig.minimumNumberOfHits = 50 # optimized (min in 10k-event sample)
replace AlignmentProducer.algoConfig.maxRelParameterError = 1e12
replace AlignmentProducer.maxLoops = 1

es_source = PoolDBESSource {
    using CondDBCommon
    untracked uint32 authenticationMethod = 1
    VPSet toGet = {
      {
          string record = \"DTAlignmentRcd\"
          string tag = \"DTAlignmentRcd\"
      },
      {
          string record = \"DTAlignmentErrorExtendedRcd\"
          string tag = \"DTAlignmentErrorExtendedRcd\"
      },
      {
          string record = \"CSCAlignmentRcd\"
          string tag = \"CSCAlignmentRcd\"
      },
      {
          string record = \"CSCAlignmentErrorExtendedRcd\"
          string tag = \"CSCAlignmentErrorExtendedRcd\"
      }
    }
}
replace AlignmentProducer.applyDbAlignment = true

block MuonAlignmentStationSelector = {
    PSet mb1 = {
        vdouble rRanges   = {370., 470.}
        vdouble zRanges   = {-560., 560.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet mb2 = {
        vdouble rRanges   = {470., 570.}
        vdouble zRanges   = {-560., 560.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet mb3 = {
        vdouble rRanges   = {570., 670.}
        vdouble zRanges   = {-560., 560.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet mb4 = {
        vdouble rRanges   = {670., 870.}
        vdouble zRanges   = {-560., 560.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me11 = {
        vdouble rRanges   = {50., 275.}
        vdouble zRanges   = {-700., -500., 500., 700.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me12 = {
        vdouble rRanges   = {275., 480.}
        vdouble zRanges   = {-750., -650., 650., 750.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me13 = {
        vdouble rRanges   = {480., 800.}
        vdouble zRanges   = {-750., -650., 650., 750.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me21 = {
        vdouble rRanges   = {50., 350.}
        vdouble zRanges   = {-875., -750., 750., 875.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me22 = {
        vdouble rRanges   = {350., 800.}
        vdouble zRanges   = {-875., -750., 750., 875.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me31 = {
        vdouble rRanges   = {50., 350.}
        vdouble zRanges   = {-980., -875., 875., 980.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me32 = {
        vdouble rRanges   = {350., 800.}
        vdouble zRanges   = {-980., -875., 875., 980.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
    PSet me41 = {
        vdouble rRanges   = {50., 350.}
        vdouble zRanges   = {-1100., -980., 980., 1100.}
        vdouble etaRanges = {} vdouble phiRanges = {} vdouble xRanges = {} vdouble yRanges = {}
    }
}
";

    close(FILE);
}

sub source_cff($dirname) {
    system("mkdir $dirname/sources");

    foreach $N (1..50) {
	open(FILE, sprintf("> $dirname/sources/job%d.cff", $N));

	printf FILE "source = PoolSource {
    untracked vstring fileNames = {
        \"$filetemplate\",
        \"$filetemplate\"
    }
}
untracked PSet maxEvents = {untracked int32 input = -1}
", ($N-1), ($N-1+50);

	close(FILE);
    } # end loop over N
}

sub collect_py($dirname, $pass, $iter) {
    open(FILE, "> $dirname/$pass/$iter/collect.py");
    print FILE "import Alignment.CommonAlignmentMonitor.muonHIP as muonHIP

njobs = range(1, 50+1)

merged_residual_hists = muonHIP.ROOT.TFile(\"$dirname\_$pass\_$iter.root\", \"recreate\")
iter1dir = merged_residual_hists.mkdir(\"iter1\")
wxresiddir = iter1dir.mkdir(\"wxresid_chamber\")
wyresiddir = iter1dir.mkdir(\"wyresid_chamber\")

wxresid = {}
wyresid = {}
for rawid in muonHIP.chambers.keys():
  wxresiddir.cd()
  wxresid[rawid] = muonHIP.ROOT.TH1F(\"wxresid_chamber_\%d\" \% rawid, \"wxresid_chamber_\%d\" \% rawid, 250, -5., 5.)
  if not muonHIP.chambers[rawid].barrel:
    wyresiddir.cd()
    wyresid[rawid] = muonHIP.ROOT.TH1F(\"wyresid_chamber_\%d\" \% rawid, \"wyresid_chamber_\%d\" \% rawid, 250, -5., 5.)

tfile = {}
for n in njobs:
  tfile[n] = muonHIP.ROOT.TFile(\"job\%d/$dirname\_$pass\_$iter.root\" \% n)

for rawid in muonHIP.chambers.keys():
  tl = muonHIP.ROOT.TList()
  for t in tfile.values():
    thehist = tfile[n].Get(\"iter1/wxresid_chamber/wxresid_chamber_\%d\" \% rawid)
    tl.Add(thehist)
  wxresid[rawid].Merge(tl)

  if not muonHIP.chambers[rawid].barrel:
    tl = muonHIP.ROOT.TList()
    for t in tfile.values():
      thehist = tfile[n].Get(\"iter1/wyresid_chamber/wyresid_chamber_\%d\" \% rawid)
      tl.Add(thehist)
    wyresid[rawid].Merge(tl)

merged_residual_hists.Write()
merged_residual_hists.Close()
";
    close(FILE);
}
