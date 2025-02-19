#!/usr/bin/env perl
 
$output_ = substr($ARGV[0],0,-3) . "root";
$output = substr($output_,14);

$afeb =
"process PROD = {
 
source = DaqSource{ string reader = \"CSCFileReader\"
                        untracked int32 maxEvents = -1
PSet pset = {untracked vstring fileNames = {\"$ARGV[0]\"} }
        }
 
        module cscunpacker = CSCDCCUnpacker {
               untracked bool Debug = false
               untracked bool PrintEventNumber = false
               FileInPath theMappingFile = \"OnlineDB/CSCCondDB/test/csc_slice_test_map.txt\"
               untracked bool UseExaminer = false
        }
 
        module analyzer = CSCAFEBAnalyzer {
        string TestName=\"AFEBThresholdScan\" 
#         string TestName=\"AFEBConnectivity\"
        string HistogramFile = \"$output\" 
#         string HistogramFile = \"$output\"
 
         InputTag CSCSrc = cscunpacker:MuonCSCWireDigi
         }
    
        path p = {cscunpacker,analyzer}
}";

print "$afeb\n";
open(CONFIGFILE, ">CSCAFEBAnalysis.cfg");
print CONFIGFILE "$afeb";
close(CONFIGFILE);

