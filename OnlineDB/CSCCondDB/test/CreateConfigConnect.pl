#!/usr/local/bin/perl

$connect=
"process TEST = {
        source = DaqSource{ string reader = \"CSCFileReader\"
                        untracked int32 maxEvents = -1
                PSet pset = {untracked vstring fileNames ={\"$ARGV[0]\"}}
        }
 
        module cscunpacker = CSCDCCUnpacker {
                //untracked bool PrintEventNumber = false
                untracked bool Debug = false
                FileInPath theMappingFile = \"OnlineDB/CSCCondDB/test/csc_slice_test_map.txt\"
                untracked int32 Verbosity = 0
                untracked uint32 ErrorMask = 3754946559
                untracked uint32 ErrorMask = 3754946559
        }
 
        module analyzer = CSCCFEBConnectivityAnalyzer {
                untracked int32 Verbosity = 0
                #change to true to send constants to DB !!
                untracked bool debug = false
        }
 
        path p = {cscunpacker,analyzer}
 
}";

print "$connect\n";
open(CONFIGFILE, ">CSCCFEBconnect.cfg");
print CONFIGFILE "$connect";
close(CONFIGFILE);
 
