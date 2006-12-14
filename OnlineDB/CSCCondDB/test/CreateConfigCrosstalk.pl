#!/usr/local/bin/perl

$xtalk = 
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
	} 

        module analyzer = CSCCrossTalkAnalyzer {
 		untracked int32 Verbosity = 0
		#change to true to send constants to DB !!
		untracked bool debug = false
	}

       	path p = {cscunpacker,analyzer}

}";

print "$xtalk\n"; 

open(CONFIGFILE, ">CSCxtalk.cfg");
print CONFIGFILE "$xtalk";
close(CONFIGFILE); 
