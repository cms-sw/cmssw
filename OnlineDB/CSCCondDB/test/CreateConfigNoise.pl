#!/usr/local/bin/perl

$noise =  
"process TEST = {
 	source = DaqSource{ string reader = \"CSCFileReader\"
               	 	untracked int32 maxEvents = -1
               	PSet pset = {untracked vstring fileNames ={\"$ARGV[0]\"}}
	}

	module cscunpacker = CSCDCCUnpacker {
        //untracked bool PrintEventNumber = false
	untracked bool Debug = false
	untracked int32 Verbosity = 0
	FileInPath theMappingFile = \"OnlineDB/CSCCondDB/test/csc_slice_test_map.txt\" 
	} 

        module analyzer = CSCNoiseMatrixAnalyzer {
		untracked int32 Verbosity = 0
		#change to true to send constants to DB !!
		untracked bool debug = false
	}

        
	path p = {cscunpacker,analyzer}
}";

print "$noise\n"; 
open(CONFIGFILE, ">CSCmatrix.cfg");
print CONFIGFILE "$noise";
close(CONFIGFILE);
