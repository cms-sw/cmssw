#!/usr/bin/env perl

#read in config updates from ConfigUpdate.txt
open (UPDATES, "ConfigUpdate.txt");
while (<UPDATES>) {
    $DSOURCE_INPUT = $DSOURCE_INPUT . $_ ;
}
close(UPDATES);

$RUI_DUMMY = substr($ARGV[0],30);
$RUI = substr($RUI_DUMMY,0, 5);

$noise =  
"process TEST = {
 	source = DaqSource{ string reader = \"CSCFileReader\"
               	PSet pset = {untracked vstring $RUI ={\"$ARGV[0]\"}
                untracked string dataType  = \"DAQ\"
                untracked int32 input = -1
                $DSOURCE_INPUT
                untracked int32 firstEvent = 0
	}
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
