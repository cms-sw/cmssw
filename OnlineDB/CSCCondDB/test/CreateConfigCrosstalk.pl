#!/usr/bin/env perl

#read in config updates from ConfigUpdate.txt
open (UPDATES, "ConfigUpdate.txt");
while (<UPDATES>) {
    $DSOURCE_INPUT = $DSOURCE_INPUT . $_ ;
}
close(UPDATES);

$RUI_DUMMY = substr($ARGV[0],30);
$RUI = substr($RUI_DUMMY,0, 5);

$xtalk = 
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
                InputTag InputObjects = source
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
