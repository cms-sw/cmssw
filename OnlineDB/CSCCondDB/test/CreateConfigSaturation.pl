#!/usr/bin/env perl

open(RUNFILE, "tempFile.txt");

@runs = readline(RUNFILE);
close(RUNFILE);

foreach $run (@runs)  
{
$runName = substr($run,0,45);
$runName = "/tmp/csccalib/" . $runName;
$list = "$list" . "\"" . "$runName" . "\",";
}  

#cut off the last comma from the run list. 

$list = substr($list,0,-1);

#read in config updates from ConfigUpdate.txt
open (UPDATES, "ConfigUpdate.txt");
while (<UPDATES>) {
    $DSOURCE_INPUT = $DSOURCE_INPUT . $_ ;
}
close(UPDATES);

$RUI_DUMMY = substr($ARGV[0],30);
$RUI = substr($RUI_DUMMY,0, 5);

$saturation = 

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
        untracked int32 debugVerbosity = 0 
        FileInPath theMappingFile = \"OnlineDB/CSCCondDB/test/csc_slice_test_map.txt\"
        }
 
        module analyzer = CSCSaturationAnalyzer {
                untracked int32 Verbosity = 0
                #change to true to send constants to DB !!
                untracked bool debug = false
        } 
 
         
        path p = {cscunpacker,analyzer} 
}";


print "$saturation\n"; 

#output .cfg file
open(CONFIGFILE, ">CSCsaturation.cfg");
print CONFIGFILE "$saturation";
close(CONFIGFILE); 

#read in dummy runs list, for reading. 
open(DUMMYRUNSOLD, "GoodSaturationRunsDummy.txt");
@dummyruns = readline(DUMMYRUNSOLD);
close(DUMMYRUNSOLD);

#compare the runs just processed, in @runs to the old list of runs, in @dummyruns.
#if the just-processed run is in the file, do not re-write it. otherwise, write it. 

#perl cannot compare a string to an array, so this reads in the array @runs as a 
#variable. all members is @runs are now in text in $runVar. 
$runVar="";
foreach $run (@runs){
    $runVar=$runVar . $run
    }
print "$runVar\n";

#initialize variable which will become the new run list for processing.
$dummyrunsNew = "";
#read in each line in the input dummyrun file. if it is in runVar, i.e. if it
#was processed in this config, do NOT rewrite it. otherwise, write it. 
foreach $dummyrun (@dummyruns){  
    #this if reads as "if the string pointed to by the variable dummyrun is contained 
    #in the string pointed to by the variable runVar, then..."
	if ( $runVar =~ m/($dummyrun)/){ print "$dummyrun processed\n"}
	else { 
	    print "$dummyrun NOT processed\n";
	    $dummyrunsNew = $dummyrunsNew . $dummyrun;
	       };
    }


print "\n\n ******dummrunsNew******\n$dummyrunsNew";

#re-open dummy runs list, for writing, in order to remove already processed runs. 
open(DUMMYRUNS, ">GoodSaturationRunsDummy.txt");
print DUMMYRUNS "$dummyrunsNew";
close(DUMMYRUNS);

