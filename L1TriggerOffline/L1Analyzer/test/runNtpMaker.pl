#!/usr/local/bin/perl

# This scrits must be used ./runNtpMaker.pl RunNumber
# It looks for streamer files from run with name $GLOBAL following
# the CMS conventions for the path name.
# You can set the number of jobs you want to run via $numJobs.
# You can set the number of data files you would like to process per job.
# The code runs locally and sequentially. Alternative script will be provided to
# submit jobs on the caf batch system. 

 
#------Configure here ---------------------------------------
#$GLOBAL="GlobalCruzet4";
#$GLOBAL="GlobalRAFF";
$GLOBAL="GlobalBeamCommissioning08";
$cfgdir=  "/afs/cern.ch/user/l/lorenzo/scratch0/CMSSW_2_1_4/src/L1TriggerOffline/L1Analyzer/test";
$nfiles= 10;  #number of files processed per job
$numJobs = 2; #number of jobs you want to submit (use -1 if all)
#-------------------------------------------------------------


if(! $ARGV[0] ) {print "Usage: ./runNtpMaker.pl RunNumber\n"; exit;}

$RUN = $ARGV[0]; 

$A = substr($RUN,0,2);
$ZERO = "0";
$A = $ZERO.$A;
$B = substr($RUN,2);

##look in castor and get list of files
system("nsls /castor/cern.ch/cms/store/data/$GLOBAL/HLTDEBUG/000/$A/$B/ > files.txt");
$count = `wc -l < files.txt`;
die "wc failed: $?" if $?;
chomp($count);
print "-----------------------------\n";
print "Total number of file: $count\n";
print "-----------------------------\n";

#read file list
open(MYFILE, "<files.txt");
$line=0;
$myn=0;
@names = ();
while(<MYFILE>)
{
$name = $_;
chomp($name);
push(@names, $name); #accumulate the files for i_th-job.

#print "$name";
$line++;
$job = $line%$nfiles; 
#print "line $line and job $job\n";
if($job==0 || eof(MYFILE)){
$total=@names;
$myn=$line/$nfiles;
if(eof(MYFILE)) {$myn=0;}
#print "myn $myn\n";


#create i_th configuration file 
open CFGFILE, "> l1prompt_$RUN\_$myn\_cfg.py";

print CFGFILE "import FWCore.ParameterSet.Config as cms\n";
print CFGFILE "\n";
print CFGFILE "process = cms.Process(\"L1Prompt\")\n";
print CFGFILE "\n";
print CFGFILE "process.load(\"L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi\")\n"; 
print CFGFILE "process.load(\"L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff\")\n"; 
print CFGFILE "process.load(\"L1TriggerOffline.L1Analyzer.gtUnpack_cff\")\n"; 
print CFGFILE "process.load(\"L1TriggerOffline.L1Analyzer.L1PromptAnalysis_cfi\")\n";
print CFGFILE "process.l1PromptAnalysis.OutputFile = '$RUN\_$myn.root'\n";
print CFGFILE "\n";
print CFGFILE "process.source = cms.Source(\"NewEventStreamFileReader\",\n";
print CFGFILE "fileNames = cms.untracked.vstring(\n";
$ii=0;
foreach $ll (@names)
{
$ii++;
#print "$ll";
if($ii != $total){
print CFGFILE "\'/store/data/$GLOBAL/HLTDEBUG/000/$A/$B/$ll\',\n";
} else
{
print CFGFILE "\'/store/data/$GLOBAL/HLTDEBUG/000/$A/$B/$ll\'\n";
}
}
@names=();
print CFGFILE ")\n";
print CFGFILE ")\n";
print CFGFILE "\n";
print CFGFILE "process.maxEvents = cms.untracked.PSet(\n";
print CFGFILE "    input = cms.untracked.int32(-1)\n";
print CFGFILE ")\n";
print CFGFILE "\n";
print CFGFILE "process.p = cms.Path(process.l1GtUnpack+process.l1GtEvmUnpack+process.l1PromptAnalysis)\n";
print CFGFILE "\n";
system("cmsRun l1prompt_$RUN\_$myn\_cfg.py > $RUN\_$myn.log");
print "cmsRun l1prompt_$RUN\_$myn\_cfg.py > $RUN\_$myn.log\n";
if($myn==$numJobs) {exit;}

}
}
print "End submission...\n";

