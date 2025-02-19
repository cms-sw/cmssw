#!/usr/bin/env perl

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
$GLOBAL="Commissioning08";
$pathToFiles="Cosmics/RAW/v1";
#$eventSource="\"NewEventStreamFileReader\"";
$eventSource="\"PoolSource\"";
$nfiles= 100;  #number of files processed per job
$numJobs = 2; #number of jobs you want to submit (use -1 if all)
$nEvents = -1; #number of events you want to run in each job (use -1 if all)
$lumiMin = -1; #select based on lumi number, if all put -1
$lumiMax = 9999; #select based on lumi number, if all put a high number 
#-------------------------------------------------------------


if(! $ARGV[0] ) {print "Usage: ./runNtpMaker.pl RunNumber\n"; exit;}

$RUN = $ARGV[0]; 

$A = substr($RUN,0,2);
$ZERO = "0";
$A = $ZERO.$A;
$B = substr($RUN,2);

##look in castor and get list of files
system("nsls /castor/cern.ch/cms/store/data/$GLOBAL/$pathToFiles/000/$A/$B/ > files.txt");
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
$job=1; # so that nothing is processed if no lumi section is found
@names = ();
while(<MYFILE>)
{
$name = $_;
chomp($name);

#check lumi section assuming std output naming convention
$indx = index($name, '.');
$indx = index($name, '.',$indx+1);
$lumi = substr($name,$indx+1,4);
$lumi = $lumi+0; #needed to convert string to number

if($lumi > $lumiMin && $lumi < $lumiMax) {
#print "lumi is $lumi\n"; 
push(@names, $name); #accumulate the files for i_th-job.
#print "$name";
$line++;
$job = $line%$nfiles; 

  if(eof(MYFILE)) 
  {
    $job=0; #so that if there are not enough files wrt $nfiles
  }         #it uses those.
}
#end lumo part

if($job==0){

$total=@names;
$myn=$line/$nfiles;

if(eof(MYFILE) || int($myn)==0) {$myn=0;}

#create i_th configuration file 
open CFGFILE, "> l1prompt_$RUN\_$myn\_cfg.py";

print CFGFILE "import FWCore.ParameterSet.Config as cms\n";
print CFGFILE "\n";
print CFGFILE "process = cms.Process(\"L1Prompt\")\n";
print CFGFILE "\n";
print CFGFILE "process.load(\"L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi\")\n"; 
print CFGFILE "process.load(\"L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff\")\n"; 
print CFGFILE "process.load(\"L1TriggerOffline.L1Analyzer.dttfUnpack_cff\")\n"; 
print CFGFILE "process.load(\"L1TriggerOffline.L1Analyzer.gtUnpack_cff\")\n"; 
print CFGFILE "process.load(\"L1TriggerOffline.L1Analyzer.gctUnpack_cff\")\n"; 
print CFGFILE "process.load(\"L1TriggerOffline.L1Analyzer.L1PromptAnalysis_cfi\")\n";
print CFGFILE "process.l1PromptAnalysis.OutputFile = '$RUN\_$myn.root'\n";
print CFGFILE "\n";
print CFGFILE "process.source = cms.Source($eventSource,\n";
print CFGFILE "fileNames = cms.untracked.vstring(\n";
$ii=0;
foreach $ll (@names)
{
$ii++;
#print "$ll";
if($ii != $total){
print CFGFILE "\'/store/data/$GLOBAL/$pathToFiles/000/$A/$B/$ll\',\n";
} else
{
print CFGFILE "\'/store/data/$GLOBAL/$pathToFiles/000/$A/$B/$ll\'\n";
}
}
@names=();
print CFGFILE ")\n";
print CFGFILE ")\n";
print CFGFILE "\n";
print CFGFILE "process.maxEvents = cms.untracked.PSet(\n";
print CFGFILE "    input = cms.untracked.int32($nEvents)\n";
print CFGFILE ")\n";
print CFGFILE "\n";
print CFGFILE "process.p = cms.Path(process.l1GtUnpack+process.l1GctHwDigis+process.l1GtEvmUnpack+process.l1dttfunpack+process.l1PromptAnalysis)\n";
print CFGFILE "\n";
system("cmsRun l1prompt_$RUN\_$myn\_cfg.py > $RUN\_$myn.log");
print "cmsRun l1prompt_$RUN\_$myn\_cfg.py > $RUN\_$myn.log\n";
if($myn==$numJobs) {exit;}

}
}
print "End submission...\n";

