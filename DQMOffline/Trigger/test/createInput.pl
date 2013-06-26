#!/usr/bin/env perl

$runNum_ = $ARGV[0];
$string_ = $ARGV[1];


#configure here
#$ntpdir=  "/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2";
#$ntpdir=  "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/";
#$ntpdir=  "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/";
#$ntpdir=  "/store/data/Commissioning10/MinimumBias/RECO/v3/";
#$ntpdir=  "/store/data/Commissioning10/MinimumBias/RECO/v5/";
#$ntpdir=  "/store/data/Commissioning10/Cosmics/RECO/v3/";
#$ntpdir=  "/store/express/Commissioning10/ExpressPhysics/FEVT/v6/";
$ntpdir=  "/store/express/Commissioning10/ExpressPhysics/FEVT/v7/";

#####################
$tempDir = "./runs";

if(! $ARGV[0] ) {print "Usage: ./createInput.pl [RUN] [stringToSearch]\n"; exit;}
if(! $ARGV[1] ) {print "GETTING ALL FILES with sufix $string_!!!!!\n"; $string_ = "root";}

if ($string_ eq "dat") {
 $ntpdir= "/store/streamer/Data/A/";
}

$A = substr($runNum_,0,3);
$B = substr($runNum_,3);
     print "A=$A\n";
     print "B=$B\n";


     print "Checking directory: \"/castor/cern.ch/cms$ntpdir\" \n";
     print "nsls /castor/cern.ch/cms$ntpdir/000/$A/$B/ | grep $string_ > tmp.lis\n";
     system("nsls /castor/cern.ch/cms$ntpdir/000/$A/$B/ | grep $string_ > tmp.lis");
     open(FILE, "tmp.lis") or die "Can't open `..tmp.lis': $!";
     @v = <FILE>;
     $lines=0;

#open CFGFILE, "> $runNum_\_cfi.py";
#open CFGFILE, "> curRun_files_cfi.py";
     unless(-d $tempDir){
      system("mkdir ./runs");
     }
     open CFGFILE, "> ./runs/Run\_$runNum_\_cfi.py";
print CFGFILE "import FWCore.ParameterSet.Config as cms\n";
#print CFGFILE "maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )\n";
#print CFGFILE "readFiles = cms.untracked.vstring()\n"; 
#print CFGFILE "secFiles = cms.untracked.vstring()\n"; 
#print CFGFILE "source = cms.Source (\"NewEventStreamFileReader\",fileNames = readFiles, secondaryFileNames = secFiles)\n"; 
#     print CFGFILE "readFiles.extend( (\n"; 
if ($string_ eq "dat") {
     print CFGFILE "source = cms.Source (\"NewEventStreamFileReader\",\n";
}
if ($string_ eq "root") {
     print CFGFILE "source = cms.Source (\"PoolSource\",\n";
}
     print CFGFILE "fileNames = cms.untracked.vstring(\n"; 
#process.source = cms.Source("PoolSource",
##    fileNames = cms.untracked.vstring('/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/006BA396-9378-DE11-BBC8-000423D944FC.root')
     
     print "@v\n";
     $size=@v-1;
     foreach (@v) {
     chomp($_);
     if($lines<$size) { print CFGFILE "'$ntpdir/000/$A/$B/$_',\n"; }
     if($lines==$size) { print CFGFILE "'$ntpdir/000/$A/$B/$_'\n"; }
     $lines ++;
     }
     print CFGFILE "));\n"; 
#print CFGFILE "secFiles.extend( (\n"; 
#     print CFGFILE "))\n"; 
    
     close CFGFILE;
     
     print "\n";
     print "Total number of files read: $lines\n";
     system("rm -f tmp.lis");
system("cp -f runs/Run_$runNum_\_cfi.py curRun\_files\_cfi.py");
#system("cp -f $runNum_\_cfi.py ../python/");

FINE :
print "End submission...\n";
print "picked up files with sufix:   $string_ \n";
print "from dir: $ntpdir \n";
