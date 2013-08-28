#!/usr/bin/env perl


#configure here
$ntpdir=  "/store/streamer/RunPrep09/A";

#####################


$runNum_ = $ARGV[0];
$string_ = $ARGV[1];

if(! $ARGV[0] ) {print "Usage: ./createInput.pl [RUN] [stringToSearch]\n"; exit;}
if(! $ARGV[1] ) {print "GETTING ALL FILES!!!!!\n"; $string_ = "dat";}

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

     open CFGFILE, "> $runNum_\_cfi.py";
     print CFGFILE "import FWCore.ParameterSet.Config as cms\n";
     print CFGFILE "maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )\n";
     print CFGFILE "readFiles = cms.untracked.vstring()\n"; 
     print CFGFILE "secFiles = cms.untracked.vstring()\n"; 
     print CFGFILE "source = cms.Source (\"NewEventStreamFileReader\",fileNames = readFiles, secondaryFileNames = secFiles)\n"; 
     print CFGFILE "readFiles.extend( (\n"; 
     
     print "@v\n";
     $size=@v-1;
     foreach (@v) {
     chomp($_);
     if($lines<$size) { print CFGFILE "'$ntpdir/000/$A/$B/$_',\n"; }
     if($lines==$size) { print CFGFILE "'$ntpdir/000/$A/$B/$_'\n"; }
     $lines ++;
     }
     print CFGFILE "));\n"; 
     print CFGFILE "secFiles.extend( (\n"; 
     print CFGFILE "))\n"; 
    
     close CFGFILE;
     
     print "\n";
     print "Total number of files read: $lines\n";
     system("rm -f tmp.lis");
     system("cp -f $runNum_\_cfi.py ../python/");

FINE :
print "End submission...\n";
