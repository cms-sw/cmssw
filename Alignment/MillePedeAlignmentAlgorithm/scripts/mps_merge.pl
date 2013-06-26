#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     03-Jul-2007
#     A. Parenti, DESY Hamburg    24-Apr-2008
#
#     $Revision: 1.29 $
#     $Date: 2013/04/22 11:26:31 $
#
#  produce cfg file for merging run
#
#  Usage:
#
#  mps_merge.pl [-c] inCfg mergeCfg mergeDir njobs

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

$inCfg = "undefined";
$mergeCfg = "undefined";
$mergeDir = "undefined";
$nJobs = "undefined";

# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    elsif ($arg =~ "c") {
# Check which jobs are "OK" and write just them to the cfg file
      $checkok = 1;
    }
    elsif ($arg =~ "d") {
      $localdir = 1;
    }
    elsif ($arg =~ "u") {
      $updateDb = 1;
    }
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $inCfg = $arg;
    }
    elsif ($i eq 2) {
      $mergeCfg = $arg;
    }
    elsif ($i eq 3) {
      $mergeDir = $arg;
    }
    elsif ($i eq 4) {
      $nJobs = $arg;
    }

  }
}

if ($nJobs eq "undefined") {
  print "Insufficient information given\n";
  exit 1;
}

if ($checkok == 1) {
  read_db();
}

# open the input file
open INFILE,"$inCfg";
undef $/;  # undefining the INPUT-RECORD_SEPARATOR means slurp whole file
$body = <INFILE>;  # read whole file
close INFILE;
$/ = "\n"; # back to normal

# protect the #MILLEPEDEBLOCK directive by renaming it
$body =~ s/\#MILLEPEDEBLOCK/PROTECTEDMILLEPEDEBLOCK/mg;

# remove comment lines
$nn = ($body =~ s/^\s*\#.*$/COMMENTLINEREMOVED/mg);
$nn = ($body =~ s/COMMENTLINEREMOVED\n//mg);
$nn = ($body =~ s/COMMENTLINEREMOVED//mg);

# restore the #MILLEPEDEBLOCK directive by renaming it again
$body =~ s/PROTECTEDMILLEPEDEBLOCK/\#MILLEPEDEBLOCK/mg;

# create pede dir
unless (-d $mergeDir) {
    system "mkdir $mergeDir";
}

# initialize replaceBlock
$replaceBlock = "";

# change name of log file
# not anymore: there cannot be '.' in python names and it is not needed anyway
# $nn = ($body =~ s/alignment\.log/alignment\_merge\.log/g);
## print "alignment.log nn= $nn\n";

# replace "save to DB" directives
$saveAlignmentConstants = "from CondCore.DBCommon.CondDBSetup_cfi import *\n"
                        . "process.PoolDBOutputService = cms.Service(\"PoolDBOutputService\",\n"
                        . "    CondDBSetup,\n"
                        . "    timetype = cms.untracked.string('runnumber'),\n"
                        . "    connect = cms.string('sqlite_file:alignments_MP.db'),\n"
                        . "    toPut = cms.VPSet(cms.PSet(\n"
                        . "        record = cms.string('TrackerAlignmentRcd'),\n"
                        . "        tag = cms.string('Alignments')\n"
                        . "    ),\n"
                        . "        cms.PSet(\n"
                        . "            record = cms.string('TrackerAlignmentErrorRcd'),\n"
                        . "            tag = cms.string('AlignmentErrors')\n"
                        . "        ),
        cms.PSet(
            record = cms.string('TrackerSurfaceDeformationRcd'),
            tag = cms.string('Deformations')
        ),
        cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd_peak'),
            tag = cms.string('SiStripLorentzAngle_peak')
        ),
        cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd_deco'),
            tag = cms.string('SiStripLorentzAngle_deco')
        ),
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string('SiPixelLorentzAngle')
        ),
        cms.PSet(
            record = cms.string('SiStripBackPlaneCorrectionRcd'),
            tag = cms.string('SiStripBackPlaneCorrection')
        )
)\n"
                        . ")\n"
                        . "process.AlignmentProducer.saveToDB = True";

$nn = ($body =~ /AlignmentProducer\.saveToDB.+?False/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n$saveAlignmentConstants";
  print "No AlignmentProducer.saveToDB directive found, adding saveToDB=True to replace block\n";
}

# change mode to pede
$nn = ($body =~ s/mode \= \'mille\'/mode \= \'pede\'/);
if ($nn != 1) { # 
  $nn = ($body =~ s/mode \= \'full\'/mode \= \'pede\'/); # maybe it was set to full mode in template...
  if ($nn != 1) {
    $replaceBlock = "$replaceBlock\nprocess.AlignmentProducer.algoConfig.mode = 'pede'";
    print "No AlignmentProducer.algoConfig.mode directive found, adding one to replace block\n";
  }
}

# blank binary output file string
$nn  = ($body =~ s/binaryFile \= \'.+?\'/binaryFile \= \'\'/);
$nn += ($body =~ s/binaryFile \= \".+?\"/binaryFile \= \'\'/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.AlignmentProducer.algoConfig.binaryFile = \'\'\n";
  print "No AlignmentProducer.algoConfig.binaryFile directive found, adding one to replace block\n";
}

# build list of binary files
$binaryList = "";
$iIsOk = 1;
for ($i=1; $i<=$nJobs; ++$i) {
  $sep = ",\n                ";
  if ($iIsOk == 1) { $sep = "\n                " ;}

  if ($checkok==1 && @JOBSTATUS[$i-1] ne "OK") {next;}
  ++$iIsOk;

  $newName = sprintf "milleBinary%03d.dat",$i;
  if($JOBSP2[$i-1] ne "" && defined $JOBSP2[$i-1])
    {
      my $weight = $JOBSP2[$i-1];
      print "Adding $newName to list of binary files using weight $weight\n";
      $binaryList = "$binaryList$sep\'$newName -- $weight\'";
    }
  else
    {
      print "Adding $newName to list of binary files\n";
      $binaryList = "$binaryList$sep\'$newName\'";
    }
}

# replace list of binary files
$nn  = ($body =~ s/mergeBinaryFiles = \[(.|\n)*?\]/mergeBinaryFiles = \[$binaryList\]/);
$nn += ($body =~ s/mergeBinaryFiles = cms.vstring\(\)/mergeBinaryFiles = \[$binaryList\]/);

if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.AlignmentProducer.algoConfig.mergeBinaryFiles = \[$binaryList\]";
  print "No AlignmentProducer.algoConfig.mergeBinaryFiles directive found, adding one to replace block\n";
}

# set merging of tree files
$nn = ($body =~ s/process.AlignmentProducer.algoConfig.treeFile = \'.+?\'/process.AlignmentProducer.algoConfig.treeFile = \'treeFile_merge.root\'/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.AlignmentProducer.algoConfig.treeFile = \'treeFile_merge.root\'";
  print "No AlignmentProducer.algoConfig.treeFile directive found, adding one to replace block\n";
}

# build list of tree files
$treeList = "";
$iIsOk = 1;
for ($i=1; $i<=$nJobs; ++$i) {
  $sep = ",\n                ";
  if ($iIsOk == 1) { $sep = "\n                " ;}

  if ($checkok==1 && @JOBSTATUS[$i-1] ne "OK") {next;}
  ++$iIsOk;

  $newName = sprintf "treeFile%03d.root",$i;
  $treeList = "$treeList$sep\'$newName\'";
}

# replace list of tree files
$nn = ($body =~ s/mergeTreeFiles = \[(.|\n)*?\]/mergeTreeFiles = \[$treeList\]/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.AlignmentProducer.algoConfig.mergeTreeFiles = \[$treeList\]";
  print "No AlignmentProducer.algoConfig.mergeTreeFiles directive found, adding one to replace block\n";
}

# replace name of monitor file
$nn  = ($body =~ s/monitorFile \= \'.+?\'/monitorFile \= \'millePedeMonitor\_merge\.root\'/);
$nn += ($body =~ s/monitorFile \= \".+?\"/monitorFile \= \'millePedeMonitor\_merge\.root\'/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.AlignmentProducer.algoConfig.monitorFile = \'millePedeMonitor_merge.root\'";
  print "No AlignmentProducer.algoConfig.monitorFile directive found, adding one to replace block\n";
}


# replace name of pede steer file
$nn = ($body =~ m/steerFile \=/);
$nn = ($body =~ s/steerFile \= +\'.+?\'/steerFile \= \'pedeSteer\_merge\'/);
if ($nn != 1) {
    print "No pede steer file directive found, use default\n";
}

# replace name of pede dump file
$nn = ($body =~ s/pedeDump \= +\'.+?\'/pedeDump \= \'pede\_merge.dump\'/);
if ($nn != 1) {
    print "No pede dump file directive found, use default\n";
}


# AP 03.08.2010 - Temporary hack from Jula Draeger et al., in order to run event setup in CMSSW >= 351
# Set maxevents to one
$nn = ($body =~ s/ *?process.maxEvents *?= *?cms.untracked.PSet\(.*?\n.+?\n.*?\).*?/process.maxEvents = cms.untracked.PSet(\n    input = cms.untracked.int32(1)\n)/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.maxEvents = cms.untracked.PSet(\n    input = cms.untracked.int32(1)\n)";
  print "No process.maxEvents directive found, adding one to replace block\n";
}

# Make source an EmptySource
$nn  = ($body =~ /process.source = cms.Source\(\"EmptySource\"\)/);
$nn += ($body =~ /process.source = cms.Source\(\'EmptySource\'\)/);

if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.source = cms.Source\(\"EmptySource\"\)";
  print "No cms.Source\(\"EmptySource\"\) directive found, adding one to replace block\n";
  $replaceBlock = "$replaceBlock\nprocess.dump = cms.EDAnalyzer\(\"EventContentAnalyzer\"\)";
  $replaceBlock = "$replaceBlock\nprocess.p = cms.Path\(process.dump\)";
}

$nn = ($body =~ s/#MILLEPEDEBLOCK/$replaceBlock/);


# store the output file
open OUTFILE,">$mergeCfg";
print OUTFILE $body;
close OUTFILE;

