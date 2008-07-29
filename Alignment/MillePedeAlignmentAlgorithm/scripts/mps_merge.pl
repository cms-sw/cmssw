#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     03-Jul-2007
#     A. Parenti, DESY Hamburg    24-Apr-2008
#
#     $Revision: 1.13 $
#     $Date: 2008/07/21 20:07:27 $
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
$saveAlignmentConstants = "process.load(\"CondCore.DBCommon.CondDBSetup_cfi\")\n"
                        . "process.PoolDBOutputService = cms.Service(\"PoolDBOutputService\",\n"
                        . "    process.CondDBSetup,\n"
                        . "    timetype = cms.untracked.string('runnumber'),\n"
                        . "    connect = cms.string('sqlite_file:alignments_MP.db'),\n"
                        . "    toPut = cms.VPSet(cms.PSet(\n"
                        . "        record = cms.string('TrackerAlignmentRcd'),\n"
                        . "        tag = cms.string('Alignments')\n"
                        . "    ),\n"
                        . "        cms.PSet(\n"
                        . "            record = cms.string('TrackerAlignmentErrorRcd'),\n"
                        . "            tag = cms.string('AlignmentErrors')\n"
                        . "        ))\n"
                         .")";

$nn = ($body =~ /AlignmentProducer\.saveToDB.+?false/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n$saveAlignmentConstants";
  print "No AlignmentProducer.saveToDB directive found, adding saveToDB=true to replace block\n";
}

# change mode to pede
$nn = ($body =~ s/mode \= \'mille\'/mode \= \'pede\'/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.MillePedeAlignmentAlgorithm.mode = 'pede'";
  print "No MillePedeAlignmentAlgorithm.mode directive found, adding one to replace block\n";
}

# blank binary output file string
$nn = ($body =~ s/binaryFile \= \'.+?\'/binaryFile \= \'\'/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.MillePedeAlignmentAlgorithm.binaryFile = \'\'\n";
  print "No MillePedeAlignmentAlgorithm.binaryFile directive found, adding one to replace block\n";
}

# build list of binary files
$binaryList = "";
for ($i=1; $i<=$nJobs; ++$i) {
  $sep = ", ";
  if ($i == 1) { $sep = "" ;}

  if ($checkok==1 && @JOBSTATUS[$i-1] ne "OK") {next;}

  $newName = sprintf "milleBinary%03d.dat",$i;
  print "Adding $newName to list of binary files\n";
  $binaryList = "$binaryList$sep\'$newName\'";
  # create symbolic link
  $jobDirName = sprintf "job%03d",$i;
  ## system "cd $mergeDir; ln -s ../$jobDirName/milleBinary.dat $newName; cd -";
}

# replace list of binary files
$nn = ($body =~ s/mergeBinaryFiles = \[(.|\n)*?\]/mergeBinaryFiles = \[$binaryList\]/);

if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.MillePedeAlignmentAlgorithm.mergeBinaryFiles = \[$binaryList\]";
  print "No MillePedeAlignmentAlgorithm.mergeBinaryFiles directive found, adding one to replace block\n";
}

# set merging of tree files
$nn = ($body =~ s/process.MillePedeAlignmentAlgorithm.treeFile = \'.+?\'/process.MillePedeAlignmentAlgorithm.treeFile = \'treeFile_merge.root\'/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.MillePedeAlignmentAlgorithm.treeFile = \'treeFile_merge.root\'";
  print "No MillePedeAlignmentAlgorithm.treeFile directive found, adding one to replace block\n";
}

# build list of tree files
$treeList = "";
for ($i=1; $i<=$nJobs; ++$i) {
  $sep = ", ";
  if ($i == 1) { $sep = "" ;}

  if ($checkok==1 && @JOBSTATUS[$i-1] ne "OK") {next;}

  $newName = sprintf "treeFile%03d.root",$i;
  $treeList = "$treeList$sep\'$newName\'";
  # create symbolic link
  $jobDirName = sprintf "job%03d",$i;
  $result = `cd $mergeDir; rm -f ../$jobDirName/$newName; ln -s ../$jobDirName/treeFile.root $newName; cd -`;
}

# replace list of tree files
$nn = ($body =~ s/mergeTreeFiles = \[(.|\n)*?\]/mergeTreeFiles = \[$treeList\]/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.MillePedeAlignmentAlgorithm.mergeTreeFiles = \[$treeList\]";
  print "No MillePedeAlignmentAlgorithm.mergeTreeFiles directive found, adding one to replace block\n";
}

# replace name of monitor file
$nn = ($body =~ s/monitorFile \= \'.+?\'/monitorFile \= \'millePedeMonitor\_merge\.root\'/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\nprocess.MillePedeAlignmentAlgorithm.monitorFile = \'millePedeMonitor_merge.root\'";
  print "No MillePedeAlignmentAlgorithm.monitorFile directive found, adding one to replace block\n";
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


# Remove any existing maxEvents directive...
$nn = ($body =~ s/process.maxEvents = cms.untracked.PSet\(\n.+?\n\)//);

$nn = ($body =~ s/process.source = cms.Source\(.+?\n\)/process.source = cms.Source\(\"EmptySource\"\)/s);

if ($nn != 1) {
    print "No source directive found, use default\n";
}

$nn = ($body =~ s/#MILLEPEDEBLOCK/$replaceBlock/);


# store the output file
open OUTFILE,">$mergeCfg";
print OUTFILE $body;
close OUTFILE;

