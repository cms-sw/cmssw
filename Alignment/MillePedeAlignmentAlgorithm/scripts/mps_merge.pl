#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     03-Jul-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.14 $
#     $Date: 2008/03/25 16:15:57 $
#
#  produce cfg file for merging run
#
#  Usage:
#
#  mps_merge.pl inCfg mergeCfg mergeDir njobs



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
$nn = ($body =~ s/alignment\.log/alignment\_merge\.log/g);
## print "alignment.log nn= $nn\n";

# change mode to pede
$nn = ($body =~ s/mode \= \"mille\"/mode \= \"pede\"/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n    replace MillePedeAlignmentAlgorithm.mode = \"pede\" ";
  print "No MillePedeAlignmentAlgorithm.mode directive found, adding one to replace block\n";
}

# set output directory
$nn = ($body =~ s/string fileDir \= \"\"/string fileDir \= \"$mergeDir\"/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n   replace MillePedeAlignmentAlgorithm.fileDir = \"$mergeDir\"";
  print "No MillePedeAlignmentAlgorithm.fileDir directive found, adding one to replace block\n";
}

# blank binary output file string
$nn = ($body =~ s/binaryFile \= \".+?\"/binaryFile \= \"\"/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n   replace MillePedeAlignmentAlgorithm.binaryFile = \"\"";
  print "No MillePedeAlignmentAlgorithm.binaryFile directive found, adding one to replace block\n";
}

# build list of binary files
$binaryList = "";
for ($i=1; $i<=$nJobs; ++$i) {
  $sep = ",\n                ";
  if ($i == $nJobs) { $sep = "" ;}

  $newName = sprintf "milleBinary%03d.dat",$i;
  print "Adding $newName to list of binary files\n";
  $newLine = "\"$newName\"$sep";
  $binaryList = "$binaryList $newLine";
  # create symbolic link
  $jobDirName = sprintf "job%03d",$i;
  ## system "cd $mergeDir; ln -s ../$jobDirName/milleBinary.dat $newName; cd -";
}

# replace list of binary files
$nn = ($body =~ s/mergeBinaryFiles = \{\}/mergeBinaryFiles = \{ $binaryList \}/);

if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n   replace MillePedeAlignmentAlgorithm.mergeBinaryFiles = \{ $binaryList \}";
  print "No MillePedeAlignmentAlgorithm.mergeBinaryFiles directive found, adding one to replace block\n";
}

# set merging of tree files
$nn = ($body =~ s/treeFile +\= \".+?\"/treeFile \= \"treeFile\_merge\.root\"/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n   replace MillePedeAlignmentAlgorithm.treeFile = \"treeFile\_merge\.root\" ";
  print "No MillePedeAlignmentAlgorithm.treeFile directive found, adding one to replace block\n";
}

# build list of tree files
$treeList = "";
for ($i=1; $i<=$nJobs; ++$i) {
  $sep = ",\n                ";
  if ($i == $nJobs) { $sep = "" ;}

#GF  $newName = sprintf "treeFile%03d.dat",$i;
  $newName = sprintf "treeFile%03d.root",$i;
  $newLine = "\"$newName\"$sep";
  $treeList = "$treeList $newLine";
  # create symbolic link
  $jobDirName = sprintf "job%03d",$i;
  $result = `cd $mergeDir; rm -f ../$jobDirName/$newName; ln -s ../$jobDirName/treeFile.root $newName; cd -`;
}

# replace list of tree files
$nn = ($body =~ s/mergeTreeFiles += \{\}/mergeTreeFiles = \{ $treeList \}/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n   replace MillePedeAlignmentAlgorithm.mergeTreeFiles = { $treeList }";
    print "No MillePedeAlignmentAlgorithm.mergeTreeFiles directive found, adding one to replace block\n";
}

# replace name of monitor file
$nn = ($body =~ s/monitorFile \= \".+?\"/monitorFile \= \"millePedeMonitor\_merge\.root\"/);
if ($nn != 1) {
  $replaceBlock = "$replaceBlock\n   replace MillePedeAlignmentAlgorithm.monitorFile = \"millePedeMonitor_merge.root\" ";
  print "No MillePedeAlignmentAlgorithm.monitorFile directive found, adding one to replace block\n";
}


# replace name of pede steer file
$nn = ($body =~ m/steerFile \=/);
$nn = ($body =~ s/steerFile \= +\".+?\"/steerFile \= \"pedeSteer\_merge\"/);
if ($nn != 1) {
    print "No pede steer file directive found, use default\n";
}

# replace name of pede dump file
$nn = ($body =~ s/pedeDump \= +\".+?\"/pedeDump \= \"pede\_merge.dump\"/);
if ($nn != 1) {
    print "No pede dump file directive found, use default\n";
}

# replace source directive... this is nasty.
$newSource = "source = EmptySource {}\n"
    ."    untracked PSet maxEvents = {untracked int32 input = 0}";
$nn = ($body =~ s/source \= PoolSource.+?\}.+?\}/$newSource/s);
if ($nn != 1) {
    print "No source directive found, use default\n";
}

$nn = ($body =~ s/#MILLEPEDEBLOCK/$replaceBlock/);


# store the output file
open OUTFILE,">$mergeCfg";
print OUTFILE $body;
close OUTFILE;

