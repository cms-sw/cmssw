#!/usr/bin/env perl
#
# This perl script can be used to change the input file being
# played back by a Storage Manager playback system.
#
# 23-Aug-2007, KAB

use strict;

# trim function
sub trim {
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}

# string length function
sub strlen {
  my $string = shift;
  my $len = split //, $string;
  return $len;
}

# function to replace env vars with the values
sub replaceEnvVars {
  my $inputString = shift;
  my $workString = $inputString;

  my $startPos = index($workString, "\$", 0);
  while ($startPos >= 0 && $startPos < strlen($workString) - 1) {
    $startPos++;
    #print STDERR "start pos = $startPos\n";
    my $endPos = $startPos;
    my $candidateEnvVar = substr($workString,$startPos,($endPos-$startPos+1));
    my $envVarValue = $ENV{$candidateEnvVar};
    #print STDERR "_${candidateEnvVar}_${envVarValue}_\n";
    while ("$envVarValue" eq "" && $endPos < strlen($workString)-1) {
      $endPos++;
      $candidateEnvVar = substr($workString,$startPos,($endPos-$startPos+1));
      $envVarValue = $ENV{$candidateEnvVar};
      #print "_${candidateEnvVar}_${envVarValue}_\n";
    }
    if ($envVarValue) {
      $workString = substr($workString, 0, $startPos-1) . $envVarValue .
        substr($workString, $endPos+1);
    }

    $startPos = index($workString, "\$", $startPos);
  }

  return $workString;
}

############################################
# Ask the user to choose a new input file. #
############################################

# list all of the files in the "data" directory
my $dataDir = "data";
my @fullFileList;
if (opendir DATADIR, $dataDir) {
  @fullFileList = readdir DATADIR;
  closedir DATADIR;
}

# filter out unwanted files
my @candidateFileList;
foreach my $file (@fullFileList) {
  if ($file =~ m/root$/ || $file =~ m/dat$/) {
    push @candidateFileList, $file;
  }
}

# this no longer makes sense now that we allow free-form input
## exit if there are no candidate files
#if (@candidateFileList == 0) {
#  my $localDir = `pwd`;
#  chomp $localDir;
#  print STDERR "\n";
#  print STDERR "ERROR: No candidate files found in ${localDir}/${dataDir}.\n";
#  print STDERR "Aborting...\n";
#  exit;
#}

# display the candidate file names
@candidateFileList = sort(@candidateFileList);
my $idx = -1;
print STDOUT "\nPlease select a new input file from the following list:\n";
printf STDOUT "%3d. %s\n", $idx++, "<<< User-specified file >>>";
foreach my $file (@candidateFileList) {
  printf STDOUT "%3d. %s\n", $idx, $file;
  $idx++;
}

# get the selection from the user
print STDOUT "Selected file number: ";
my $selectedIndex = <STDIN>;
chomp $selectedIndex;
$selectedIndex = trim($selectedIndex);

# validate the selection
if ("$selectedIndex" eq "") {
  print STDERR "\n";
  print STDERR "No file was selected.\n";
  print STDERR "Aborting...\n";
  exit;
}
if (! ($selectedIndex =~ m/^-?\d+$/)) {
  print STDERR "\n";
  print STDERR "Invalid selection ($selectedIndex): value must be numeric.\n";
  print STDERR "Aborting...\n";
  exit;
}
if ($selectedIndex < -1 || $selectedIndex >= @candidateFileList) {
  print STDERR "\n";
  print STDERR "Invalid selection ($selectedIndex): value out of range.\n";
  print STDERR "Aborting...\n";
  exit;
}

# ask the user for the file name and path, if needed
my $filePath;
if ($selectedIndex == -1) {
  print STDOUT "\nEnter the full path of the new input file: ";
  $filePath = <STDIN>;
  chomp $filePath;
  $filePath = trim($filePath);
  #print STDERR "BEFORE = $filePath\n";
  $filePath = replaceEnvVars($filePath);
  #print STDERR "AFTER = $filePath\n";
}
else {
  my $selectedFile = $candidateFileList[$selectedIndex];
  $filePath = "${dataDir}/${selectedFile}";
}

# if the user entered an LFN, output a warning
if ($filePath =~ m/^\/store\//) {
  print STDERR "\n";
  print STDERR "************************************************************\n";
  print STDERR "You have entered an LFN.  This script can not verify that " .
    "the path is correct.\n";
  print STDERR "As such, you should be careful to check that events are " .
    "read in correctly,\n";
  print STDERR "*and* that the builder unit log file doesn't fill up with error messages.\n";
  print STDERR "************************************************************\n";
}

# otherwise, check that the file is accessible
else {
  # cross-check that the selected file is OK
  if (! (-e "$filePath")) {
    print STDERR "\n";
    print STDERR "Unable to find selected file ($filePath).\n";
    print STDERR "Aborting...\n";
    exit;
  }
  if (-d "$filePath") {
    print STDERR "\n";
    print STDERR "The selected file ($filePath) is a directory, not a file.\n";
    print STDERR "Aborting...\n";
    exit;
  }
}

#######################################
# Ask the user for the new run number #
#######################################

print STDOUT "\nEnter the new run number: ";
my $newRunNumber = <STDIN>;
chomp $newRunNumber;
$newRunNumber = trim($newRunNumber);

# validate the run number
if (! $newRunNumber) {
  print STDERR "\n";
  print STDERR "No run number was specified.\n";
  print STDERR "Aborting...\n";
  exit;
}
if (! ($newRunNumber =~ m/^\d+$/)) {
  print STDERR "\n";
  print STDERR "Invalid run number ($newRunNumber): value must be numeric.\n";
  print STDERR "Aborting...\n";
  exit;
}

###############################################
# Update the run number in the SM config file #
###############################################

my $inputFile = "cfg/sm_playback.xml";
my $outputFile = "${inputFile}.modified";

open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

while (my $line = <FILEIN>) {
    chomp $line;

    # replace the run number, if needed
    if ($line =~ m/^(.*)\<runNumber(.*)\>\s*\d+\s*\<\/runNumber\>(.*)$/) {
      $line = $1 . "<runNumber" . $2 . ">" . $newRunNumber .
        "</runNumber>" . $3;
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
}

close FILEIN;
close FILEOUT;

rename $inputFile, "${inputFile}.old";
rename $outputFile, $inputFile;

###############################################################
# Update the filter unit playback file with the new data file #
###############################################################

my $targetFile = "cfg/fu_playbackDataProvider.py";
my $sourceFile;
if ($filePath =~ m/dat$/) {
  $sourceFile = "cfg/fu_playbackDataProvider.py.streamerFile";
}
elsif ($filePath =~ m/root$/) {
  $sourceFile = "cfg/fu_playbackDataProvider.py.rootFile";
}
else {
  print STDERR "\n";
  print STDERR "Unsupport data file type.\n";
  print STDERR "Aborting...\n";
  exit;
}
my $workFile = "${targetFile}.modified";

open FILEIN, $sourceFile or die "Unable to open input file $sourceFile\n.";
open FILEOUT, ">$workFile" or die "Unable to open output file $workFile\n";

my $pathPrefix = "";
if (! ($filePath =~ m/^\//)) {
  $pathPrefix = "../../";
}
if (! ($filePath =~ m/^\/store\//)) {
  $pathPrefix = "file:" . $pathPrefix;
}
while (my $line = <FILEIN>) {
    chomp $line;

    # fill in the data file name, as appropriate
    # (adding leading "../../" if the path is relative)
    if ($line =~ m/^(.*)INPUT_FILE_SPEC_GOES_HERE(.*)$/) {
      $line = $1 . $pathPrefix . $filePath . $2;
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
}

close FILEIN;
close FILEOUT;

rename $targetFile, "${targetFile}.old";
rename $workFile, $targetFile;

#################################################
# Restart the system after getting confirmation #
#################################################

print STDOUT "\nRestart the system now (y/n [n])? ";
my $restartChoice = <STDIN>;
chomp $restartChoice;
$restartChoice = trim($restartChoice);
$restartChoice = lc($restartChoice);

if ($restartChoice =~ m/^y/) {
  print STDOUT "\n*** Restarting the system.\n";
  print STDOUT "*** Please allow 2-3 minutes for the restart to complete " .
    "if this is the\n";
  print STDOUT "*** first restart in the last 10 minutes. If this is the " .
    "second or third\n";
  print STDOUT "*** restart in a short time frame, the restart may take " .
    "10-15 minutes.\n";
  my $result = `killall -9 xdaq.exe`;
  printf STDOUT "$result";
}
else {
  print STDOUT "\n*** The system was *not* restarted.\n";
  print STDOUT "*** Your changes will not take effect until the next restart.\n";
  print STDOUT "*** You can use 'killall -9 xdaq.exe' to force a restart.\n";
}
