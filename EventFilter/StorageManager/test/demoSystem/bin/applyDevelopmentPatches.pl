#!/usr/bin/env perl
#
# 15-Nov-2007 -- 30-Dec-2008, KAB

use Getopt::Std;
use File::Basename;
use strict;

# parse any command-line switches
use vars qw($opt_b $opt_d $opt_e $opt_f $opt_h $opt_i $opt_p $opt_r $opt_s);
my $optionSuccess = getopts("bdefhiprs");

# if the option parsing failed or the user requested help or there are
# command-line arguments other than switches, print out a usage message
if (! $optionSuccess || $opt_h || @ARGV)
{
  print "Usage:  applyDevelopmentPatches [-h] [-b] [-d] [-e] [-f] [-i] [-p] [-r] [-s]\n";
  print " where  -h : display usage Help (this message)\n";
  print "        -b : patch the EventFilter/AutoBU/BU code\n";
  print "        -d : patch the IORawData/DaqSource code\n";
  print "        -e : patch the EventFilter/AutoBU/BUEvent code\n";
  print "        -f : patch the EventFilter/Processor code\n";
  print "        -i : patch the IOPool/Streamer code\n";
  print "        -p : patch the FWCore/Modules/Prescaler code\n";
  print "        -r : patch the EventFilter/ResourceBroker code\n";
  print "        -s : patch the EventFilter/StorageManager code\n";
  print " If no switches are specified, -b and -f are assumed.\n";
  exit;
}

my $doDefaults = (!$opt_b && !$opt_d && !$opt_e && !$opt_f && !$opt_i && !$opt_p && !$opt_r && !$opt_s);

# *****************************************************************
# Modify the AutoBU code to support a sleep between events.
# Also, reset the event number to zero at beginRun and
# drain events at endRun.
# *****************************************************************
if ($doDefaults || defined($opt_b)) {

  my $inputFile = "EventFilter/AutoBU/interface/BU.h";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $threeLineWindow = "";
  while (my $line = <FILEIN>) {
    chomp $line;

    # update the multi-line "window" with this new line
    if ($threeLineWindow =~ m/(.*)\n(.*)\n(.*)/) {
      $threeLineWindow = $2 . "\n" . $3;
    }
    $threeLineWindow .= "\n" . $line;

    # add attributes for readout throttling
    if ($threeLineWindow =~ m/memory\s+pool\s+for\s+i20\s+communication.+toolbox::mem::Pool.+i2oPool_/s &&
        $line =~ m/^\s*$/) {
      $line .= "\n    // readout throttling for playback\n" .
        "    xdata::UnsignedInteger32 throttlingSleepUs_;\n" .
        "    unsigned int sleepUsec_;\n";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

  # ***************************************************************

  my $inputFile = "EventFilter/AutoBU/src/BU.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $firstBlockFlag = 1;
  my $commentOutFlag = 0;
  my $threeLineWindow = "";
  while (my $line = <FILEIN>) {
    chomp $line;

    # update the multi-line "window" with this new line
    if ($threeLineWindow =~ m/(.*)\n(.*)\n(.*)/) {
      $threeLineWindow = $2 . "\n" . $3;
    }
    $threeLineWindow .= "\n" . $line;

    # initialize the sleep time attributes
    if ($line =~ m/(\s*),\s*i2oPool_\s*\(\s*0\s*\)/) {
      $line .= "\n" . $1 . ", throttlingSleepUs_(0)";
    }
    if ($threeLineWindow =~ m/(\s*)string\s+msg.+configuring\s+FAILED.*fsm_.fireFailed.*}/s) {
      $line .= "\n\n" . $1 . "sleepUsec_ = throttlingSleepUs_;";
    }
    if ($line =~ m/(\s*)gui_\s*-\>\s*addStandardParam\s*\(\s*\"foundRcmsStateListener\"/) {
      $line .= "\n" . $1 . "gui_->addStandardParam(\"throttlingSleepUs\", &throttlingSleepUs_);";
    }

    # add a sleep statement to the generateEvent method
    if ($threeLineWindow =~ m/\s*bool\s+BU::generateEvent\(BUEvent\* \w+\)\s+{\s*$/s) {
      $line .= "\n  if (sleepUsec_) usleep(sleepUsec_);";
    }

    # limit the number of FEDs created in Random mode
    if ($threeLineWindow =~ m/\n(\s*)unsigned\s+int\s+fedSizeMin\s*\=\s*fedHeaderSize\_\s*\+\s*fedTrailerSize\_\;\s+for\s*\(\s*unsigned\s+int\s+(\S+)\s*\=\s*0\s*\;\s*\2\s*\<\s*validFedIds\_\.size\(\)\s*\;\s*\2\+\+\s*\)\s*\{\s*$/s) {
      $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "unsigned int limitedFedIds = validFedIds_.size();" .
        "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "if (limitedFedIds > 20) {limitedFedIds = 20;}" .
        "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "for (unsigned int $2=0;$2<limitedFedIds;++$2) {";
      #print STDOUT "$threeLineWindow\n";
    }

    # add an event number reset to the enabling() method
    if ($threeLineWindow =~ m/bool\s+BU::enabling\(.+\)\s*\{\s*$/s) {
      $line .= "\n  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT " .
        "SYSTEM\n  evtNumber_ = 0;";
    }

    # comment out code to allow events to drain at endRun
    # [the order of these "if" statements is VERY important]
    if ($commentOutFlag && $line =~ m/\{/ && ! ($line =~ m/replay\_\.value\_.*nbEventsBuilt\_/)) {
      $commentOutFlag += 1;
    }
    if (! $commentOutFlag &&
        $line =~ m/^(\s*)(if\s*\(\s*0\s*\!\=\s*PlaybackRawDataProvider::instance\(\)\s*&&\s*)/) {
      $commentOutFlag += 1;
    }
    if ($commentOutFlag) {
      $line = "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM\n//" . $line;
    }
    if ($commentOutFlag && $line =~ m/\}/) {
      $commentOutFlag -= 1;
      if ($commentOutFlag == 0 && $firstBlockFlag) {
        $firstBlockFlag = 0;
        $line .= "\n    lock();";
        $line .= "\n    freeIds_.push(events_.size());";
        $line .= "\n    unlock();";
        $line .= "\n    postBuild();";
        $line .= "\n    unsigned int oldValue = 0;";
        $line .= "\n    while (!builtIds_.empty() || oldValue != nbEventsBuilt_) {";
        $line .= "\n      oldValue = nbEventsBuilt_;";
        $line .= "\n      LOG4CPLUS_INFO(log_,\"wait to flush ... #builtIds=\"<<builtIds_.size());";
        $line .= "\n      ::sleep(1); // nbEventsBuilt_ may change during this sleep...";
        $line .= "\n    }";
      }
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# *****************************************************************
# Modify the AutoBU code to provide debug information
# on buffer sizes.
# *****************************************************************
if (defined($opt_e)) {

  my $inputFile = "EventFilter/AutoBU/src/BUEvent.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  while (my $line = <FILEIN>) {
    chomp $line;

    # add the definition of the maximum observed event size
    if ($line =~ m/bool\s+BUEvent::computeCrc_\s*=\s*true;/) {
      $line .= "\n// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM";
      $line .= "\nunsigned int BUEvent::maxEvtSize_=0;";
    }

    # add the buffer size to the buffer overflow error message
    if ($line =~ m/(\s*)cout\<\<\"BUEvent::writeFed\(\) ERROR: buffer overflow\.\"\<\<endl\;/) {
      $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "cout<<\"BUEvent::writeFed() ERROR: buffer overflow. (buffer size = \"<<bufferSize_<<\")\"<<endl;";
    }

    # print out a message when the maximum observed buffer size changes
    if ($line =~ m/^(\s*)evtSize\_\s*\+\=\s*size\;/) {
      $line .= "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM";
      $line .= "\n" . $1 . "if (evtSize_ > maxEvtSize_) {";
      $line .= "\n" . $1 . "  maxEvtSize_ = evtSize_;";
      $line .= "\n" . $1 . "  std::cout << \"Maximum observed event size = \" << maxEvtSize_";
      $line .= "\n" . $1 . "            << \" (buffer size = \" << bufferSize_ << \")\" << std::endl;";
      $line .= "\n" . $1 . "}";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

  # ***************************************************************

  my $inputFile = "EventFilter/AutoBU/interface/BUEvent.h";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  while (my $line = <FILEIN>) {
    chomp $line;

    # add the declaration of the maximum observed event size
    if ($line =~ m/(\s*)static\s+bool\s+computeCrc\_\;/) {
      $line .= "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM";
      $line .= "\n" . $1 . "static unsigned int maxEvtSize_;";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# ************************************************************
# Skip over the checking of file paths in the storage manager.
# ************************************************************
if (defined($opt_s)) {

  my $inputFile = "EventFilter/StorageManager/src/StorageManager.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $runNumberChangeDone = 0;
  my $threeLineWindow = "";
  while (my $line = <FILEIN>) {
    chomp $line;

    # update the multi-line "window" with this new line
    if ($threeLineWindow =~ m/(.*)\n(.*)\n(.*)/) {
      $threeLineWindow = $2 . "\n" . $3;
    }
    $threeLineWindow .= "\n" . $line;
    #print STDOUT "========================================\n";
    #print STDOUT "$threeLineWindow\n";

    # comment out calls to checkDirectoryOK since we will not be writing
    # out data anyway and this allows us to use env vars in the cfg file.
    if ($line =~ m/(\s*)checkDirectoryOK\s*\(\s*\w+\_\.toString\s*\(\s*\)\s*\);/) {
      $line = "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT " .
        "SYSTEM\n//" . $line;
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# ***************************************************************************
# Add message dumps to the EventStreamHttpReader class.
# ***************************************************************************
if (defined($opt_s)) {

  my $inputFile = "EventFilter/StorageManager/src/EventStreamHttpReader.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $includeDone = 0;

  while (my $line = <FILEIN>) {
    chomp $line;

    # add the DumpTools header file
    if (! $includeDone && $line =~ m/^(\s*)\#include /) {
      $includeDone = 1;
      $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "#include \"IOPool/Streamer/interface/DumpTools.h\"\n" . $line;
    }

    # add the INIT message dump
    if ($line =~ m/^(\s*)InitMsgView\s+(\S+)\(/) {
      $line .= "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "//dumpInitVerbose(&" . $2 . ");";
    }

    # add the Event message dump
    if ($line =~ m/^(\s*)EventMsgView\s+(\S+)\(/) {
      $line .= "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "//dumpEventView(&" . $2 . ");";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# ***************************************************************************
# Add message dumps to the StreamerFileReader class.
# ***************************************************************************
if (defined($opt_i)) {

  my $inputFile = "IOPool/Streamer/src/StreamerFileReader.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $includeDone = 0;

  while (my $line = <FILEIN>) {
    chomp $line;

    # add the DumpTools header file
    if (! $includeDone && $line =~ m/^(\s*)\#include /) {
      $includeDone = 1;
      $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "#include \"IOPool/Streamer/interface/DumpTools.h\"\n" . $line;
    }

    # add the INIT message dump
    if ($line =~ m/^(\s*)const\s+InitMsgView\s*\*\s*(\S+)\s*\=/) {
      $line .= "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "dumpInitVerbose(" . $2 . ");";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# ***************************************************************************
# Add message dumps to the StreamerInputFile class.
# ***************************************************************************
if (defined($opt_i)) {

  my $inputFile = "IOPool/Streamer/src/StreamerInputFile.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $includeDone = 0;

  while (my $line = <FILEIN>) {
    chomp $line;

    # add the DumpTools header file
    if (! $includeDone && $line =~ m/^(\s*)\#include /) {
      $includeDone = 1;
      $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "#include \"IOPool/Streamer/interface/DumpTools.h\"\n" . $line;
    }

    # add the Event message dump
    if ($line =~ m/^(\s*)(\S+)\s*\=\s*new\s+EventMsgView/) {
      $line .= "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "dumpEventView(" . $2 . ");";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# ********************************************************
# Randomize the prescaling in the Prescaler module a bit.
# ********************************************************
if (defined($opt_p)) {

  my $inputFile = "FWCore/Modules/src/Prescaler.h";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  while (my $line = <FILEIN>) {
    chomp $line;

    # add the boost random header file
    if ($line =~ m/\#include\s+\"FWCore\/Framework\/interface\/EDFilter\.h\"/) {
      $line .= "\n" . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . "#include <boost/random.hpp>";
    }

    # add the additional class attributes
    if ($line =~ m/(\s*)int\s+n\_\;\s*\/\/\s*accept\s*one\s*in\s*n/) {
      $line =  $1 .  "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "int acceptedCount_;" .
        "\n" . $line .
        "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "double lowBound_;" .
        "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "double highBound_;" .
        "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "boost::mt19937 baseGenerator_;" .
        "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "boost::shared_ptr< boost::uniform_int<int> > distribution_;" .
        "\n" . $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . $1 . "boost::shared_ptr< boost::variate_generator<boost::mt19937, boost::uniform_int<> > > generator_;";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

  # ***************************************************************

  my $inputFile = "FWCore/Modules/src/Prescaler.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $threeLineWindow = "";
  while (my $line = <FILEIN>) {
    chomp $line;

    # update the multi-line "window" with this new line
    if ($threeLineWindow =~ m/(.*)\n(.*)\n(.*)/) {
      $threeLineWindow = $2 . "\n" . $3;
    }
    $threeLineWindow .= "\n" . $line;
    #print STDOUT "========================================\n";
    #print STDOUT "$threeLineWindow\n";

    # add header files
    if ($line =~ m/\#include \"FWCore\/ParameterSet\/interface\/ParameterSet\.h\"/) {
      $line .= "\n" . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n" . "#include <iostream>";
    }

    if ($line =~ m/(\s*)count_\s*\(\s*\)\s*\,/) {
      $line .= "\n" . $1 .
        "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
          "\n" . $1 . "acceptedCount_(0),";
    }

    if ($threeLineWindow =~ m/n\_\(ps\.getParameter\<int\>\(\"prescaleFactor\"\)\)\s+\{(\s+)\}/s) {
      $line = "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  lowBound_ = 0.97 * (1.0 / (double) n_);" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  highBound_ = 1.03 * (1.0 / (double) n_);" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  distribution_.reset(new boost::uniform_int<int>(1,n_));" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  generator_.reset(new boost::variate_generator<boost::mt19937, boost::uniform_int<> >(baseGenerator_, *distribution_));" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  unsigned int seedValue = 0xffff & (int) this;" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  baseGenerator_.seed(static_cast<unsigned int>(seedValue));" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  int randNum = -1;" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  int dummyCalls = 0xff & (int) this;" .
        "$1  // TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "$1  for (int idx = 0; idx < dummyCalls; idx++) {" .
        "$1    randNum = (*generator_)();" .
        "$1  }" .
        "$1}";
    }

    if ($line =~ m/(\s*)return\s+count\_\%n\_\s*\=\=0\s*\?\s*true\s*:\s*false\;/) {
      $line = "$1// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n$1int randNum = (*generator_)();" .
        "\n$1// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n$1bool accept = (randNum == 1);" .
        "\n$1// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n$1double currentRatio = ((double) acceptedCount_) / ((double) count_);" .
        "\n$1// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n$1if (currentRatio < lowBound_) {" .
        "\n$1  accept = true;;" .
        "\n$1}" .
        "\n$1// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n$1else if (currentRatio > highBound_) {" .
        "\n$1  accept = false;" .
        "\n$1}" .
        "\n$1// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n$1if (accept) {" .
        "\n$1  ++acceptedCount_;" .
        "\n$1}" .
        "\n$1// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
        "\n$1return accept;";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# *****************************************************************
# Modify the ResourceBroker code to fix the race condition in the
# tests for duplication discard messages from the StorageManager.
# *****************************************************************
if (defined($opt_r)) {

  my $inputFile = "EventFilter/ResourceBroker/src/FUResourceTable.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $foundSpuriousDiscardFix = 0;
  while (my $line = <FILEIN>) {
    chomp $line;

    # only modify the code if it looks like the protection
    # for the spurious SM discards is in this version
    if ($line =~ m/acceptSMDataDiscard_/) {
      $foundSpuriousDiscardFix = 1;
    }

    if ($foundSpuriousDiscardFix) {

      # fix the sending of the INIT message
      if ($line =~ m/^(\s*).*sm\_\-\>sendInitMessage\(/) {
        $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
          "\n" . $1 . "acceptSMDataDiscard_[fuResourceId] = true;\n" . $line;
      }

      # fix the sending of the Event message
      if ($line =~ m/^(\s*).*sm\_\-\>sendDataEvent\(/) {
        $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
          "\n" . $1 . "acceptSMDataDiscard_[fuResourceId] = true;\n" . $line;
      }

      # fix the sending of the Error Event message
      if ($line =~ m/^(\s*).*sm\_\-\>sendErrorEvent\(/) {
        $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
          "\n" . $1 . "acceptSMDataDiscard_[fuResourceId] = true;\n" . $line;
      }

      # fix the sending of the DQM Event message
      if ($line =~ m/^(\s*).*sm\_\-\>sendDqmEvent\(/) {
        $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
          "\n" . $1 . "acceptSMDqmDiscard_[fuDqmId] = true;\n" . $line;
      }

      # clean up the late accept assignment for INIT, Event, and Error Event msgs
      if ($line =~ m/^(\s*).*acceptSMDataDiscard\_\[fuResourceId\]\s*\=\s*true/) {
        $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
          "\n" . $1 . "//" . $line;
      }

      # clean up the late accept assignment for DQM Event msgs
      if ($line =~ m/^(\s*).*acceptSMDqmDiscard\_\[fuDqmId\]\s*\=\s*true/) {
        $line = $1 . "// TEMPORARY HACK FOR A STORAGE MANAGER DEVELOPMENT SYSTEM" .
          "\n" . $1 . "//" . $line;
      }

    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# ************************************************************
# Modify the FUEventProcessor to call the asynchronous
# stop method of the EventProcessor so that we can gracefully
# shut down FU event consumer processes.
# Also, protect against a null prescale service so that we
# can use the SpotLight web page.
# Also, add support for skipping attaches to the DQM shared
# memory.
# Also, add support for calling "declareRunNumber" when needed.
# ************************************************************
if ($doDefaults || defined($opt_f)) {

  my $inputFile;
  if (-e "EventFilter/Processor/interface" &&
      -e "EventFilter/Processor/interface/FUEventProcessor.h") {
    $inputFile = "EventFilter/Processor/interface/FUEventProcessor.h";
  }
  else {
    $inputFile = "EventFilter/Processor/src/FUEventProcessor.h";
  }
  my $outputFile = "${inputFile}.modified";
  print STDOUT "$inputFile $outputFile\n";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $declaringClass = 0;
  while (my $line = <FILEIN>) {
    chomp $line;

    # add the configuration parameters that will be used to control
    # the new behavior
    if ($declaringClass && $line =~ m/(\s*)};/) {
      $declaringClass = 0;
      $line = $1 . "  xdata::Boolean                   ignoreDQMAttachErrors_;\n" .
        $1 . "  xdata::Boolean                   declareRunNumberIfInvalid_;\n" .
        $1 . "  xdata::Boolean                   useAsynchronousStop_;\n" .
        $line;
    }
    if ($line =~ m/class\s+FUEventProcessor/) {
      $declaringClass = 1;
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

  # ***************************************************************

  my $inputFile = "EventFilter/Processor/src/FUEventProcessor.cc";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  my $declaringConstructor = 0;
  my $insideSpotlightCode = 0;
  my $fourLineWindow = "";
  while (my $line = <FILEIN>) {
    chomp $line;

    # update the multi-line "window" with this new line
    if ($fourLineWindow =~ m/(.*)\n(.*)\n(.*)\n(.*)/) {
      $fourLineWindow = $2 . "\n" . $3 . "\n" . $4;
    }
    $fourLineWindow .= "\n" . $line;

    # add initialization to the constructor
    if ($declaringConstructor && $line =~ m/\{/) {
      $declaringConstructor = 0;
      $line = "  , ignoreDQMAttachErrors_(false)\n" .
        "  , declareRunNumberIfInvalid_(false)\n" .
        "  , useAsynchronousStop_(false)\n" .
        $line;
    }
    if ($line =~ m/FUEventProcessor::FUEventProcessor.+ApplicationStub/) {
      $declaringConstructor = 1;
    }

    # attach the cfg params to the infospace
    if ($line =~ m/(\s*)ispace\s*-\>\s*fireItemAvailable\s*\(\s*\"foundRcmsStateListener\"/) {
      $line .= "\n" . $1 . "ispace->fireItemAvailable(\"ignoreDQMAttachErrors\",&ignoreDQMAttachErrors_);" .
        "\n" . $1 . "ispace->fireItemAvailable(\"declareRunNumberIfInvalid\",&declareRunNumberIfInvalid_);" .
        "\n" . $1 . "ispace->fireItemAvailable(\"useAsynchronousStop\",&useAsynchronousStop_);";
    }

    # ignore DQM shared memory attach errors
    if ($fourLineWindow =~ m/FUEventProcessor.+Dqm.+Shm.+string\s+errmsg/s &&
        $line =~ m/(\s*)bool\s+success\s*=\s*false/) {
      $line .= "\n" . $1 . "if (ignoreDQMAttachErrors_) success = true;"
    }

    # modify the run number setter method that is used
    if ($fourLineWindow =~ m/evtProcessor_\s*-\>\s*clearCounters.+if\s*\(\s*isRunNumberSetter/s &&
        $line =~ m/(\s*)evtProcessor_\s*-\>\s*setRunNumber/) {
      $line = $1 . "if (runNumber_.value_ > 0 || !declareRunNumberIfInvalid_) {\n" .
        "  " . $line . "\n" .
        $1 . "}\n" .
        $1 . "else {\n" .
        $1 . "  evtProcessor_->declareRunNumber(runNumber_.value_);\n" .
        $1 . "}\n";
    }

    # modify the stop call that is used
    if ($line =~ m/(\s*).*evtProcessor\_\-\>waitTillDoneAsync\(/) {
      $line = $1 . "if (!useAsynchronousStop_) {\n" .
        "  " . $line . "\n" .
        $1 . "}\n" .
        $1 . "else {\n" .
        $1 . "  rc = evtProcessor_->stopAsync(timeoutOnStop_.value_);\n" .
        $1 . "}\n";
    }

    # protect against a null prescale service
    if ($insideSpotlightCode &&
        $line =~ m/(\s*)(if\s*\(\s*psid\s*\!\=\s*0)(\s*\).*)/) {
      $line = $1 . $2 . " && prescaleSvc_ != 0" . $3;
    }
    if ($insideSpotlightCode &&
        $line =~ m/\S+\s+FUEventProcessor::/) {
      $insideSpotlightCode = 0;
    }
    if ($line =~ m/void\s+FUEventProcessor::spotlightWebPage/) {
      $insideSpotlightCode = 1;
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}

# ***********************************************************************
# Force lumi block numbers created in the DaqSource to increase smoothly.
# ***********************************************************************
if (defined($opt_d)) {

  my $inputFile = "IORawData/DaqSource/plugins/DaqSource.h";
  my $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  while (my $line = <FILEIN>) {
    chomp $line;

    # add attributes for smoothly increasing the lumi block number
    if ($line =~ m/(\s*)bool\s+fakeLSid\_\;/) {
      $line .= "\n" . $1 . "unsigned long   fakeEvId_;" .
        "\n" . $1 . "bool            calculateFakeLumiSection_;" .
        "\n" . $1 . "bool            firstEvent_;";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  my $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

  # ***************************************************************

  $inputFile = "IORawData/DaqSource/plugins/DaqSource.cc";
  $outputFile = "${inputFile}.modified";

  open FILEIN, $inputFile or die "Unable to open input file $inputFile\n.";
  open FILEOUT, ">$outputFile" or die "Unable to open output file $outputFile\n";

  while (my $line = <FILEIN>) {
    chomp $line;

    # initialize the configuration parameter(s)
    if ($line =~ m/(\s*),\s*fakeLSid_\s*\(\s*lumiSegmentSizeInEvents_/) {
      $line .= "\n" . $1 .
        ", calculateFakeLumiSection_(pset.getUntrackedParameter<bool>(\"calculateFakeLumiSection\",false))";
    }

    # initialize the first event number, etc.
    if ($line =~ m/(\s*)produces\<FEDRawDataCollection\>\(\)\;/) {
      $line = "\n" . $1 . "firstEvent_ = true;" .
        "\n" . $1 . "fakeEvId_ = 0;\n\n" . $line;
    }

    # modify the fake lumi block calculation
    if ($line =~ m/(\s*)if\(fakeLSid\_ \&\& luminosityBlockNumber\_ \!\= \(\(eventId\.event\(\)\s*\-\s*1\)\/lumiSegmentSizeInEvents\_ \+ 1\)\) \{/) {
      my $indent = $1;

      # read in the existing code block
      my @existingCodeLines;
      while (my $tmpLine = <FILEIN>) {
        chomp $tmpLine;
        if ($tmpLine =~ m/\}/) {last;}
        $tmpLine =~ s/\t/        /g;
        push @existingCodeLines, $tmpLine;
      }

      # handle the non-calculateFakeLumiSection case
      my $modifiedIf = $line;
      $modifiedIf =~ s/fakeLSid_\s*\&\&\s*//;
      my $part2 = "    " . $modifiedIf;
      for my $text (@existingCodeLines) {
        $part2 .= "\n  " . $text;
      }

      # handle the calculateFakeLumiSection case
      my $part1 = $part2;
      $part1 =~ s/\!\=/\</;
      $part1 =~ s/eventId.event\(\)/fakeEvId_/g;
      $part1 =~ s/\)\s*\{/ || firstEvent_) {\n${indent}      firstEvent_ = false;/;

      # put it all together
      $line =~ s/(.*fakeLSid_).*(\)\s*\{\s*)/$1$2/;
      $line .= "\n" . $indent . "  if(calculateFakeLumiSection_) {";
      $line .= "\n" . $indent . "    ++fakeEvId_;";
      $line .= "\n" . $part1;
      $line .= "\n" . $indent . "    }";
      $line .= "\n" . $indent . "  }";
      $line .= "\n" . $indent . "  else {";
      $line .= "\n" . $part2;
      $line .= "\n" . $indent . "    }";
      $line .= "\n" . $indent . "  }";
      $line .= "\n" . $indent . "}";
    }

    # write the input line to the output file
    print FILEOUT "$line\n";
  }

  close FILEIN;
  close FILEOUT;

  rename $inputFile, "${inputFile}.orig";
  rename $outputFile, $inputFile;

  print STDOUT "\n";
  print STDOUT "============================================================\n";
  print STDOUT " Modification made to ${inputFile}:\n";
  print STDOUT "============================================================\n";
  $result=`diff $inputFile ${inputFile}.orig`;
  print STDOUT "$result";

}
