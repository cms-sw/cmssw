package Mpslib;  # assumes Some/Module.pm

# $Revision: 1.9 $ by $Author: jbehr $
#
# Meaning of the database variables:
#
# (1) Header
# 
#  $header  - version information
#  $batchScript - base script for serial job
#  $cfgTemplate - template for cfg file
#  $infiList  -  list of input files to be serialized
#  $class - batch class information (might contain two ':'-separated)
#  $addFiles - job name for submission
#  $driver - specifies whether merge job is foreseen
#  $nJobs - number of serial jobs (not including merge job)
#  $mergeScript - base script for merge job
#  $mssDir - directory for mass storage (e.g. Castor)
#  $updateTime - time of last update (seconds since 1970)
#  $updateTimeHuman - time of last update (human readable)
#  $elapsedTime - seconds since last update
#  $mssDirPool - pool for $mssDir (e.g. cmscaf/cmscafuser)
#  $pedeMem - Memory allocated for pede
#  $spare1
#  $spare2
#  $spare3
#
# (2) Job-level variables
#  
#  @JOBDIR  - name of job directory (not full path)
#  @JOBSTATUS - status of job
#  @JOBRUNTIME -  present CPU time of job
#  @JOBNEVT - number of events processed by job
#  @JOBHOST - presently used to store remark
#  @JOBINCR - CPU increment since last check
#  @JOBREMARK - comment
#  @JOBSP1 - spare
#  @JOBSP2 - possible weight for pede
#  @JOBSP3 - possible name as given to mps_setup.pl -N <name> ...

  
  use Exporter   ();
  use vars       qw( $VERSION @ISA @EXPORT @EXPORT_OK %EXPORT_TAGS);
#  use DBI;
  # set the version for version checking
  $VERSION     = 1.01;

  @ISA         = qw(Exporter);
  @EXPORT      = qw(
                    write_db
                    read_db 
                    print_memdb
		    get_class
		    @JOBID
		    $header 
		    $batchScript $cfgTemplate $infiList $class $addFiles $driver $nJobs
                    $mergeScript $mssDir $updateTime $updateTimeHuman $elapsedTime $mssDirPool $pedeMem $spare1 $spare2 $spare3
		    @JOBDIR 
		    @JOBSTATUS @JOBNTRY @JOBRUNTIME @JOBNEVT @JOBHOST @JOBINCR @JOBREMARK @JOBSP1 @JOBSP2 @JOBSP3
                   );

our $eos = "/afs/cern.ch/project/eos/installation/cms/bin/eos.select";

sub write_db() {
  $header = "mps database schema 3.2" ;
  $currentTime = `date +%s`;
  chomp $currentTime;
  $elapsedTime = 0;
  if ($updateTime != 0) { $elapsedTime = $currentTime - $updateTime; }
  $updateTime = $currentTime;
  $updateTimeHuman = `date`;
  chomp $updateTimeHuman;
  $spare1 = "-- unused --";
  $spare2 = "-- unused --";
  $spare3 = "-- unused --";

  system "[[ -a mps.db ]] && cp -p mps.db mps.db~"; # GF: backup if exists (in case of interupt during write)
  open DBFILE,">mps.db";
  printf DBFILE "%s\n",$header;
  printf DBFILE "%s\n",$batchScript;
  printf DBFILE "%s\n",$cfgTemplate;
  printf DBFILE "%s\n",$infiList;
  printf DBFILE "%s\n",$class;
  printf DBFILE "%s\n",$addFiles;
  printf DBFILE "%s\n",$driver;
  printf DBFILE "%s\n",$mergeScript;
  printf DBFILE "%s\n",$mssDir;
  printf DBFILE "%d\n",$updateTime;
  printf DBFILE "%s\n",$updateTimeHuman;
  printf DBFILE "%d\n",$elapsedTime;
  printf DBFILE "%s\n",$mssDirPool;
  printf DBFILE "%d\n",$pedeMem;
  printf DBFILE "%s\n",$spare1;
  printf DBFILE "%s\n",$spare2;
  printf DBFILE "%s\n",$spare3;
  my $i;
  for ($i = 0; $i < @JOBID; ++$i) {
    printf DBFILE "%03d:%s:%05d:%s:%d:%d:%d:%s:%d:%s:%s:%s:%s\n",
    $i+1,@JOBDIR[$i],@JOBID[$i],@JOBSTATUS[$i],@JOBNTRY[$i],@JOBRUNTIME[$i],@JOBNEVT[$i],@JOBHOST[$i],
    @JOBINCR[$i],@JOBREMARK[$i],@JOBSP1[$i],@JOBSP2[$i],@JOBSP3[$i];
  }
  close DBFILE;
}

sub read_db() {
  open DBFILE,"mps.db";
  @JOBDIR = ();
  @JOBID= ();
  @JOBSTATUS = ();
  @JOBNTRY = ();
  @JOBRUNTIME = ();
  @JOBNEVT = ();
  @JOBHOST = ();
  @JOBINCR = ();
  @JOBREMARK = ();
  @JOBSP1 = ();
  @JOBSP2 = ();
  @JOBSP3 = ();

  $header = <DBFILE>;
  $batchScript = <DBFILE>;
  $cfgTemplate = <DBFILE>;
  $infiList = <DBFILE>;
  $class = <DBFILE>;
  $addFiles = <DBFILE>;
  $driver = <DBFILE>;
  $mergeScript = <DBFILE>;
  $mssDir = <DBFILE>;
  $updateTime = <DBFILE>;
  $updateTimeHuman = <DBFILE>;
  $elapsedTime = <DBFILE>;
  $mssDirPool = <DBFILE>;
  $pedeMem = <DBFILE>;
  $spare1 = <DBFILE>;
  $spare2 = <DBFILE>;
  $spare3 = <DBFILE>;
  chomp $header;
  chomp $batchScript;
  chomp $cfgTemplate;
  chomp $infiList;
  chomp $class;
  chomp $addFiles;
  chomp $driver;
  chomp $mergeScript;
  chomp $mssDir;
  chomp $updateTime;
  chomp $updateTimeHuman;
  chomp $elapsedTime;
  chomp $mssDirPool;
  chomp $pedeMem;
  chomp $spare1;
  chomp $spare2;
  chomp $spare3;

  my $nMilleJobs = 0;
  $nJobs = 0; 
  while (<DBFILE>) { # loop through all jobs to read
    #chomp $_;
    my $line;
    my $nsplit = ($line,@JOBDIR[$nJobs],@JOBID[$nJobs],@JOBSTATUS[$nJobs],
    @JOBNTRY[$nJobs],@JOBRUNTIME[$nJobs],@JOBNEVT[$nJobs],@JOBHOST[$nJobs],@JOBINCR[$nJobs],
    @JOBREMARK[$nJobs],@JOBSP1[$nJobs],@JOBSP2[$nJobs],@JOBSP3[$nJobs])
      = split(":",$_);
    chomp $JOBSP3[$nJobs];
    unless (@JOBDIR[$nJobs] =~ m/jobm/) { # count mille jobs
	++$nMilleJobs;
    }
    ++$nJobs;
  }
  $nJobs=$nMilleJobs;
  close DBFILE;
}

sub print_memdb() {
  print "=== mps database printout ===\n";
  print "$header\n";
  printf "Script: %s\ncfg: %s\nfiles: %s\nclass: %s\nname: %s\ndriver: %s\nmergeScript: %s\nmssDir: %s\nupdateTime: %s\nelapsed: %d\nmssDirPool: %s\npedeMem: %d\n",$batchScript,
  $cfgTemplate,$infiList,$class,$addFiles,$driver,$mergeScript,$mssDir,$updateTimeHuman,$elapsedTime,$mssDirPool,$pedeMem;
  printf "%3s %6s %9s %6s %3s %5s %8s %6s %8s %s\n",
  '###',"dir","jobid","stat","try","rtime","nevt","t/evt","remark","name";
  my $i;
  my $totEvt = 0;
  my $totCpu = 0;
  for ($i=0; $i<$nJobs; ++$i) {
    $cpuFactor = get_cpufactor(@JOBHOST[$i]);
    $thisCpu = @JOBRUNTIME[$i] * $cpuFactor;
    my $cpuPerEvt = 0;
    if (@JOBRUNTIME[$i]>0 and @JOBNEVT[$i]>0) {
      $cpuPerEvt = $thisCpu / @JOBNEVT[$i];
    }
    printf "%03d %6s %09d %6s %3d %5d %8d %6.3f %8s %s\n",
    $i+1,@JOBDIR[$i],@JOBID[$i],@JOBSTATUS[$i],@JOBNTRY[$i],
      $thisCpu,@JOBNEVT[$i],$cpuPerEvt,@JOBHOST[$i],$JOBSP3[$i];
    if (@JOBNEVT[$i] > 0) {
      $totEvt = $totEvt + @JOBNEVT[$i];
    }
    $totCpu = $totCpu + $thisCpu;
  }
  
  # if merge mode, print merge job(s) as well
  if ($driver eq "merge") {
      $i = $nJobs;
      while ($i < @JOBID) {
        $cpuFactor = get_cpufactor(@JOBHOST[$i]);
        $thisCpu = @JOBRUNTIME[$i] * $cpuFactor;
        printf "%3s %6s %09d %6s %3d %5d %8d %6.3f %8s\n",
        "MMM",@JOBDIR[$i],@JOBID[$i],@JOBSTATUS[$i],@JOBNTRY[$i],
        $thisCpu,@JOBNEVT[$i],0,@JOBHOST[$i];
	++$i;
      }
  }

  my $meanCpuPerEvt = 0;
  if ($totEvt>0) {
    $meanCpuPerEvt = $totCpu / $totEvt;
  }
  print "----------------------------------------------\n";
  printf "                       Event total: %10d\n",$totEvt;
  printf "                         CPU total: %10.1f s\n",$totCpu;
  printf "                    Mean CPU/event: %10.3f s\n",$meanCpuPerEvt;
}

sub get_class {
    # returns job class as stored in db
    # one and only argument may be "mille" or "pede" for mille or pede jobs
    my @CLASSES = split ":",$class;
    if (@CLASSES < 1 || @CLASSES > 2) {
	print "\nget_class():\n  class must be of the form 'class' or 'classMille:classPede', but is '$class'!\n\n";
	exit 1;
	return "";
    } elsif ($_[0] eq "mille") {
	return $CLASSES[0];
    } elsif ($_[0] eq "pede") {
	if (@CLASSES == 1) {
	    return $CLASSES[0];
	} elsif (@CLASSES == 2) {
	    return $CLASSES[1];
	}
    } else {
	print "\nget_class():\n  Know class only for 'mille' or 'pede', not $_[0]!\n\n";
	exit 1;
	return "";
    }
}

sub get_cpufactor() {
  # $hostName = $_[0];
  # $znumber = 0;
  # my $cf;
  # if ( ($hostName =~ m/zenith(\d+)/) == 1) {
  #   $znumber = $1;
  # }
  # if ($znumber >=40 && $znumber < 100) {
  #   $cf = 1.;
  # }
  # elsif ($znumber >= 100 && $znumber < 155) {
  #   $cf = 3.06/2.2;
  # }
  # elsif ($znumber >= 155) {
  #   $cf = 1.25*3.06/2.2;
  # }
  # else {
  #   $cf = 1.;
  # }
  # $theFactor = $cf;
  $theFactor = 1;
}
