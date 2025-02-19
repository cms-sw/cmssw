package Kapslib;

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
#  $spare1
#  $spare2
#  $spare3
#  $spare4
#  $spare5
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
#  @JOBSP2 - spare
#  @JOBSP3 - spare

  use Exporter   ();
  use vars       qw( $VERSION @ISA @EXPORT @EXPORT_OK %EXPORT_TAGS);
  # set the version for version checking
  $VERSION     = 1.00;

  @ISA         = qw(Exporter);
  @EXPORT      = qw(
                    write_db
                    read_db 
                    print_memdb
		    get_class
		    @JOBID
		    $header 
		    $batchScript $cfgTemplate $infiList $class $addFiles $driver $nJobs
                    $mergeScript $mssDir $updateTime $updateTimeHuman $elapsedTime $spare1 $spare2 $spare3 $spare4 $spare5
		    @JOBDIR 
		    @JOBSTATUS @JOBNTRY @JOBRUNTIME @JOBNEVT @JOBHOST @JOBINCR @JOBREMARK @JOBSP1 @JOBSP2 @JOBSP3
                   );

sub write_db() {
  $header = "kaps database schema 3.0    R. Mankel 2-Aug-2007";
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
  $spare4 = "-- unused --";
  $spare5 = "-- unused --";

  open DBFILE,">kaps.db";
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
  printf DBFILE "%s\n",$spare1;
  printf DBFILE "%s\n",$spare2;
  printf DBFILE "%s\n",$spare3;
  printf DBFILE "%s\n",$spare4;
  printf DBFILE "%s\n",$spare5;
  my $i;
  for ($i = 0; $i < @JOBID; ++$i) {
    printf DBFILE "%03d:%s:%05d:%s:%d:%d:%d:%s:%d:%s:%s:%s:%s\n",
    $i+1,@JOBDIR[$i],@JOBID[$i],@JOBSTATUS[$i],@JOBNTRY[$i],@JOBRUNTIME[$i],@JOBNEVT[$i],@JOBHOST[$i],
    @JOBINCR[$i],@JOBREMARK[$i],@JOBSP1[$i],@JOBSP2[$i],@JOBSP3[$i];
  }
  close DBFILE;
}

sub read_db() {
  open DBFILE,"kaps.db";
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
  $spare1 = <DBFILE>;
  $spare2 = <DBFILE>;
  $spare3 = <DBFILE>;
  $spare4 = <DBFILE>;
  $spare5 = <DBFILE>;
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
  chomp $spare1;
  chomp $spare2;
  chomp $spare3;
  chomp $spare4;
  chomp $spare5;
  $nJobs = 0; 
  while (<DBFILE>) {
    chomp $_;
    my $line;
    ($line,@JOBDIR[$nJobs],@JOBID[$nJobs],@JOBSTATUS[$nJobs],
    @JOBNTRY[$nJobs],@JOBRUNTIME[$nJobs],@JOBNEVT[$nJobs],@JOBHOST[$nJobs],@JOBINCR[$nJobs],
    @JOBREMARK[$nJobs],@JOBSP1[$nJobs],@JOBSP2[$nJobs],@JOBSP3[$nJobs])
      = split(":",$_);
    ++$nJobs;
  }
  # check if last job is a merge job
  if (@JOBDIR[$nJobs-1] eq "jobm") {
      # reduce nJobs since we should not count merge job
      --$nJobs;
  }
  close DBFILE;
}

sub print_memdb() {
  print "=== kaps database printout ===\n";
  print "$header\n";
  printf "Script %s card %s infi %s class %s files %s driver %s mergeScript %s mssDir %s updateTime %s elapsed %d\n",$batchScript,
  $cfgTemplate,$infiList,$class,$addFiles,$driver,$mergeScript,$mssDir,$updateTimeHuman,$elapsedTime;
  printf "%3s %8s %7s %5s %4s %6s %9s %10s %10s \n",
  '###',"dir","jobid","stat","ntry","rtime","nevt","time/evt","remark";
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
    printf "%03d %8s %07d %5s %4d %6d %9d %10.3f %10s\n",
    $i+1,@JOBDIR[$i],@JOBID[$i],@JOBSTATUS[$i],@JOBNTRY[$i],
      $thisCpu,@JOBNEVT[$i],$cpuPerEvt,@JOBHOST[$i];
    if (@JOBNEVT[$i] > 0) {
      $totEvt = $totEvt + @JOBNEVT[$i];
    }
    $totCpu = $totCpu + $thisCpu;
  }
  
  # if merge mode, print merge job as well
  if ($driver eq "merge") {
      $i = $nJobs;
      $cpuFactor = get_cpufactor(@JOBHOST[$i]);
      $thisCpu = @JOBRUNTIME[$i] * $cpuFactor;
      printf "%3s %6s   %07d %5s %4d %6d %9d %10.3f %10s\n",
    "MMM",@JOBDIR[$i],@JOBID[$i],@JOBSTATUS[$i],@JOBNTRY[$i],
      $thisCpu,@JOBNEVT[$i],0,@JOBHOST[$i];
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

# returns job class as stored in db
sub get_class {
    return $class;
}

sub get_cpufactor() {
  $theFactor = 1;
}
