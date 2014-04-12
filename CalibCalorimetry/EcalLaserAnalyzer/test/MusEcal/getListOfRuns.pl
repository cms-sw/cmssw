#!/usr/bin/env perl

# get list of runs and gather header information 
# Julie Macles, Gautier Hamel de Monchenault

#use Date::Manip;

$datadir  = @ARGV[0];
$dir      = @ARGV[1];
$firstRun = @ARGV[2];
$lastRun  = @ARGV[3];

$laserdir       = "${datadir}/Laser/Analyzed";
$testpulsedir   = "${datadir}/TestPulse/Analyzed";
$runsdir        = "${datadir}/Runs";
$listdir        = "$dir";

if( $firstRun eq "" )
{
    $firstRun = "0"; $lastRun = "100000000";
}
elsif( $lastRun eq "+" )
{
    $lastRun = "100000000";
}
elsif( $lastRun eq "" )
{
    $lastRun = $firstRun;
}


open( LREDLIST,  ">${listdir}/runlist_Red_Laser")    || die "cannot open output file\n";
open( LBLUELIST, ">${listdir}/runlist_Blue_Laser")   || die "cannot open output file\n";
open( TPLIST,    ">${listdir}/runlist_Test_Pulse")   || die "cannot open output file\n";

$firstLaser = 1;
$firstTP    = 1;
$firstTS = 0;

opendir( RUNSDIR, $laserdir) || die "cannot open directory $laserdir\n";
@runsdir = sort  readdir( RUNSDIR );
foreach my $rundir (@runsdir)
{
    next unless( $rundir =~  /Run(\d*)_LB(.*)/ );
    $run = $1;
    $lumiblock = $2;

    my $curtype = "LASER";
    next if( $run < $firstRun || $run > $lastRun );
    next if( !open( HEADERFILE, "${laserdir}/${rundir}/header.txt") );

    my $timestampbeg = 0;
    my $timestampend = 0;
    my $mgpagain = -1;
    my $memgain  = -1;

    my $blueevents = 0;
    my $bluepower  = -10;
    my $bluefilter = -10;
    my $bluedelay  = -10;

    my $redevents = 0;
    my $redpower  = -10;
    my $redfilter = -10;
    my $reddelay  = -10;

    my $curcolor  = "NOCOLOR";

    while ( <HEADERFILE> )
    {
        chomp($_);
        $theLine = $_;

        if( $theLine =~ /RUN = (.*)/ )
        {
            if ( $run != $1 ) {
                print "Run number not properly filled: $run versus $1 \n";
            }
        }
        if( $theLine =~ /(.*) Events/ )
        {
            $curtype = $1;
            ($curtype =~ /LASER/) || last;
        }

        if( $theLine =~ /RUNTYPE = (\d*)/ )
        {
            $runtype = $1;
            ( $runtype == 4 || $runtype == 5 || $runtype == 6 || $runtype == 16 )  ||  last;
        }
        if( $theLine =~ /TIMESTAMP_BEG = (.*)/ ){ $timestampbeg = $1; }
        if( $theLine =~ /TIMESTAMP_END = (.*)/ ){ $timestampend = $1; }
        if( $theLine =~ /MPGA_GAIN = (.*)/ ){ $mgpagain = $1; }
        if( $theLine =~ /MEM_GAIN  = (.*)/ ){ $memgain = $1; }
        if( $theLine =~ /blue laser/ ){ $curcolor = "BLUE"; }
        if( $theLine =~ /red laser/ ){  $curcolor = "RED"; }

        if( $theLine =~ /events = (.*)/ ){
            if( $curcolor eq "BLUE" ){
                $blueevents = $1;
            }elsif ( $curcolor eq "RED" ){
		$redevents = $1;
            }
        }

        if( $theLine =~ /power  = (.*)/ ){
            if( $curcolor eq "BLUE" ){
                $bluepower = $1;
            }elsif ( $curcolor eq "RED" ){
                $redpower = $1;
            }
        }
        if( $theLine =~ /filter = (.*)/ ){
            if( $curcolor eq "BLUE" ){
                $bluefilter = $1;
            }elsif ( $curcolor eq "RED" ){
                $redfilter = $1;
            }
        }

        if( $theLine =~ /delay  = (.*)/ ){
            if( $curcolor eq "BLUE" ){
                $bluedelay = $1;
            }elsif ( $curcolor eq "RED" ){
                $reddelay = $1;
            }
        }
    }

    if( $firstLaser )
    {
        $firstLaser = 0;
        if( $firstTP )
        {
            $firstTS    = $timestampbeg;
        }
    }
    $diffTS = $timestampbeg - $firstTS ;
    if($redevents >0 ){
        print LREDLIST "Run${run}_LB${lumiblock}\t$run\t$lumiblock\t$redevents\t$timestampbeg\t$timestampend\t$mgpagain\t$memgain\t$redpower\t$redfilter\t$reddelay\n";
    }
    if($blueevents > 0 ){
        print LBLUELIST "Run${run}_LB${lumiblock}\t$run\t$lumiblock\t$blueevents\t$timestampbeg\t$timestampend\t$mgpagain\t$memgain\t$bluepower\t$bluefilter\t$bluedelay\n";
    }
}
closedir( RUNSDIR );
close( LBLUELIST );
close( LREDLIST );

opendir( RUNSDIR, $testpulsedir ) || die "cannot open directory $dir\n";
@runsdir = sort  readdir( RUNSDIR );
foreach my $rundir (@runsdir)
{
    next unless( $rundir =~  /Run(\d*)_LB(.*)/ );
    $run = $1;
    $lumiblock = $2;

    my $curtype = "LASER";
    next if( $run < $firstRun || $run > $lastRun );
    next if ( !open( HEADERFILE, "${testpulsedir}/${rundir}/header.txt") );

    my $timestampbeg = 0;
    my $timestampend = 0;
    my $mgpagain = -1;
    my $memgain = -1;
    my $events = 0;

    while (<HEADERFILE>)
    {
        chomp($_);
        $theLine = $_;
        if( $theLine =~ /(.*) Events/ )
        {
            $curtype = $1;
            ( ($curtype =~ /LASER/) || ($curtype =~ /TESTPULSE/) ) || last;
        }
        if( $theLine =~ /RUN = (\d*)/ )
        {
            if ( $run != $1 ) {
                print "Run number not properly filled: $run versus $1 \n";
            }
        }

        if( $theLine =~ /RUNTYPE = (\d*)/ )
        {
            $runtype = $1;
            if( $curtype =~ /TESTPULSE/ ){
                ( $runtype == 7 || $runtype == 8 || $runtype == 17 )  ||  last;
            }
        }

        if($curtype =~ /TESTPULSE/) {

            if( $theLine =~ /TIMESTAMP_BEG = (\d*)/ ){ $timestampbeg = $1; }
            if( $theLine =~ /TIMESTAMP_END = (\d*)/ ){ $timestampend = $1; }
            if( $theLine =~ /MPGA_GAIN = (.*)/ ){ $mgpagain = $1; }
            if( $theLine =~ /MEM_GAIN  = (.*)/ ){ $memgain = $1; }
            if( $theLine =~ /EVENTS = (\d*)/ ){ $events = $1; }
        }
    }

    if( $firstTP )
    {
        $firstTP = 0;
        if( $firstLaser )
        {
            $firstTS = $timestamp_beg;
        }
    }
    if($events > 0 ){
        print TPLIST "Run${run}_LB${lumiblock}\t$run\t$lumiblock\t$events\t$timestampbeg\t$timestampend\t$mgpagain\t$memgain -1 -1 -1\n";
    }
}
closedir( RUNSDIR );
close( TPLIST );
