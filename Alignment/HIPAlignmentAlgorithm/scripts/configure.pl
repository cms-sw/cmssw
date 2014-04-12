#!/usr/bin/env perl
use File::Basename;

print "Configuring the python executables and run scripts...\n";

$odir = $ARGV[0];
#$datafile = $ARGV[1];
#$flag   = $ARGV[2];

#prendo dal file
$datafile1 = $ARGV[1];

open (datafile1) or die "Can't open the file!";
@dataFileInput1 = <datafile1>;

$j = 0;

foreach $data1 ( @dataFileInput1 ) {

$data1 =~ m/\,/;
$datafile = $`;
$flag1 = $';
$flag1 =~ m/$/;
$flag = $`;
#$flag = $';

print "Output directory: $odir \n";
print "Datafile: $datafile \n";

# open datafile, get skim name
open (datafile) or die "Can't open the file!";
@dataFileInput = <datafile>;

$dataskim = basename( $datafile, ".dat" );

system( "
cp python/common_cff_py.txt $odir/.;
cp python/$dataskim\TrackSelection_cff_py.txt $odir/.;
cp python/align_tpl_py.txt $odir/.;
cp python/collect_tpl_py.txt $odir/.;
" );


# open common_cff.py
$COMMON = "$odir/common_cff_py.txt";
open (COMMON) or die "Can't open the file!";
@commonFileInput = <COMMON>;

# open selections
$SELECTION = "$odir/$dataskim\TrackSelection_cff_py.txt";
open (SELECTION) or die "Can't open the file!";
@selectionsInput = <SELECTION>;

## setting up parallel jobs




foreach $data ( @dataFileInput ) {
	$j++;
	# do stuff
	# print "$data";
	system( "
	mkdir $odir/job$j; 
	cp python/align_tpl_py.txt $odir/job$j/align_cfg.py;
	cp scripts/runScript.csh $odir/job$j/.;
	" );
	# run script
	open OUTFILE,"$odir/job$j/runScript.csh";
        insertBlock( "$odir/job$j/align_cfg.py", "<COMMON>", @commonFileInput );
	insertBlock( "$odir/job$j/align_cfg.py", "<SELECTION>", @selectionsInput );
	# replaces for align job
	replace( "$odir/job$j/align_cfg.py", "<FILE>", "$data" );
	replace( "$odir/job$j/align_cfg.py", "<PATH>", "$odir/job$j" );
	replace( "$odir/job$j/align_cfg.py", "<SKIM>", "$dataskim" );
##flag
	replace( "$odir/job$j/align_cfg.py", "<FLAG>", "$flag" );	
	# replaces for runScript
        replace( "$odir/job$j/runScript.csh", "<ODIR>", "$odir/job$j" );
	replace( "$odir/job$j/runScript.csh", "<JOBTYPE>", "align_cfg.py" );
	close OUTFILE;
	system "chmod a+x $odir/job$j/runScript.csh";
}

}

system( "
mkdir $odir/main/;
cp python/initial_tpl_py.txt $odir/main/initial_cfg.py;
cp python/collect_tpl_py.txt $odir/main/collect_cfg.py;
cp python/upload_tpl_py.txt $odir/upload_cfg.py;
cp scripts/runScript.csh $odir/main/.;
cp scripts/runControl.csh $odir/main/.;
cp scripts/checkError.sh $odir/main/.;
");
## setting up initial job
replace( "$odir/main/initial_cfg.py", "<PATH>", "$odir" );
insertBlock( "$odir/main/initial_cfg.py", "<COMMON>", @commonFileInput );
replace( "$odir/main/initial_cfg.py", "<FLAG>", "" );	
## setting up collector job
replace( "$odir/main/collect_cfg.py", "<PATH>", "$odir" );
replace( "$odir/main/collect_cfg.py", "<JOBS>", "$j" );
insertBlock( "$odir/main/collect_cfg.py", "<COMMON>", @commonFileInput );
replace( "$odir/main/collect_cfg.py", "<FLAG>", "" );	
replace( "$odir/main/runScript.csh", "<ODIR>", "$odir/main" );
replace( "$odir/main/runScript.csh", "<JOBTYPE>", "collect_cfg.py" );
system "chmod a+x $odir/main/runScript.csh";

## setting up upload job
replace( "$odir/upload_cfg.py", "<PATH>", "$odir" );
insertBlock( "$odir/upload_cfg.py", "<COMMON>", @commonFileInput );

# replace sub routines #
###############################################################################

sub replace {
	
	$infile = @_[0];
	$torepl = @_[1];
	$repl = @_[2];
	
	
	open(INFILE,"$infile") or die "cannot open $infile";;
	@log=<INFILE>;
	close(INFILE);
	
	system("rm -f tmp");
	open(OUTFILE,">tmp");
	
	foreach $line (@log) {
		$linecopy = $line =~ s/$torepl/$repl/;
		if ($line =~ /$torepl/) { print OUTFILE $linecopy; }
		else { print OUTFILE $line; }
	}
	
	close(OUTFILE);
	system("mv tmp $infile");
	
}

sub insertBlock {
	
	($infile, $torepl, @repl) = @_;
	
	open(INFILE,"$infile") or die "cannot open $infile";;
	@log=<INFILE>;
	close(INFILE);
	
	system("rm -f tmp");
	open(OUTFILE,">tmp");
	
	foreach $line (@log) {
		if ($line =~ /$torepl/) { print OUTFILE @repl; }
		else { print OUTFILE $line; }
	}
	
	close(OUTFILE);
	system("mv tmp $infile");
	
}



