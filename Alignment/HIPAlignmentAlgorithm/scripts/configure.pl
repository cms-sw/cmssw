#!/usr/bin/env perl
use File::Basename;

print "Configuring the python executables and run scripts...\n";

$odir = $ARGV[0];
$datalist = $ARGV[1];
@datafile=split(/;/,$datalist);

$dircount=0;

foreach $dataitem ( @datafile ){
    print "Output directory: $odir \n";
    print "Datafile: $dataitem \n";
    
# open datafile, get skim name
    open (dataitem) or die "Can't open the file!";
    @dataFileInput = <dataitem>;
    
    $dataskim = basename( $dataitem, ".dat" );
    
    system( "
cp python/common_cff_py.txt $odir/.;
cp python/$dataskim\TrackSelection_cff_py.txt $odir/.;
cp python/align_tpl.py $odir/.;
cp python/collect_tpl.py $odir/.;
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
$j = $dircount;

foreach $data ( @dataFileInput ) {
	$j++;
	# do stuff
	# print "$data";
	system( "
	mkdir $odir/job$j; 
	cp python/align_tpl.py $odir/job$j/align_cfg.py;
	cp scripts/runScript.csh $odir/job$j/.;
	" );
	# run script
	open OUTFILE,"$odir/job$j/runScript.csh";
	# replaces for align job
	replace( "$odir/job$j/align_cfg.py", "<FILE>", "$data" );
	replace( "$odir/job$j/align_cfg.py", "<PATH>", "$odir/job$j" );
	replace( "$odir/job$j/align_cfg.py", "<SKIM>", "$dataskim" );
	insertBlock( "$odir/job$j/align_cfg.py", "<COMMON>", @commonFileInput );
	insertBlock( "$odir/job$j/align_cfg.py", "<SELECTION>", @selectionsInput );
	# replaces for runScript
	replace( "$odir/job$j/runScript.csh", "<ODIR>", "$odir/job$j" );
	replace( "$odir/job$j/runScript.csh", "<JOBTYPE>", "align_cfg.py" );
	close OUTFILE;
	system "chmod a+x $odir/job$j/runScript.csh";
}
    $dircount=$j;
system( "
mkdir $odir/main;
cp python/initial_tpl.py $odir/main/initial_cfg.py;
cp python/collect_tpl.py $odir/main/collect_cfg.py;
cp python/upload_tpl.py $odir/upload_cfg.py;
cp scripts/runScript.csh $odir/main/.;
cp scripts/runControl.csh $odir/main/.;
cp scripts/checkError.sh $odir/main/.;
");
## setting up initial job
replace( "$odir/main/initial_cfg.py", "<PATH>", "$odir" );
insertBlock( "$odir/main/initial_cfg.py", "<COMMON>", @commonFileInput );

## setting up collector job
replace( "$odir/main/collect_cfg.py", "<PATH>", "$odir" );
replace( "$odir/main/collect_cfg.py", "<JOBS>", "$j" );
insertBlock( "$odir/main/collect_cfg.py", "<COMMON>", @commonFileInput );
replace( "$odir/main/runScript.csh", "<ODIR>", "$odir/main" );
replace( "$odir/main/runScript.csh", "<JOBTYPE>", "collect_cfg.py" );
system "chmod a+x $odir/main/runScript.csh";
system "chmod a+x $odir/main/checkError.sh";

## setting up upload job
replace( "$odir/upload_cfg.py", "<PATH>", "$odir" );
insertBlock( "$odir/upload_cfg.py", "<COMMON>", @commonFileInput );

}#end loop on datafiles

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



