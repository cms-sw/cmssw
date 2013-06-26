#! /usr/bin/env perl 
use File::Basename;

print "Configuring the python executables and run scripts...\n";

## dir is the place where files live
## odir is the place where you want to have the output
$iter = $ARGV[0];
$dir = $ARGV[1];
$odir = $ARGV[2];


# for controlling which processes to run
# set all to 1 if running the whole chain
$createDBs = 1;
$intoNtuples = 1;
$geomComparison = 1;
$plotFigs = 1;


# 0. Set path/configuration
print "Set path and configuration... \n";
#system("
#cd ../..
#eval `scramv1 runtime -csh`
#cd -
#");

# open common_cff.py
print "Opening common... \n";
$COMMON = "python/common_cff_py.txt";
open (COMMON) or die "Can't open the file!";
@commonFileInput = <COMMON>;

### start loop over number of iterations

# loop
print "Starting loop... \n";
for ($j = 1; $j <= $iter; $j++){
	print "Loop $j \n";	
	
	if ($createDBs == 1){
		# 1. configure the upload_serial_tpl_py.txt
		system("
		cp python/upload_serial_tpl_py.txt $dir/upload_serial_cfg.py
		");
		replace( "$dir/upload_serial_cfg.py", "<PATH>", "$dir" );
		replace( "$dir/upload_serial_cfg.py", "<OUTPATH>", "$odir" );
		replace( "$dir/upload_serial_cfg.py", "<N>", "$j" );
		insertBlock( "$dir/upload_serial_cfg.py", "<COMMON>", @commonFileInput );
		
		print "Create IOIteration.root ... \n";
		# 2. configure the IOIteration.root file
		system("
		cp data/IOIteration_serial.root $dir/.
		");
		replace( "$dir/IOIteration_serial.root", "<ITER>", "$j" );
		
		# 3. run temporary upload_serial_cfg.py
		print "Run upload_serial... \n";
		system("
		cmsRun $dir/upload_serial_cfg.py
		");
	}
	
	if ($intoNtuples == 1){
		# 4. convert .db to .root, remove.db
		system("
		cp test/serialIntoNtuples_tpl.py $odir/serialIntoNtuples_cfg.py
		");
		replace( "$odir/serialIntoNtuples_cfg.py", "<PATH>", "$odir" );
		replace( "$odir/serialIntoNtuples_cfg.py", "<N>", "$j" );
		system("
		cmsRun $odir/serialIntoNtuples_cfg.py
		rm $odir/alignments_$j.db
		");
	}
	
	if ($geomComparison == 1){
		# 5. do geometry comparison
		system("
		cp test/serialGeomCompare_tpl.py $odir/serialGeomCompare_cfg.py
		");
		replace( "$odir/serialGeomCompare_cfg.py", "<PATH>", "$odir" );
		replace( "$odir/serialGeomCompare_cfg.py", "<N>", "$j" );
		system("
		cmsRun $odir/serialGeomCompare_cfg.py
		");
	}
	
	if ($plotFigs == 1){
		# 6. generate figures
		system("
		cp scripts/compareGeomSerial_tpl.C scripts/compareGeomSerial.C
		");
		replace( "scripts/compareGeomSerial.C", "<PATH>", "$odir" );
		replace( "scripts/compareGeomSerial.C", "<N>", "$j" );
		system("
		cd scripts
		root -l -q -b 'compareGeomSerial.C()'
		cd ../
		");
	}
	
}




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
