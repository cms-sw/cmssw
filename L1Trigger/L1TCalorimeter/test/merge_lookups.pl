#! /usr/bin/perl

sub usage() {
    print "usage:  perl_start.pl [OPTIONS] <file1> [... <filen>]\n";
    print "\n";
    print "Merge several lookup tables into one, using the upper bits.\n";
    print "WARNING:  currently several variables must be configured inside script by hand!\n";
    print "\n";
    print "Options are any of the following:\n";
    print "  --verbose             output detailed information.\n";
    print "  --help                print the message.\n";
    print "\n";
    exit 0;
}


$VERBOSE       = 0;

# Must configure this for each use case:
# (tauCalibrationLUT)
$NBITS         = 7;  # reserve this many bits for original address space
$HEADER        = "#<header> V1 10 9 </header>\n";
# (egIsoLUT)
# $NBITS         = 15;  # reserve this many bits for original address space
# $HEADER        = "#<header> V1 16 1 </header>\n";

@args = ();   
# parse the command line arguments for options:

while($arg = shift){
    if ($arg =~ /^--/){
	if ($arg =~ /--help/)           { usage();             }
	if ($arg =~ /--verbose/)        { $VERBOSE       = 1;  }
	#if ($arg =~ /--demo=(\S+)/)  { $DEMO   = $1; }
    } else {
	push @args,$arg;
    }
}     
    
# if ($#args < 2) { usage(); };
$findex = 0;


print "# A single LUT file assembled from several LUT files by merge_lookups.pl\n";
print $HEADER;

while ($file = shift @args){
    $foffset = $findex * (2**$NBITS);
    print "# LUT data extracted from file ", $file, ", will offset address by ";
    printf("0x%x\n", $foffset);
    

    
    open(INPUT, $file);
    while($line = <INPUT>){	
	# this version only allows data fields with no comments
	# if ($line =~ /^\s*(\d+)\s*(\d+)\s*$/){
	if ($line =~ /^\s*(\d+)\s*(\d+)/){
	    # print "# orig line:  ", $line;
	    # printf("# orig addr in hex:  0x%x\n", $1);
	    #printf("0x%x %d\n", $1+$foffset, $2);
	    printf("%d %d\n", $1+$foffset, $2);
	} else {
	    print $line;
	}
    }
    close(INPUT);    
    $findex++;
}
