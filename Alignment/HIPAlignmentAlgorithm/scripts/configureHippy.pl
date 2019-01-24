#!/usr/bin/env perl
use File::Basename;

print "Configuring the python executables and run scripts...\n";
$success=1;

$odir = $ARGV[0];
$datafile1 = $ARGV[1];
$iovrange = $ARGV[2];
$incommoncfg = $ARGV[3];
$inaligncfg = $ARGV[4];
$intrkselcfg = $ARGV[5];
$inuseSurfDef = $ARGV[6];
$inredirectProxy = $ARGV[7];

$strUseSD = "";
if ($inuseSurfDef == 0){
   $strUseSD = "False";
}
else{
   $strUseSD = "True";
}


open (datafile1) or die "Can't open the file!";
@dataFileInput1 = <datafile1>;

open (iovrange) or die "Can't open the iovfile!";
@iovInput1 = <iovrange>;

print "IOV file: $iovrange \n";

$j = 0;
$k = 0;

foreach $iovv ( @iovInput1) {
   chomp($iovv);
   $iovstr .= "$iovv,";
}
chop($iovstr);
print "IOVs: $iovstr\n";

system( "
mkdir -p $odir/main/;
cp $incommoncfg $odir/common_cff_py.txt;
cp $inaligncfg $odir/align_tpl_py.txt;
cp python/initial_tpl_py.txt $odir/;
cp python/collect_tpl_py.txt $odir/;
cp python/upload_tpl_py.txt $odir/;
cp scripts/runScript.csh $odir/;
cp scripts/runControl.csh $odir/main/;
cp scripts/checkError.sh $odir/main/;
");
$success*=replace( "$odir/common_cff_py.txt", "<iovs>", "$iovstr" );
$success*=replace( "$odir/common_cff_py.txt", "<SURFDEFOPT>", "$strUseSD" );
$success*=replace( "$odir/runScript.csh", "<PROXYREDIRECT>", "$inredirectProxy" );


foreach $data1 ( @dataFileInput1 ) {

   chomp $data1;
   next if (index($data1, '#') >= 0);

   my @dataspecs = split(',', $data1);
   $datafile = $dataspecs[0];
   #($dataskim,$path,$suffix) = fileparse($datafile,,qr"\..[^.]*$");
   $trkselfile = $dataspecs[1];
   chomp $trkselfile;
   $flag = $dataspecs[2];
   $flaglower = lc($flag);
   if ( $trkselfile eq "" ){
      $trkselfile = "$flaglower\TrackSelection_cff_py.txt";
   }
   print "Picking track selection configuration from $trkselfile \n";
   $flagopts = "NOOPTS";
   if (defined($dataspecs[3])){
      print "A flag option is defined.\n";
      $flagopts = $dataspecs[3];
   }
   else{
      print "No flag option is defined.\n";
   }

   print "Output directory: $odir \n";
   print "Datafile: $datafile \n";
   print "Flag: $flag \n";
   print "Flag options: $flagopts \n";

   # open datafile, get skim name
   open (datafile) or die "Can't open the file!";
   @dataFileInput = <datafile>;

   system( "
   cp $intrkselcfg/$trkselfile $odir/;
   " );


   # open common_cff.py
   $COMMON = "$odir/common_cff_py.txt";
   open (COMMON) or die "Can't open the file!";
   @commonFileInput = <COMMON>;

   # open selections
   $SELECTION = "$odir/$trkselfile";
   open (SELECTION) or die "Can't open the file!";
   @selectionsInput = <SELECTION>;

   ## setting up parallel jobs
   foreach $data ( @dataFileInput ) {
      chomp($data);
      if ( ( $data ne "" ) and ( index($data, '#') == -1 ) ){
         $jsuccess=1;
         $j++;
         # do stuff
         # print "$data";
         system( "
         mkdir -p $odir/job$j;
         cp $odir/align_tpl_py.txt $odir/job$j/align_cfg.py;
         cp $odir/runScript.csh $odir/job$j/;
         " );
         # run script
         open OUTFILE,"$odir/job$j/runScript.csh";
         insertBlock( "$odir/job$j/align_cfg.py", "<COMMON>", @commonFileInput );
         insertBlock( "$odir/job$j/align_cfg.py", "<SELECTION>", @selectionsInput );
         # $success*=replaces for align job
         $jsuccess*=replace( "$odir/job$j/align_cfg.py", "<FILE>", "$data" );
         $jsuccess*=replace( "$odir/job$j/align_cfg.py", "<PATH>", "$odir/job$j" );
         #$jsuccess*=replace( "$odir/job$j/align_cfg.py", "<SKIM>", "$dataskim" );
         $jsuccess*=replace( "$odir/job$j/align_cfg.py", "<FLAGOPTS>", "$flagopts" );
         $jsuccess*=replace( "$odir/job$j/align_cfg.py", "<FLAG>", "$flag" );
         # $success*=replaces for runScript
         $jsuccess*=replace( "$odir/job$j/runScript.csh", "<ODIR>", "$odir/job$j" );
         $jsuccess*=replace( "$odir/job$j/runScript.csh", "<JOBTYPE>", "align_cfg.py" );
         close OUTFILE;
         system "chmod a+x $odir/job$j/runScript.csh";
         if ($jsuccess == 0){
            print "Job $j did not setup successfully. Decrementing job number back.\n";
            system "rm -rf $odir/job$j";
            $j--;
         }
      }
   }


}

foreach $iov ( @iovInput1) {
   chomp($iov);
   print "Configuring IOV $iov\n";
   $k++;
   system( "
   cp $odir/upload_tpl_py.txt $odir/upload_cfg_$k.py;
   cp $odir/initial_tpl_py.txt $odir/main/initial_cfg_$k.py;
   cp $odir/collect_tpl_py.txt $odir/main/collect_cfg_$k.py;
   cp $odir/runScript.csh $odir/main/runScript_$k.csh;
   " );
   # run script
   ## setting up initial job
   $success*=replace( "$odir/main/initial_cfg_$k.py", "<PATH>", "$odir" );
   insertBlock( "$odir/main/initial_cfg_$k.py", "<COMMON>", @commonFileInput );
   #$success*=replace( "$odir/main/initial_cfg_$k.py", "<FLAG>", "" );
   $success*=replace( "$odir/main/initial_cfg_$k.py", "<iovrun>", "$iov" );
   ## setting up collector job
   $success*=replace( "$odir/main/collect_cfg_$k.py", "<PATH>", "$odir" );
   $success*=replace( "$odir/main/collect_cfg_$k.py", "<JOBS>", "$j" );
   insertBlock( "$odir/main/collect_cfg_$k.py", "<COMMON>", @commonFileInput );
   #$success*=replace( "$odir/main/collect_cfg_$k.py", "<FLAG>", "" );
   $success*=replace( "$odir/main/collect_cfg_$k.py", "<iovrun>", "$iov" );
   $success*=replace( "$odir/main/runScript_$k.csh", "<ODIR>", "$odir/main" );
   $success*=replace( "$odir/main/runScript_$k.csh", "<JOBTYPE>", "collect_cfg_$k.py" );
   ## setting up upload job
   $success*=replace( "$odir/upload_cfg_$k.py", "<PATH>", "$odir" );
   $success*=replace( "$odir/upload_cfg_$k.py", "<iovrun>", "$iov" );
   insertBlock( "$odir/upload_cfg_$k.py", "<COMMON>", @commonFileInput );
   #close OUTFILE;
   system "chmod a+x $odir/main/runScript_$k.csh";
}

if($success==0){
   system("touch $odir/ERROR");
}



# replace sub routines #
###############################################################################

sub replace {
   $result = 1;

   $infile = @_[0];
   $torepl = @_[1];
   $repl = @_[2];

   $tmpindc = "tmp";
   $tmpfile = "$infile$tmpindc";
   if( $repl =~ /^$/ ){
      print "Replacing lines $torepl with empty line in $tmpfile is not possible! \n";
      $result = 0;
   }
   elsif( $repl !~ /\S*/ ){
      print "Replacing lines $torepl with a line matching whitespace in $tmpfile is not possible! \n";
      $result = 0;
   }
   open(INFILE,"$infile") or die "cannot open $infile";
   @log=<INFILE>;
   close(INFILE);

   system("rm -f $tmpfile");
   open(OUTFILE,">$tmpfile");

   foreach $line (@log) {
      $linecopy = $line;
      $linecopy =~ s|$torepl|$repl|;
      print OUTFILE $linecopy;
   }

   close(OUTFILE);
   system("mv $tmpfile $infile");

   return $result
}

sub insertBlock {

   ($infile, $torepl, @repl) = @_;

   open(INFILE,"$infile") or die "cannot open $infile";;
   @log=<INFILE>;
   close(INFILE);

   $tmpindc = "tmp";
   $tmpfile = "$infile$tmpindc";

   system("rm -f $tmpfile");
   open(OUTFILE,">$tmpfile");

   foreach $line (@log) {
      if ($line =~ /$torepl/) { print OUTFILE @repl; }
      else { print OUTFILE $line; }
   }

   close(OUTFILE);
   system("mv $tmpfile $infile");

}
