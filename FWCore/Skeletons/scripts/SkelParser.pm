package SkelParser;
# $Id $
###########################################################################
#  simple little script to make event setup producer skeletons
# 
#  execution:  mkesprod producername recordname datatype1 [ [datatype2] ...]  
# 
#  output:  producername/
#                         Buildfile
#                         interface/
#                         sr/producername.cc
#                               producername_DONT_TOUCH.cc
#                         doc/
#                         test/
#                         python/
#  required input:
# 
#  producername = name of the producer
#  recordname   = name of the record to which the producer adds data
#  datatype     = a list of the types of data created by the producer
# 
#  optional input:
# 
#  none
# 
#  example:
#  mkesprod MyProducer  MyRecord MyData
#        --> write  MyProducer/
#                               Buildfile
#                               interface/
#                               src/MyProducer.cc
#                               doc/
#                               test/
#                               python/  
#                              
#   the script tries to read in
#   a filename .tmpl in users HOME directory which contains the following lines
#             First : your first name
#             Last : your last name
#   if .tmpl is not found and firstname and lastname are blank the
#   enviroment variable LOGNAME is used to obtain the "real life" name
#   from the output of finger.
#
#   Enviroment variable CMS_SKEL may point to a directory that
#   contains the skeleton files.
#
#   mkesprod will not overwrite existing files
#
#   Skeleton Keywords (Case matters):
#      prodname  :  overwritten with routine name
#      John Doe  :  overwritten with author's name
#      day-mon-xx:  overwritten with todays date
#      RCS(key)  :  becomes $key$
#
#   author of the script: Chris Jones
#                         (based on scripts used by the CLEO experiment)
#   
###########################################################################

BEGIN {
    use Exporter ();
    our ($VERSION, @ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS);
    
    # set the version for version checking
    $VERSION     = 1.00;
    # if using RCS/CVS, this may be preferred
    $VERSION = sprintf "%d.%03d", q$Revision: 1.7 $ =~ /(\d+)/g;
    
    @ISA         = qw(Exporter);
    @EXPORT      = qw(&copy_file &make_file &grandparent_parent_dir &mk_package_structure &find_mkTemplate_dir);
    %EXPORT_TAGS = ( );     # eg: TAG => [ qw!name1 name2! ],
    
    # your exported package globals go here,
    # as well as any optionally exported functions
    @EXPORT_OK   = qw();
}
our @EXPORT_OK;

sub find_mkTemplate_dir {
  my $commandFullPath = $_[0];
  my $base_dir = $ENV{"CMS_SKEL"};
  if (!$base_dir) {
    #see if directory is in same directory as the script
    use File::Basename;
    $base_dir = dirname($commandFullPath);
    if (! -e "$base_dir/mkTemplates") {
      #see if the directory is in the user's release area
      $base_dir = $ENV{"CMSSW_BASE"};
      if(!$base_dir) { die "could not find environment variable CMSSW_BASE. Please setup scram runtime environment." }
      $base_dir = "$base_dir/src/FWCore/Skeletons/scripts";
      if(! -e "$base_dir/mkTemplates") {
        #get it from the release area
        $base_dir= $ENV{"CMSSW_RELEASE_BASE"};
        $base_dir="$base_dir/src/FWCore/Skeletons/scripts";
      }
    }
  }
  return "$base_dir/mkTemplates";

}


sub verifypath () {
  my $envvar = "CMSSW_BASE";
  if (!exists $ENV{$envvar}) {
    print STDERR "$envvar not set: Please do 'scramv1 run'.\n";
    return 0;
  }
  
  my $basepath = $ENV{$envvar};
  # strip off all but the last component of the basepath
  $basepath =~ s!.*/!!;

  my $cwd = `pwd`;
  chomp($cwd);

  # check that the current working directory is of the form
  # $basepath/src/something and return the result.
  # 
  # This form allows subsubdirectories:
  #$cwd =~ m!$basepath/src/..*!;

  # If subsubdirectories aren't allowed, use this form:    
  $cwd =~ m!$basepath/src/.[^/]*$!;
}


sub mk_package_structure {
  if (! verifypath() ) { die "Packages must be created in a 'subsystem'.\n  Please go to '\$CMSSW_BASE/src', create or choose a subdirectory from there\n  and then run the script from that subdirectory.\n"; }


    my $name = $_[0];

    mkdir("$name", 0777) || die "can not make dir $name";
    mkdir("$name/interface", 0777) || die "can not make dir $name/interface";
    mkdir("$name/src", 0777) || die "can not make dir $name/src";
    mkdir("$name/test", 0777) || die "can not make dir $name/test";
    mkdir("$name/doc", 0777) || die "can not make dir $name/doc";
    mkdir("$name/python",0777) || die "can not make dir $name/python";
}

sub grandparent_parent_dir {
    my $cwd;
    chomp($cwd = `pwd`);
    ($cwd =~ m!/([^/]*)/([^/]*)$!);
}

# copy file
sub copy_file {
# first argument is file to be copied
my $skeleton = $_[0]; 
# second argument is the name of the new file
my $outfile = $_[1];

if (-s "$outfile") {
    print "  W: $outfile FILE ALREADY EXISTS WILL NOT OVERWRITE!!\n";
    print "  W: *****************************************************\n";
} else {

#   write out some stuff to the screen
    print "  I: copying file $skeleton to $outfile\n";

#open the skeleton file and output file
    open(::IN,$skeleton)    or die "Opening $skeleton: $!\n";
    open(::OUT,">$outfile") or die "Opening $outfile: $!\n";

# loop over lines in "skeleton" and overwrite where neccessary
    while(<::IN>) {
	print ::OUT $_;
    }
    close(::IN);   
    close(::OUT);
}
}


# generate file
sub make_file {
# first argument is the skeleton file to use
my $skeleton = $_[0];
# second argument is the name of the output file
my $outfile = $_[1];

my $substitutions = $_[2];

my $magic_tokens;
if (exists $_[3] ) {
  $magic_tokens = $_[3];
}

my $author1 = $_[4];
my $author2 = $_[5];

if (-s "$outfile") {
    print "  W: $outfile FILE ALREADY EXISTS WILL NOT OVERWRITE!!\n";
    print "  W: *****************************************************\n";
} else {
#  get the current date
    $now = `date`;
    chop($now);

# get authors name from $HOME/.tmpl file

    $afrom = "command line";
    $author = "$author1 $author2";

    $home = $ENV{"HOME"};

    if ($author1 eq "" && -s "$home/.tmpl") {
       open(IN,"$home/.tmpl");
       $afrom = "users .tmpl file";
       while(<IN>) {
	  if (/First\w*/) {
	     @words = split(/:/, $_);
	     $author1 = $words[1]; 
	     chop($author1);
	  } elsif (/Last\w*/) {
	     @words = split(/:/, $_);
	     $author2 = $words[1];
	     chop($author2);
	  }
       }
       close(IN);
       $author = "$author1 $author2";
    }
#
# if author is still blank fill it in with REAL LIFE name
#
    if ($author1 eq "") {
	@words = getpwnam(getlogin());
	$author = $words[6];
	chomp($author);
	$afrom = "the gcos entry";
    }
#   write out some stuff to the screen
    print "  I: using skeleton: $skeleton \n";
    print "  I: authors name is: $author, determined by $afrom \n";
    print "  I: creating file: $outfile \n";

#open the skeleton file and output file
    open(IN,$skeleton)    or die "Opening $skeleton: $!\n";
    open(OUT,">$outfile") or die "Opening $outfile: $!\n";

# loop over lines in "skeleton" and overwrite where neccessary
    while(<IN>) {
#	print "size @$substitutions \n";
	foreach $subst (@$substitutions) {
	    #print $subst;
	    eval $subst;
	}
#  Preserve case for lowercase
#	s/prodname/$name/g;
#	s/recordname/$recordname/g;
#	s/skelsubsys/$gSUBSYS/g;
#  Map uppercase to uppercase
#	s/PRODNAME/\U$name/g;
	s/John Doe/$author/;
	s/day-mon-xx/$now/;
#  Handle RCS keywords
	s/RCS\((\w+)\)/\$$1\$/g;
	#print $_;
#  Handle embeded perl commands
	if( /\@perl(.*)@\\perl/ ) {
	    #print $1;
	    eval "$1";
	    #print $result;
	    s/\@perl(.*)@\\perl/$result/;
	}
#   Handle embedded examples
# expect tags for the line of form example_A(_B_C, etc.)
# print line if any of command line flags equals any of A(B,C etc.)
# so example_track_mc lines get printed if either -track or
# -mc is specified
        if ( /^\@example_(\S+)/ )
        {
          $okprint = 0;
          my @tokenlist = split '_', $1;
          foreach $token ( @$magic_tokens )
          {
            foreach $othertoken ( @tokenlist )
            {
              if ( $token eq $othertoken )
              {
                $okprint = 1;
                s/^\@example_(\S+) ?//;
                  my $value = $magic_values{$token};
                s/\@example=/$value/g;
                
              }
            }
          }
          if(! $okprint) {
            next
          }
        }
	print OUT $_;
    }
    close(IN);   
    close(OUT);
    if ($flag =~ /[bpC]/) {
	chmod(0755,$outfile);
    }
}
}


1;
