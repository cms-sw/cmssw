#!/usr/bin/perl -w
#
#use strict;
#use warnings;
use HTTP::Request::Common;
use HTTP::Response;
use LWP::UserAgent;
use Crypt::SSLeay;
use Getopt::Long;
use Cwd;
use Fcntl qw(:DEFAULT :flock);
use File::stat;

#program version
my $VERSION="0.1";

GetOptions(
           "h",         \$opt_h,                # short help
           "help",      \$longhelp,             # long help
           "a=s",       \$opt_a,                # set authorization method in MCDB (login, cert, pkcs12, globus)
           "header=s",       \$header,                # set the type of LHEF header (hepml or MG - MadGraph)
           "dsname=s",       \$dsname,                # set DataSetName
           "verbose",       \$verbose,                # be verbose
           "debug",       \$debug,                # be very verbose
           "replace",       \$replace,                # replace description in the article -artid from the header of first file in the argument list
           "not2web",       \$not2web,                # do not post article to WEB (default is post after description is complete)
           "authors=s",       \$authors,                # set additional authors logins
           "category=s",       \$category,                # set categories for this article
           "descriptiononly",       \$descriptiononly,    # do not upload any files only create article with the header from first file in the argument list
           "uploadonly",       \$uploadonly,              # do upload files to article -artid only, do not parse header and create article
           "artid=s",       \$artid                # set manually the ArticleID to upload files or set description 
          );

# Print help message
 help() unless @ARGV;
 help() if $opt_h ;
 help_long() if  $longhelp;

#Define authorization method for MCDB: login/password (login); LCG certificate (cert) or PKCS12 LCG certificate (pkcs12);
#please, provide the necessary information below for the corresponding type of authorization


my $session_id="";
my $req="";
my $ArticleID="";
my $GSIPROXY ="";
my $auth= $opt_a || "globus";
my $header_type = $header || "MG";

#....RUN main subroutine
eval { &main; };

#....If Errors
if ($@) {
    print "Error: $@  \n";
}
exit;



sub main {


my @files=();

print "The following files are going to be uploaded to MCDB and CASTOR:\n" if defined $verbose and not defined $descriptiononly;
print "The description will be defined from the first file:\n" if defined $verbose and defined $descriptiononly;

if(defined $uploadonly and !( defined $artid or defined $dsname)){
   print "You have to specify the -artid or -dsname option with option -uploadonly \n";
   exit(1);
}

 foreach my $file (@ARGV)
   {
     if( !-e $file ){print "FILE $file does not exist, please, check it. \n"; exit(1);}
     if( !-r $file ){print "FILE $file is not readable, please, check it. \n"; exit(1);}
#     if( !-T $file ){print "FILE $file is not a TEXT file, please, check it. LHEF file should be a text file \n"; exit(1);}
     push @files, $file ;
     print "$file \n" if defined $verbose;
   }


# WEB Agent description, necessary to send the header of LHEF to MCDB
$ua = LWP::UserAgent->new();
$ua->agent('MCDBUpload/0.1 ');
$ua->from('$ENV{USER} at $ENV{HOSTNAME}');
$ua->protocols_allowed( [ 'https'] );
my $filebat= $ARGV[0];
$ENV{HTTPS_VERSION} = 3;
my @request_form=();
my $hepml="";

#Authorization in MCDB
 if($auth eq "login"){
    my $netloc="mcdb.cern.ch:443";
    my $realm="MCDB login";
    my $uname=""; #CERN AFS login name
    my $pass=''; #CERN AFS password
    $ua->credentials($netloc, $realm, $uname => $pass);
  }elsif($auth eq "pkcs12"){
     # CLIENT PKCS12 CERT SUPPORT
      $ENV{HTTPS_PKCS12_FILE}     = '~/.globus/pkcs12.pkcs12';
      $ENV{HTTPS_PKCS12_PASSWORD} = 'PKCS12_PASSWORD';  # you can put here <STDIN> to enter it during the run
  }elsif($auth eq "cert"){
      $ENV{HTTPS_CERT_FILE} = '~/.globus/usercert.pem';
      $ENV{HTTPS_KEY_FILE}  = '~/.globus/userkey.pem';
  }else{
      $ENV{HTTPS_CA_DIR} = (defined $ENV{X509_CERT_DIR})?$ENV{X509_CERT_DIR}:"/etc/grid-security/certificates";
      $GSIPROXY = (defined $ENV{X509_USER_PROXY})?$ENV{X509_USER_PROXY}:"/tmp/x509up_u$<";
      $ENV{HTTPS_CA_FILE} = $GSIPROXY;
      $ENV{HTTPS_CERT_FILE} = $GSIPROXY;
      $ENV{HTTPS_KEY_FILE}  = $GSIPROXY;
      $ENV{HTTPS_DEBUG} = 1 if defined $debug;

  }
#END Authorization in MCDB

# Extract the header from LHEF file  
 $hepml=extract_hepml_header($files[0]) if $header_type eq "hepml";
 $hepml=extract_madgraph_header($files[0]) if $header_type eq "MG";

# Try to find the previous unfinished uploading sessions from the same directory 
 @old_sessions= glob("mcdb_upload*.session");
 if(scalar(@old_sessions) > 0 ){
  print "Something is wrong with the previous uploading jobs, there are ", $#old_sessions+1," old sessions in this directory: @old_sessions \n"; 
  exit(1);
 }
# elsif( scalar(@old_sessions) == 1 ){
#  $old_sessions[0] =~ /mcdb_upload_(.*).session/;
#  $old_session_id = $1;
#  print "Continue the previous uploading session: $session_id \n";
#  push @request_form, SESSION => "$old_session_id";
# }
    
  
  # Form the request to MCDB 
  push @request_form, HEPML => $hepml if $hepml ne "";
  push @request_form, DSNAME => $dsname if defined $dsname;
  push @request_form, ARTICLEID => $artid if defined $artid;
  push @request_form, REPLACEHEADER => "yes" if defined $replace;
  push @request_form, NOTTOWEB => "1" if defined $not2web;
  push @request_form, AUTHORS => $authors if defined $authors;
  push @request_form, CATEGORIES => $category if defined $category;
  push @request_form, HEPMLTYPE => $header_type if $hepml ne "";
  
  
  print "@request_form \n" if defined $debug;
  $req = POST 'https://mcdb.cern.ch/cgi-bin/authors/upload_server.cgi', \@request_form ;


# Request to send LHEF HEADER to MCDB and get the CASTOR directory to upload the file 
 $response = $ua->request($req);
 if ($response->is_success) {
     print "Header has been uploaded to MCDB: ",$response->content, "\n" if defined $verbose;
 } else {
     print "ERROR in the header uploading to MCDB: ",$response->as_string(), "\n";
     exit(1);
 }

 $session_id = $response->header('SESSIONID');
 $import_dir = $response->header('CASTORDIR');
 $ArticleID = $response->header('ARTID');
 $MCDBDSNAME = $response->header('MCDBDSNAME');
 $MCDB_response = $response->content;

 if($session_id eq "" or $import_dir eq "" or  $ArticleID eq ""){
   print "something is wrong on MCDB side, MCDB did not provide session_id: $session_id or castor dir: $import_dir or Article ID: $ArticleID \n";
   exit(1);
 }

 if(defined $dsname and  $MCDBDSNAME ne $dsname )
   {print "DataSetName: $MCDBDSNAME returned from MCDB is not equal to the defined by user $dsname for this article $ArticleID The priority is for the ArticleID, not DataSetName. \n";} 

 my $OLD_ARTICLE_ID="";
 my $OLD_CASTOR_DIR="";
 if(scalar(@old_sessions)==1 and $session_id ne $old_session_id){
   print "Old session id $old_session_id does not match the new one $session_id provided by MCDB server \n";
 }
 
 my $session_file="mcdb_upload_".$session_id.".session" ;
 my $session_log="mcdb_upload_".$session_id.".log";
 print "SESSION_FILE: $session_file\n" if defined $verbose;

 sysopen(SESSION_FILE, $session_file, O_CREAT | O_RDWR | O_SYNC)
      or die "Cannot write to $session_file: $!" ;

  
  # check for the previous broken session, try to restore it
  if(scalar(@old_sessions)==1){
    while(<SESSION_FILE>){
      if(/ArticleID=\s*(\S*).*/){$OLD_ARTICLE_ID=$1;}
      if(/CASTORDIR=\s*(\S*).*/){$OLD_CASTOR_DIR=$1;}
      
    }
    if($OLD_CASTOR_DIR ne $import_dir){
      print "Different uploading directories in local old session config and provided from MCDB server, use the last one \n";

    }
     if($OLD_ARTICLE_ID ne $ArticleID){
      print "Different ArticleID in local old session config and provided from MCDB server, use the last one \n";

    }
  }

   seek(SESSION_FILE,0,0);
   print SESSION_FILE "ArticleID=$ArticleID \n";
   print SESSION_FILE "DataSetName=$MCDBDSNAME \n";
   print SESSION_FILE "CASTORDIR=$import_dir \n";
   print SESSION_FILE $MCDB_response;
   print $MCDB_response if defined $verbose;
 
 # Uploading of files to CASTOR and registration of the uploaded file in MCDB
 if(not defined $descriptiononly){ 
   foreach my $file (@files){
    eval { &file_upload_grid($file, $import_dir); };
# ....If Errors
     if ($@) {
      print SESSION_FILE "!!! ERROR in uploading of file $file to CASTOR: $@ \n";
      print "!!! ERROR in uploading of $file to CASTOR: $@  \n";
      return($@);
     }
    
    print SESSION_FILE "File $file has been uploaded to CASTOR \n";
    print "File $file has been uploaded to CASTOR \n" if defined $verbose;
    
    $hepml=extract_hepml_header($files[0]) if $header_type eq "hepml";
    $hepml=extract_madgraph_header($files[0]) if $header_type eq "MG";
    $#request_form=0;
    push @request_form, [ ARTICLEID => $ArticleID, UPLOADEDFILE => $file, SIZE => (stat($file))[7], SESSIONID => $session_id, CASTORDIR => $import_dir, HEPML => $hepml];
        
        
    $req = POST 'https://mcdb.cern.ch/cgi-bin/authors/upload_server.cgi', \@request_form ;
    $response = $ua->request($req);
     if ($response->is_success) {
      print SESSION_FILE "File $file has been registered in MCDB: \n",$response->content, "\n";
      print  "File $file has been registered in MCDB: \n",$response->content, "\n" if defined $verbose;
     }else{
      print "ERROR during the registration of $file in MCDB: ",$response->as_string(), "\n";
      exit(1);
     }
 
   }
 }
    
    # Finish the session and post article to the WEB
#    $req = POST 'https://mcdb.cern.ch/cgi-bin/authors/upl.cgi', [ ARTICLEID => $ArticleID, POSTARTICLE => "yes" ] ;
#    $response = $ua->request($req);
#    if ($response->is_success) {
#      print SESSION_FILE "Article $ArticleID has been posted to MCDB web page: ",$response->content, "\n";
#      print  "Article $ArticleID has been posted to MCDB web page:  ",$response->content, "\n" if defined $verbose;
#    }else{
#      print "ERROR during the post of $ArticleID to MCDB web page: ",$response->as_string(), "\n";
#      exit(1);
#    }
 
 close SESSION_FILE;
 system ("mv","$session_file","$session_log");
 exit (0);

}

sub file_upload_grid
{
  my $file_to_upload=shift;
  my $remote_directory=shift;
  print "$file_to_upload copy to $remote_directory\n";

system( "grid-proxy-info", "-e");
if($? != 0) {
  print "No valid proxy found, do new authorization\n";
  system "grid-proxy-init";
}

my $localdir = cwd;

  my $fullname;
  if(substr($file_to_upload, 0, 1) eq "/") {
    $fullname = $file_to_upload;
  }
  else {
    $fullname = cwd() . "/" . $file_to_upload;
  }

  print "copying $file_to_upload to gsiftp://castorgrid.cern.ch\n";
  system ("globus-url-copy", "file:$fullname ",
        "gsiftp://castorgrid.cern.ch/castor/cern.ch/sft/mcdb/incoming/$remote_directory");

#  print "register in local RC\n";
#  system "globus-job-run lxshare0219.cern.ch /bin/bash -c " .
#         "\"export GDMP_CONFIG_FILE=/opt/edg/biome/etc/gdmp.conf; " . 
#         "/opt/edg/bin/gdmp_register_local_file -d /flatfiles/SE1/biome\"";


#print "publish local RC\n";
#system "globus-job-run lxshare0219.cern.ch /bin/bash -c " .
#       "\"export GDMP_CONFIG_FILE=/opt/edg/biome/etc/gdmp.conf; " .
#       "/opt/edg/bin/gdmp_publish_catalogue\"";



}

sub extract_hepml_header 
{
  my $file=shift;
#   print "$file\n";
  my $string="";
  my $write="0";
  open(BATCH, "<$file") or die "Can't open $filebat for reading: $!\n";

  while(<BATCH>){
  #print $_;
   if(/<header>/){
      $write="1";
      $string = $_;
    }elsif ($write){
      $string .= $_;
    }
    if(/<\/header>/){last;}
    if(/<\/init>/){last;}
  }

close BATCH;
return $string;
}

sub extract_madgraph_header 
{
  my $file=shift;
#   print "$file\n";
  my $string="";
  my $write="0";
  open(BATCH, "<$file") or die "Can't open $filebat for reading: $!\n";

  while(<BATCH>){
  #print $_;
   if(/<LesHouchesEvents/.../<\/init>/){
      $string .= $_;
    }
    if(/<\/init>/){last;}
  }

close BATCH;
return $string;
}


sub help
{
 print
  "Usage:
$0 file1.lhef file2.lhef /scratch/file3.lhef ...
$0 -h  		#short help
$0 --help  	#long help
$0 -replace [-artid N] [-dsname DataSetName]  files  	#replace the description in MCDB 
$0 -descriptiononly  	#do not upload any file only create article with the header of the first file in the argument list
$0 -uploadonly [-artid N] [-dsname DataSetName] 	#do only uploading of files to article -artid or -dsname
 -a [login, pkcs12, cert, globus] 	#type of authorization in MCDB, default is globus
 -header [MG, hepml] 	#specify type of LHEF header (MG - MadGraph, hepml - HepML header)
 -dsname DataSetName 	#specify Data Set Name (analog ArticleID)
 -artid N 		#specify ArticleID in MCDB (supersede -dsname if in conflict on MCDB side)
 -authors AFSlogin1,AFSlogin2,... 	#set additional authors for the article
 -category Category1,Category2,... 	#set MCDB Category where to attache article (default is CMS08MG)
 -not2web	#do not post Article to WEB (keep in MCDB), default is post right after it is described
 -verbose	#be verbose during the run
 -debug		#print additional information during the session 
";
 exit(0);

}

sub longhelp
{
 print "
   The main purpose of this script is to provide the client part of 
Automatic Uploadnig Interface of MCDB. It parses the header of LHEF file
(MadGraph or HepML) and describe the LHEF sample(s) in MCDB automatically.
This interface can upload other types of files as well, but without automatic
description in MCDB (options -uploadonly [-artid N] [-dsname DataSetName]),
therefore you need to specify the exist article or DataSetName in MCDB.
     
     This script extract the LHEF header of the first file in the argument list
and POST it to MCDB. MCDB parses the header and create article in MCDB 
(or replace the description with option -replace [-artid N] [-dsname DataSetName]).
MCDB returns ArticleID and PATH to the specific incoming directory where the 
samples can be upload directly. This script $0 copy the files from argument list
to specific CASTOR directory by means of globus-url-copy and after the file has
been uploaded $0 register it in MCDB. The LHEF header will be parsed to set the 
number of events, cross section, cross section errors, size of the file.
In case this is not LHEF file the header will not be parsed and file will not 
be described automatically.

     Authorization is the important point of all transactions in Internet and
we have to realise it in this interface as well. Two basic types are:
1) CERN AFS login    2) LCG GRID certificate
One of them (or both) should be registered in MCDB via New Author Registration.
    The default and most easy way to authorize for these transactions is to use 
grid-proxy-init 
This is the only possible method to upload samples to CASTOR with this script. 
For the transactions with MCDB (set new article, etc.) several more methods are
possible: 1) CERN AFS login and passwords      2) PKCS12 LCG certificate 
3) PEM LCG certificate (~/.globus/usercert.pem and ~/.globus/userkey.pem)
4) the default is grid-proxy-init 
The first three methods require passwords during the run and not very flexibal
for the production chain. If you are interesting to use non default authorizations,
please, check the necessary parameters inside the script.

     Several tasks which are implemented in this interface are described below
with some examples. The identification of the article in MCDB provided by ArticleID,
but it is also possible to identify you article with DataSetName. If you set both
ArticleID and DataSetName, but they are conflict in MCDB (wrong DataSetName for 
this ArticleID) ArticleID is superseded DataSetName and new DataSetName set for this
Article. Please, do not mix files from different physics processes in one 
uploading run, because all of them are described from the header of the 
first file and will be attached to the same article.

1) The main task is to describe the set of LHEF (MadGraph or HepML header) files
in MCDB as the new article and upload the files to specific CASTOR directory. 
The examples:
$0 file1 file2 ...
In this case the description for new MCDB article will be taken from the header 
of file1 but all other files will be upload and attached to the same article.
Possible options:
-dsname DataSetName 	#specify Data Set Name 
-header [MG, hepml] 	#specify type of LHEF header (MG - MadGraph, hepml - HepML header)
-authors AFSlogin1,AFSlogin2,...   	#set additional authors for the article
-category Category1,Category2,... 	#set MCDB Category where to attache article (default is CMS08MG)
-not2web	#do not post Article to WEB (keep in MCDB), default is post right after it is described
-verbose	#be verbose during the run
-debug		#print additional information during the session 

2) Upload more samples to the exist MCDB article (do not change the description):
$0 [-artid N] [-dsname DataSetName] --uploadonly file1 file2 ... 
Possible options:
-verbose	#be verbose during the run
-debug		#print additional information during the session 

3) Replace the description in the exist article and upload new files
$0 [-artid N] [-dsname DataSetName] --replace  file1 file2 ...
Possible options:
-dsname DataSetName 	#specify Data Set Name (analog ArticleID)
-header [MG, hepml] 	#specify type of LHEF header (MG - MadGraph, hepml - HepML header)
-authors AFSlogin1,AFSlogin2,...   	#set additional authors for the article
-category Category1,Category2,... 	#set MCDB Category where to attache article (default is CMS08MG)
-not2web	#do not post Article to WEB (keep in MCDB), default is post right after it is described
-verbose	#be verbose during the run
-debug		#print additional information during the session 

4) Describe sample in new article but do not upload any file to CASTOR.
The description is taken from the header of the file 
$0 -descriptiononly file1
Possible options:
-dsname DataSetName 	#specify Data Set Name (analog ArticleID)
-a [login, pkcs12, cert, globus] 	#type of authorization in MCDB, default is globus
-header [MG, hepml] 	#specify type of LHEF header (MG - MadGraph, hepml - HepML header)
-authors AFSlogin1,AFSlogin2,...   	#set additional authors for the article
-category Category1,Category2,... 	#set MCDB Category where to attache article (default is CMS08MG)
-not2web	#do not post Article to WEB (keep in MCDB), default is post right after it is described
-verbose	#be verbose during the run
-debug		#print additional information during the session 

     
     The complete set of options is the following: 
";
#[-a login or cert or pkcs12] authorize in MCDB with 
#   AFS login/password or LCG certificate or PKCS12 certificate;
#   default method is LCG certificate. Necessary paths to certificate
#   have to be written in the begining of the script $0
help();

}

__END__

=head1 NAME

Script name - short discription of your program

=head1 SYNOPSIS

 how to us your program

=head1 DESCRIPTION

 long description of your program

=head1 SEE ALSO

 need to know things before somebody uses your program

=head1 AUTHOR

 Lev Dudko

=cut
