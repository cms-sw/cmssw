#!/usr/bin/env perl
# taken from Gaelle Boudoul
 
#----------------------------------------------------------- 
$diow_version = '1.6';

# Welcome message
#----------------
print("--------------------------------------------------------\n"); 
print(" DIOW v$diow_version: Digital Images On the Web (H. Dole)\n"); 
print("--------------------------------------------------------\n"); 

# Users's Data 
#------------- 
$myname= 'TrackingHDQMGroup';      # my Name
$myemail = 'dhidas@cern';  # my Email
#$myurl = 'http://lpsc.in2p3.fr/ams/gaelle/gaelle.html';    
$title = 'Historic DQM Tracking trends';
$outhtml = 'index.html';                                 # default output 
$mytextcolor= '000066';
$mybgcolor = 'CCCCFF';
#$mybgimage = '../images/8981.jpg';
$mylinkcolor ='FFFF00';   # color for unvisited links
$myvlinkcolor = 'FF00FF'; # color for visited link
$barcolor = '9999FF';     # background color for title bar
$diowurl = 'http://mips.as.arizona.edu/~hdole/diow/'; # URL where you can download DIOW
$nbrow = 3;               # number of rows in the table
$iconsize = 200;           # size of icons

# test if there are keywords
#---------------------------
#print("  $ARGV[0] \n");

for ($i=0; $i <= $#ARGV; $i++){
# -t 'title of the HTML page'
#----------------------------
  if ($ARGV[$i] =~ /^-t/) {$title = $ARGV[$i +1]; }
# -o output.html
#---------------
  elsif ($ARGV[$i] =~ /^-o/) {$outhtml=$ARGV[$i +1]; }
# -c 5: integer: number of rows
#------------------------------
  elsif ($ARGV[$i] =~ /^-c/) {$nbrow=$ARGV[$i +1]; }
# -icon 70: integer: size of icons
#------------------------------
  elsif ($ARGV[$i] =~ /^-icon/) {$iconsize=$ARGV[$i +1]; }
}

print("  title : $title \n");
print("  output: $outhtml \n"); 
print("-------------------------\n"); 

#first get a listing of the current directory, warts and all 
opendir THISDIR, "." or die "Whoa! Current directory cannot be opened.."; 
 
# the regexp looks for either .jpg or .jpeg at the end of a filename. 
# () means group it together, \. is an escaped period, e? means 0 or 1  
# occurences of the letter e and $ means look for it at the end of the  
# filename 
# the i appended after the slash means ignore the case. 
# Look for jpg and gif files
#@allfiles_raw = grep /(\.jpe?g)$/i, readdir THISDIR; ; Only jpg
@allfiles_raw = grep /(\.jpe?g)$/i||/(\.gif)$/i, readdir THISDIR; 
closedir THISDIR; 

# Sort files
#-----------
@allfiles = sort @allfiles_raw ;
 
open(HTMLFILE,">$outhtml") or die "Can't open $outhtml for writing"; 
print HTMLFILE "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"; 
print HTMLFILE "<html>\n"; 
print HTMLFILE "<head>\n"; 
print HTMLFILE "<title>$title: $myname 's Images</title>\n"; 
print HTMLFILE "</head>\n"; 
print HTMLFILE "<BODY TEXT=\"#$mytextcolor\" BGCOLOR=\"#$mybgcolor\"  LINK=\"#$mylinkcolor\" VLINK=\"#$myvlinkcolor\" >\n"; 

print HTMLFILE "<P>\n"; 
print HTMLFILE "<TABLE WIDTH=\"100%\">\n"; 
print HTMLFILE "<TR>\n"; 
print HTMLFILE "<TH ALIGN=\"center\" width=\"40%\" BGCOLOR=\"#$barcolor\"><FONT 
COLOR=\"#FFFFFF\" SIZE=+1> $title </FONT></TH>\n";  
print HTMLFILE "</TABLE>\n"; 
print HTMLFILE "</p>\n"; 

# Create Table of Images
#-----------------------
print HTMLFILE " \n"; 
print HTMLFILE " \n"; 
print HTMLFILE " <center>\n"; 
print HTMLFILE "<TABLE border=0 cellpadding=5 cellspacing=2>\n"; 

# Loop on images
#---------------
$count = 0;
$prefix = ' ';
$suffix = ' ';
foreach $jpegfile (@allfiles) { 
# select jpg or jpeg file and creat name for icon file
	print("Working on... $jpegfile\n"); 
	$smallfile = $jpegfile; 
# new name for lowercase filenames
	$smallfile =~ s/.jpe?g/_small.jpg/; 
# new name for uppercase filenames
	$smallfile =~ s/.JPE?G/_small.jpg/; 
# new name for lowercase filenames gif
	$smallfile =~ s/.gif/_small.gif/; 
# new name for uppercase filenames gif
	$smallfile =~ s/.GIF/_small.gif/; 
# create icon file
	system("convert -geometry x$iconsize $jpegfile $smallfile"); 
# insert in the HTML code the image and its icon
	$prefix = ' ';
	$suffix = ' ';
	if ($count == $nbrow-1 ) {
	  $prefix = ' ';
	  $suffix = '</TR>';
	  $count = -1;
	}
	if ($count == 0 ) {
	  $prefix = '<TR>';
	  $suffix = ' ';
	}
	$string = "$prefix <TD align=center> <a href=\"" . $jpegfile . "\"><img src=\"" . $smallfile . "\"hspace=5 vspace=5 border=0 ALT=\"$jpegfile\"></a> \n  <br> $jpegfile </TD> $suffix \n"; 
	print HTMLFILE $string;
	$count +=  1;
	} 

# End HTML file
#--------------
print HTMLFILE "</TABLE>\n";
print HTMLFILE " </center>\n"; 
print HTMLFILE "<p>\n"; 
# End table
#----------
print HTMLFILE "<hr width=\"100%\">\n"; 
print HTMLFILE "<table border=0 cellspacing=0 cellpadding=0 width=\"100\%\">\n"; 
# Add informations: Date and DIOW URL
#------------------------------------
print HTMLFILE "<tr><td><em>\n"; 
$d=`date +'%a %d-%b-%Y %H:%M'`; 
print HTMLFILE "Created: $d \n"; 
print HTMLFILE "</em></td><td align=right><em>\n"; 
print HTMLFILE "<tr><td align=left><em>\n";
print HTMLFILE "using <a href=\"$diowurl\">DIOW $diow_version</a>, the \"Digital Images On the Web\" PERL script\n ";
print HTMLFILE "</em></td></tr>\n"; 

print HTMLFILE "</table>\n"; 
print HTMLFILE "</body>\n"; 
print HTMLFILE "</html>\n"; 

close(HTMLFILE) or die "Can't close $outhtml. Sorry."; 

# Bye message
#------------
print(" DIOW: run OK. Bye.\n"); 
print("--------------------------------------------------------\n"); 

