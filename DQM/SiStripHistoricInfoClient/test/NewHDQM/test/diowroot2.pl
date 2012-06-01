#!/usr/bin/env perl 
#----------------------------------------------------------- 
# Name: 
#    DIOW: Digital Images On the Web
#
# Purpose: 
#    scan a directory with jp(e)g and or gif (upper or 
#    lower cases filenames ) images: create icons and 
#    create an index.html file for on line publication 
#    with icons on screen and links to original images. 
#    Useful when you have too many digital images and 
#    you do not have enough time to create web pages 
# 
# Requirements:
#    uses PERL and the convert program: best is running Linux
#
# Optional Keywords:
#   -o : means Output: name of the output HTML filename
#   -t : means Title : title of the web page
#   -c : means columnnumber: number of columns
#   -icon: means iconsize
#   -annotate: means Annotate: string to be written on the lower right corner of each image
#
# Examples:
#    chmod 755 diow.pl
#    ln -s diow.pl diow 
#    diow
#    diow -t 'My Images From Versailles, Sept-2000'
#    diow -o versailles.html
#    diow -o versailles.html -t 'My Images From Versailles, Aug-2000'
#    diow -o versailles.html -t 'My Images From Versailles, Aug-2000' -c 5 -ico 40
#    diow -o versailles.html -t 'My Images From Versailles, Aug-2000' -c 5 -ico 40 -annotate 'Versailles Aug-2000'
#
# Informations:
#    http://mips.as.arizona.edu/~hdole/diow/
#    http://wwwfirback.ias.u-psud.fr/users/dole/diow/
#    Herve Dole
#
# Modification History: 
#    ??-Original program: hj.pl by Matthew Kenworthy, UoA    
#    01-Aug-2000 This version Written by Herve Dole, IAS Orsay
#    02-Aug-2000 add -o and -t HD, IAS
#    11-Aug-2000 add version and array with legend v1.2 HD, IAS
#    11-Aug-2000 add nice keywords processing v1.3 HD, IAS
#    17-Aug-2000 add $iconsize HD, IAS v1.4
#    23-Aug-2000 add -column -ico v1.5 HD, IAS
#    23-Aug-2000 bug corrected with UPPERCASES and sorted files v1.6 HD, IAS 
#    18-Jan-2001 add gif files processing v1.7 HD, UoA
#    03-Sep-2001 add -annotate string option + png files processing v1.8 HD, UoA
#    10-Sep-2001 fixed bug when -annotate not used v1.8.1 HD, UoA
#    16-Oct-2003 XHTML 1.0 tranistionnal complient HD v2.0
#
#----------------------------------------------------------- 
$diow_version = '2.0';

# Welcome message
#----------------
print("--------------------------------------------------------\n"); 
print(" DIOW v$diow_version: Digital Images On the Web (H. Dole)\n"); 
print("--------------------------------------------------------\n"); 

# Users's Data 
#------------- 
$myname= 'CMS';      # my Name
$myemail = 'username\@yourdomain';  # my Email
$myurl = 'http://www.myurl';        # my URL
$title = 'Test';                            # default title of the page
$outhtml = 'index.html';                                 # default output HTML filename
$mytextcolor= '000000';
$mybgcolor = 'FFFFFF';
$mylinkcolor ='0000FF';   # color for unvisited links
$myvlinkcolor = 'FF00FF'; # color for visited link
$barcolor = 'FF0000';     # background color for title bar
$diowurl = 'http://mips.as.arizona.edu/~hdole/diow/'; # URL where you can download DIOW
$nbrow = 6;               # number of rows in the table
#$iconsize = 0;           # size of icons
$iconsize = 200;           # size of icons
$annotate_keyword=0;      # annotate option
$annotate_string='';      # annotate string
$annotate_font = 'helvetica'; # annotate font # helvetica
$annotate_color_font='yellow';# annotate font color
$annotate_size='20-20';    # annotate Font Size
$annotate_color_box='blue';# annotate box color
#DD NB: convert is not available on vocms01... so this is my workaround
#(convert is compiled... so this doesn't work after all)
#$mydir=`pwd`;
#@dir = split(/fig/,$mydir);
#$convert = $dir[0].'test/convert';
#$cmd  = 'test/convert';
#$convert=$mydir.$cmd;
#print $convert;
my $workDir='.';
my $outDir='.';
#my $outDir='./fig/Prompt/Run2012A/MinimumBias/';
my $outDir='fig/Prompt/Run2012A/MinimumBias/';
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
# -annotate 'My Name': string: annotation in the Images
#------------------------------------------------------
  elsif ($ARGV[$i] =~ /^-anno/) {$annotate_string=$ARGV[$i +1];
			   $annotate_keyword=1; }
  elsif ($ARGV[$i] =~ /^-D/) {$workDir=$ARGV[$i +1];
                            }
 elsif ($ARGV[$i] =~ /^-OD/) {$outDir=$ARGV[$i +1];}
}
print ("WORK: $workDir ");
# Print Some Argumenst to Check
print("  title : $title \n");
print("  output: $outhtml \n"); 
print("  nb rows: $nbrow \n"); 
print("  icon size: $iconsize \n"); 
if ($annotate_keyword==1) {
print("  annotate: $annotate_string \n");
print("  ann. font size: $annotate_size \n");
}
print("-------------------------\n"); 

#first get a listing of the current directory, warts and all 

opendir THISDIR, $workDir or die "Whoa! Current directory cannot be opened.."; 
 
# the regexp looks for either .jpg or .jpeg at the end of a filename. 
# () means group it together, \. is an escaped period, e? means 0 or 1  
# occurences of the letter e and $ means look for it at the end of the  
# filename 
# the i appended after the slash means ignore the case. 
# Look for jpg and gif and png files
#@allfiles_raw1 = grep /(\.jpe?g)$/i, readdir THISDIR; Only jpg
#@allfiles_raw = grep /(\.jpe?g)$/i||/(\.gif)$/i, readdir THISDIR; Only jpg+gif
@allfiles_raw = grep /(\.jpe?g)$/i||/(\.gif)$/i||/(\.png)$/i, readdir THISDIR;
closedir THISDIR; 

opendir THISDIR,  $workDir  or die "Whoa! Current directory cannot be opened.."; 
@rootfiles_raw = grep /(\.root)$/i, readdir THISDIR;
closedir THISDIR; 

# Sort files
#-----------
@allfiles = sort @allfiles_raw ;
@rootfiles = sort @rootfiles_raw ;

open(HTMLFILE,">$outhtml") or die "Can't open $outhtml for writing"; 

print HTMLFILE "<?xml version=\"1.0\" encoding=\"iso-8859-1\" ?>"; 
print HTMLFILE "<!DOCTYPE html "; 
print HTMLFILE "     PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\""; 
print HTMLFILE "     \"DTD/xhtml1-transitional.dtd\">"; 
print HTMLFILE "<html>\n"; 
print HTMLFILE "<head>\n"; 
print HTMLFILE "<title>$title</title>\n"; 

print HTMLFILE "<script type=\"text/javascript\"> \n";
$cc=0;
$ALL_FIGS=" var presetImagesTrk=[\"";
foreach $jpegfile (@allfiles) {
        if ($cc==0){$ALL_FIGS=$ALL_FIGS.$jpegfile."\"";}
        else {$ALL_FIGS=$ALL_FIGS.",\"".$jpegfile."\"";}
        $cc=11;
}
$ALL_FIGS=$ALL_FIGS."];";
print HTMLFILE "$ALL_FIGS\n";
print HTMLFILE "function nav(){ var u = document.reco.mylist.selectedIndex; \n";
print HTMLFILE " var w = document.epoch.mylist.selectedIndex; \n";
print HTMLFILE " var z = document.dataset.mylist.selectedIndex; \n";
print HTMLFILE "if(u==0) return; \n";
print HTMLFILE "if(w==0) return; \n";
print HTMLFILE "if(z==0) return; \n";

print HTMLFILE " var recO=document.reco.mylist.options[u].value;\n";
print HTMLFILE "var epocH = document.epoch.mylist.options[w].value;\n";
print HTMLFILE "var dataseT = document.dataset.mylist.options[z].value;\n";
print HTMLFILE " var nove = \'\';var poz=\'\'; var poz1=\'\'; \n";
print HTMLFILE "for (i = 0; i < presetImagesTrk.length; i\++) {\n";
print HTMLFILE "var ii=i\+1;var picID=\'images\'+ii;var picID1=\'Images\'+ii;\n";
print HTMLFILE "document.getElementById(picID).src=\'\';\n";
print HTMLFILE "document.getElementById(picID1).href=\'\';\n";
#print HTMLFILE "poz=\'./fig/\'+ recO + \'/\'+epocH + \'/\'+dataseT + \'/\'+$title + \'/\' + presetImagesTrk[i];\n";
#print HTMLFILE "poz=\'./fig/\'+ recO + \'/\'+epocH + \'/\'+dataseT + \'/$title/\' + presetImagesTrk[i];\n";
print HTMLFILE "poz=\'fig/\'+ recO + \'/\'+epocH + \'/\'+dataseT + \'/$title/\' + presetImagesTrk[i];\n";
print HTMLFILE "document.getElementById(picID).src=poz;\n";
print HTMLFILE "document.getElementById(picID1).href=poz; }}\n";
print HTMLFILE " </script>\n";

print HTMLFILE "</head>\n"; 


########### END OF HEAD DEFINITIONS
print HTMLFILE "<body text=\"#$mytextcolor\" bgcolor=\"#$mybgcolor\" link=\"#$mylinkcolor\" vlink=\"#$myvlinkcolor\">\n"; 
print HTMLFILE "<form name=\"reco\">\n";
print HTMLFILE "Jump to:\n";
print HTMLFILE "<select name=\"mylist\" onchange=\"nav()\">\n";
print HTMLFILE "<option value=\"-1\">Choose a Reco</option>\n";
print HTMLFILE "<option value=\"Prompt\">Prompt</option>\n";
print HTMLFILE "<option value=\"2012\">2012</option>\n";
print HTMLFILE "</select>\n";
print HTMLFILE "</form>\n";


print HTMLFILE "<form name=\"epoch\">\n";
print HTMLFILE "Jump to :\n";
print HTMLFILE "<select name=\"mylist\" onchange=\"nav()\">\n";
print HTMLFILE "<option value=\"-1\">Choose an Epoch</option>\n";
#print HTMLFILE "<option value=\"Collisions/JSON/MinimumBias\">JSON: Minimum Bias 2011</option>\n";
#print HTMLFILE "<option value=\"fig/Prompt/Run2012/MinimumBias\">JSON: Minimum Bias 2012</option>\n";
print HTMLFILE "<option value=\"Run2012\">Run2012</option>\n";
print HTMLFILE "<option value=\"Run2012A\">Run2012A</option>\n";
print HTMLFILE "<option value=\"Run2012B\">Run2012B</option>\n";
print HTMLFILE "</select>\n";
print HTMLFILE "</form>\n";


print HTMLFILE "<form name=\"dataset\">\n";
print HTMLFILE "Jump to :\n";
print HTMLFILE "<select name=\"mylist\" onchange=\"nav()\">\n";
print HTMLFILE "<option value=\"-1\">Choose a folder</option>\n";
print HTMLFILE "<option value=\"MinimumBias\">MinimumBias</option>\n";
print HTMLFILE "<option value=\"Collisions/JSON/MinimumBias\">JSON: Minimum Bias 2011</option>\n";
print HTMLFILE "<option value=\"fig/Prompt/Run2012/MinimumBias\">JSON: Minimum Bias 2012</option>\n";
print HTMLFILE "</select>\n"; 
print HTMLFILE "</form>\n";


###########setup links to other  plots
print HTMLFILE "Expert plots for:\n";
print HTMLFILE "<a href=\"index_pixel.html\">Pixel</a>\n";
print HTMLFILE "<a href=\"index_strip.html\">Strip</a>\n";
print HTMLFILE "<a href=\"index_tracker.html\">Tracking</a>\n";
print HTMLFILE "<hr>\n";
##############


print HTMLFILE "<table align=\"center\"><tbody>\n"; 
print HTMLFILE "<tr>\n"; 
print HTMLFILE "<td align=\"center\" width=\"80%\" bgcolor=\"#$barcolor\">\n";  
print HTMLFILE "<font color=\"#FFFFFF\" size=\"+1\" face=\"Helvetica\"> <b> $title </b> </font><br />\n";  
print HTMLFILE "</td>\n"; 
print HTMLFILE "</tr>\n"; 
print HTMLFILE "</tbody>\n"; 
print HTMLFILE "</table>\n"; 
print HTMLFILE "<br />\n"; 
print HTMLFILE "<br />\n"; 

# Create Table of Images
#-----------------------
print HTMLFILE " \n"; 
print HTMLFILE " \n"; 
print HTMLFILE "<table align=\"center\" border=\"1\" cellpadding=\"5\" cellspacing=\"2\">\n"; 
# Loop on images
#---------------
$count = 0;
$prefix = ' ';
$suffix = ' ';
$image_count=0;
foreach $jpegfile (@allfiles) { 
$image_count++;
# select jpg or jpeg file 
	print("Working on... $jpegfile\n"); 
# Annotate Option: Create FineName
#---------------------------------
	$annotatejpegfile = $jpegfile;
	#$rootfile = $jpegfile;
	#$rootfile =~ s/.png/.root/;
# new name for lowercase filenames jpg
	$annotatejpegfile =~ s/.jpe?g/_ann.jpg/; 
# new name for uppercase filenames jpg
	$annotatejpegfile =~ s/.JPE?G/_ann.jpg/; 
# new name for lowercase filenames png
	$annotatejpegfile =~ s/.png/_ann.png/; 
# new name for uppercase filenames png
	$annotatejpegfile =~ s/.PNG/_ann.png/; 
# new name for lowercase filenames gif
	$annotatejpegfile =~ s/.gif/_ann.gif/; 
# new name for uppercase filenames gif
	$annotatejpegfile =~ s/.GIF/_ann.gif/; 
# Annotate Processing
	if ($annotate_keyword == 1) {
#	  print ("$annotate_string \n");
#	  system("convert -gravity SouthEast -font '-*-$annotate_font-*-*-*--$annotate_size-*-*-*-*-iso8859-1' -fill $annotate_color_font -box $annotate_color_box -draw \'text 10,30 \"$annotate_string\"\' $jpegfile $annotatejpegfile"); # beware: 20 (in text argument) must be >~ to $annotate_size+10
	  system("convert -gravity SouthEast -font '-*-$annotate_font-*-*-*--$annotate_size-*-*-*-*-iso8859-1' -fill $annotate_color_font -box $annotate_color_box -draw \'text 10,30 \"$annotate_string\"\' $jpegfile $annotatejpegfile"); # beware: 20 (in text argument) must be >~ to $annotate_size
	} else {
	  $annotatejpegfile = $jpegfile
	}

# xlsfonts -fn '*-0-0-0-0-*' to check the available fonts on your system

# create name for icon file
#--------------------------
	$smallfile = $annotatejpegfile; 
# new name for lowercase filenames jpg
	$smallfile =~ s/.jpe?g/_small.jpg/; 
# new name for uppercase filenames jpg
	$smallfile =~ s/.JPE?G/_small.jpg/; 
# new name for lowercase filenames png
	$smallfile =~ s/.png/_small.png/; 
# new name for uppercase filenames png
	$smallfile =~ s/.PNG/_small.png/; 
# new name for lowercase filenames gif
	$smallfile =~ s/.gif/_small.gif/; 
# new name for uppercase filenames gif
	$smallfile =~ s/.GIF/_small.gif/; 
# Process Icon
#-------------
	#system("convert -geometry x$iconsize $annotatejpegfile $smallfile"); 
# insert in the HTML code the image and its icon
#-----------------------------------------------
	$prefix = ' ';
	$suffix = ' ';
	print "AAA:",$count , $nbrow;
	$nbrow=3;
	if ($count == $nbrow-1 ) {
	  $prefix = ' ';
	  $suffix = '</tr>';
	  $count = -1;
	}
	if ($count == 0 ) {
	  $prefix = '<tr>';
	  $suffix = ' ';
	}
	$string = "$prefix <td align=\"center\"> <a id=\"Images".$image_count."\" href=\"" .$outDir. $title."/". $annotatejpegfile . "\"><img id=\"images".$image_count."\" src=\"". $outDir. $title."/" . $annotatejpegfile . "\" hspace=\"5\" vspace=\"5\" border=\"0\" alt=\"$jpegfile\"  width=\"400\" />  </a> \n  <br /> $annotatejpegfile </td> $suffix \n"; 
	print HTMLFILE $string;
	#$string1 
	$count +=  1;
    }


# End HTML file
#--------------
if ($count != 0 ) {
  print HTMLFILE "</tr>\n";
}
print HTMLFILE "</table>\n";

print(" root files now.\n");
 
print HTMLFILE "<br />";


foreach $root (@rootfiles) {
    print("Working on... $root\n"); 
    print HTMLFILE "<a href=\"$root\">$root </a> <br />"; 
} 

# End table
#----------
print HTMLFILE "<hr />\n"; 
print HTMLFILE "<table border=\"0\" cellspacing=\"0\" cellpadding=\"0\" width=\"100\%\">\n";
 
# Add informations: Date and DIOW URL
#------------------------------------
print HTMLFILE "<tr><td><em>\n"; 
$d=`date +'%a %d-%b-%Y %H:%M'`; 
print HTMLFILE "Created: $d \n"; 
print HTMLFILE "</em></td><td align=\"right\"><em>\n"; 
# Add informations: my URL
#-------------------------
print HTMLFILE "<a href=\"$myurl\">$myname 's page</a>\n"; 
print HTMLFILE "</em></td></tr>\n"; 
# Add informations: my Email
#---------------------------
print HTMLFILE "<tr><td align=\"left\"><em>\n";
print HTMLFILE "using <a href=\"$diowurl\">DIOW $diow_version</a>, the \"Digital Images On the Web\" PERL script\n ";
print HTMLFILE "</em></td><td align=\"right\"><em>\n"; 
print HTMLFILE "<a href=\"mailto:$myemail\">email</a>\n";  
print HTMLFILE "</em></td></tr>\n"; 
print HTMLFILE "</table>\n"; 
# Add informations: W3
#---------------------
print HTMLFILE "\n"; 
print HTMLFILE "<p align=\"center\">\n"; 
print HTMLFILE "<a href=\"http://validator.w3.org/check/referer\">\n"; 
print HTMLFILE "\"Valid XHTML 1.0\!\"\n"; 
print HTMLFILE "</a>\n"; 
print HTMLFILE "</p>\n"; 

print HTMLFILE "</body>\n"; 
print HTMLFILE "</html>\n"; 

close(HTMLFILE) or die "Can't close $outhtml. Sorry."; 

# Bye message
#------------
print(" DIOW: run OK. Bye.\n"); 
print("--------------------------------------------------------\n"); 

