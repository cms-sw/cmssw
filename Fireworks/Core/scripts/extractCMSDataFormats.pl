#!/bin/env perl
# Author:      Dmytro Kovalskyi (UCSB)
#              dr.kovalskyi@gmail.com
use strict;
use warnings;
use Getopt::Long;

die "Usage:\n\t$0 <parameters and options>\n\nLook at the code for details.\n\n" if (@ARGV == 0);

my $required_packages      = "required_packages.txt";
my $dir                    = "core";
my $core_mode              = 0;
my $libs_file              = "libs.txt";
my $core_make_file         = "core.mk";
my $project_make_file      = "project.mk";
my $mode                   = "core";
  
GetOptions( "m|mode=s"     => \$mode,
            "d|dir=s"      => \$dir ) || die ("Abort.\n");

$core_mode = 1 if ( $mode =~ /core/i);

if ( $core_mode ){
    die "Please set CMSSW environment\n" if ( ! defined $ENV{"CMSSW_RELEASE_BASE"} );
    die "Please provide file $required_packages that has a list of packages (one package\n".
      "per line). All dependencies are automatically resolved, so list only what you really\n".
      "need.\n" if (! -e $required_packages);
}

# resolve dependencies and get complete list of required packages, libraries and tools
my %scram_tools = ();  
my %libraries = ();    
my %packages = ();

# processed buildfiles
my %buildfiles = ();

sub get_tool_info{
    my $tool = lc shift;
    if ( $core_mode ){
	system("mkdir -p $dir/tools");
	system("scramv1 tool info $tool > $dir/tools/$tool");
    } else {
	die "Unkown tool $tool. Abort\n" if ( ! -e "$dir/tools/$tool" );
    }
    return split(/\n/,`cat $dir/tools/$tool`);
}

sub analyze_scram_tool{
    my $tool = lc shift;
    my $libs = "";
    my $dirs = "";
    my $includes = "";
    my $tools = "";
    my $paths = "";
    if ( ! defined $scram_tools{$tool} ){
	foreach my $line( get_tool_info($tool) ){
	    $line =~ s/\n//;
	    $libs = $1 if ( $line =~ /^LIB=(.*)$/);
	    $dirs = $1 if ( $line =~ /^LIBDIR=(.*)$/);
	    $includes = $1 if ( $line =~ /^INCLUDE=(.*)$/);
	    $tools = $1 if ( $line =~ /^USE=(.*)$/);
	    $paths = $1 if ( $line =~ /^PATH=(.*)$/);
	}
	$scram_tools{$tool} = { "libs"=>"$libs", "dirs"=>"$dirs", "includes"=>"$includes",
	    "tools"=>"$tools", "paths"=>"$paths" };
    }
    if ( $tools =~ /\S/ ){
	foreach my $newtool( split(/\s+/,$tools ) ){
	    analyze_scram_tool($newtool);
	}
    }
}

sub analyze_dependencies{
    my $nNewDependencies = 0;
    foreach my $package(@_){
	my $buildfile = "$package/BuildFile";
	next if ($buildfiles{$buildfile}); # skip processed BuildFile's
	$packages{$package}++;
	$buildfiles{$buildfile}++;
	$nNewDependencies++;
	# print "processing $buildfile\n";
	my @lines = ();
	if ($core_mode) {
	    open(IN, "$ENV{CMSSW_RELEASE_BASE}/src/$buildfile") || die "Cannot open file $buildfile\n$!\n";
	    while (my $line = <IN>){
		push @lines, $line;
	    }
	    close IN;
	} else {
	    # first check local file
	    my $file = "$dir/src/$buildfile";
	    $file = "$dir/cms/$buildfile" if ( ! -e $file );
	    
	    open(IN, $file) || die "Cannot open file $file\n$!\n";
	    while (my $line = <IN>){
		push @lines, $line;
	    }
	    close IN;
	}	
	my $ignore = 0;
	foreach my $line ( @lines ){
	    $ignore = 1 if ( $line =~ /\<export\>/ );
	    $ignore = 0 if ( $line =~ /\<\/export\>/ );
	    next if ($ignore);
	    analyze_scram_tool($1) if ($line =~ /\<use\s+name\s*=\s*([^\/\>\s]+)\>/);
	    $libraries{$1}++ if ($line =~ /\<lib\s+name\s*=\s*([^\/\>\s]+)\>/);
	    if ($line =~ /\<use\s+name\s*=\s*([^\/\>\s]+\/[^\/\>\s]+)\>/){
		my $newpackage = $1;
		my $file = "$dir/src/$newpackage";
		$file = "$dir/cms/$newpackage" if ( ! -e $file );
		$file = "$ENV{CMSSW_RELEASE_BASE}/src/$newpackage" if ( $core_mode );
		if ( ! -e $file){
		    print "WARNING: package not found $newpackage. Skipped\n";
		    next;
		}
		$packages{$newpackage}++;
	    }
	}

    }
    return $nNewDependencies;
}

my @list = ();
if ( $core_mode ){
    open(pIN, $required_packages) || die "Cannot open file $required_packages\n$!\n";
    while (my $package = <pIN>){
	$package =~ s/\n//;
	next if ($package !~ /\S/);
	push @list, $package;
    }
    close pIN;
} else {
    foreach my $buildfile(`find $dir/src/ -type f -maxdepth 3 -name BuildFile`){
	push @list, $1 if ( $buildfile =~ /([^\/]+\/[^\/]+)\/BuildFile$/);
    }
    if ( @list == 0 ){
	system("touch $dir/$project_make_file");
	exit;
    }
}

my $limit = 100;
analyze_scram_tool("gsl");
my $newEntries = analyze_dependencies(@list);
while ( $limit > 0 && $newEntries){
    $newEntries = analyze_dependencies(keys %packages);
    $limit--;
}
die "Fatal error. Reached limit of iterations for the dependency analysis. Need expert help.\n" if ($limit < 1);
  
print "Number of BuildFile's analyzed: ", scalar keys %buildfiles, "\n";

print "Tools: ", scalar keys %scram_tools, "\n";
my $mk_libs = "";
my $mk_dirs = "";
my $mk_incs = "";
my $mk_text = "";

sub copy_external{
    my ($path,$fullpath) = @_;
    system("mkdir -p $path; cp -r $fullpath/* $path/");
}

my %link_libs = ();
sub add_link_libs{
    my $libs = shift;
    foreach my $lib (split(/\s+/,$libs)){
	$link_libs{$lib}++;
    }
}

my %link_dirs = ();
sub add_link_dirs{
    my $dirs = shift;
    foreach my $dir (split(/\s+/,$dirs)){
	$link_dirs{$dir}++;
    }
}

sub split_external_name{
    my $fullpath = shift;
    if ( my ($path,$name) = ($fullpath =~ /.*?\/external\/(.*?)\/([^\/]+)\/*$/) ){
	return ($path,$name,$fullpath);
    } else {
	return ("","",$fullpath);
    }
}

foreach my $tool ( sort keys %scram_tools ){
#    if ( $tool =~ /gccxml/i && $core_mode){
#	# open tool file and parse it
#	if ( `cat $dir/tools/gccxml | grep GCCXML_BASE` =~ /GCCXML_BASE=(\S+)/ ){
#	    my ($path,$name,$fullpath) = split_external_name($1);
#	    if ( $path ne "" ) {
#		copy_external("$dir/external/$path/$name",$fullpath) if ($core_mode);
#		$mk_text .= "GCCXMLDIR := external/$path/$name\n";
#	    } else {
#		$mk_text .= "GCCXMLDIR := $fullpath\n";
#	    }
#	}
#	next;
#    }
    
    if ($scram_tools{$tool}->{libs} =~ /\S/){
	$mk_libs .= " $scram_tools{$tool}->{libs}";
	if ($scram_tools{$tool}->{dirs} !~ /lcg\/root\//){
	    add_link_libs( $scram_tools{$tool}->{libs} );
	    add_link_dirs( $scram_tools{$tool}->{dirs} );
	}
    }
    
    my ($path,$name,$fullpath) = split_external_name($scram_tools{$tool}->{includes});
    if ( $path ne "" ){
	system("mkdir -p $dir/external/inc/$path/$name; cp -r $fullpath/*  $dir/external/inc/$path/$name/") if ($core_mode);
	$mk_incs .= " external/inc/$path/$name";
    } elsif ( $fullpath !~ /lcg\/root\// && $fullpath =~ /\S/){
	$mk_incs .= " $scram_tools{$tool}->{includes}";
    }
}

# copy and link libraries
if ($core_mode) {
    my %phys_files = ();
    my %linked_files = ();
    foreach my $dir ( keys %link_dirs ){
	foreach my $lib ( keys %link_libs ){
	    foreach my $file(`ls -1 $dir/*$lib*.so* 2>/dev/null`){
		$file =~ s/\n//;
		my $phys_file = `readlink -f $file`;
		$phys_file =~ s/\n//;
		$linked_files{$file}=$phys_file if ( $file ne $phys_file );
		$phys_files{$phys_file}++;
	    }
	}
    }
    foreach my $file ( keys %phys_files ){
	system("mkdir -p $dir/external/lib; cp $file $dir/external/lib/");
    }
    foreach my $link ( keys %linked_files ){
	my $source = $linked_files{$link};
	$source =~ s/^.*?([^\/]+)$/$1/;
	$link =~ s/^.*?([^\/]+)$/$1/;
	system("cd $dir/external/lib; ln -s $source $link");
    }
}

print "Packages: ", scalar keys %packages, "\n";
# foreach my $package ( sort {$packages{$b} <=> $packages{$a}} keys %packages ){
system("mkdir -p $dir/cms/") if ($core_mode);
foreach my $package ( sort keys %packages ){
    # print "$packages{$package} \t $package\n";
    if ( $core_mode ){
	system("mkdir -p $dir/cms/$package");
	system("cp -r $ENV{CMSSW_RELEASE_BASE}/src/$package/* $dir/cms/$package/");
    }
}

print "Libraries: ", scalar keys %libraries, "\n";
foreach my $lib ( sort keys %libraries ){
    # print "$libraries{$lib} \t $lib\n";
    $mk_libs .= " $lib" if ( $mk_libs !~ /$lib/ );
}

if ( $core_mode ) {
    open(OUT, ">$dir/$core_make_file")||die "Cannot write to file $dir/$core_make_file\n$!\n";
    print OUT "CoreLibs := $mk_libs\n";
    print OUT "CoreIncludes := $mk_incs\n";
    print OUT $mk_text;
    close OUT;
} else {
    open(OUT, ">$dir/$project_make_file")||die "Cannot write to file $dir/$project_make_file\n$!\n";
    print OUT "ProjectLibs := $mk_libs\n";
    print OUT "ProjectIncludes := $mk_incs\n";
    close OUT;
}    


exit
