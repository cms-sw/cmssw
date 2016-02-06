#! /usr/bin/env perl
use File::Basename;

# get the complete list of class names for replacement:

my @src = ();
my @rep = ();

open FILELIST, "L1Trigger/L1TGlobal/scripts/files.txt";
while($fullfile = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $fullfile;
    #print "INFO: processing file:  $fullfile\n";
    $file = basename($fullfile);
    #print "INFO: basename: $file\n";
    if ($file =~ /^es(\S+)\.hh/) {
	$utmclass   = "es$1";
	$cmsclass = "L1TUtm$1";
	print "INFO:  \"$utmclass\" --> cms:  \"$cmsclass\"\n";
	push @src, "$utmclass\.hh";
	push @rep, "$cmsclass.h";
	push @src, $utmclass;
	push @rep, $cmsclass;

	open OUTFILE, ">CondFormats/L1TObjects/src/$cmsclass.cc";
	print OUTFILE "// auto-generated file by import_utm.pl \n";
	print OUTFILE "#include \"CondFormats/L1TObjects/interface/$cmsclass.h\"\n";
	close OUTFILE;
	
	open OUTFILE, ">CondFormats/L1TObjects/src/T_EventSetup_$cmsclass.cc";
	print OUTFILE "// auto-generated file by import_utm.pl \n";
	print OUTFILE "#include \"CondFormats/L1TObjects/interface/$cmsclass.h\"\n";
	print OUTFILE "#include \"FWCore/Utilities/interface/typelookup.h\"\n\n";
	print OUTFILE "TYPELOOKUP_DATA_REG($cmsclass);\n";
	close OUTFILE;

	open OUTFILE, ">CondFormats/DataRecord/interface/${cmsclass}Rcd.h";
	print OUTFILE "// auto-generated file by import_utm.pl \n";
	print OUTFILE "\#ifndef CondFormatsDataRecord_${cmsclass}Rcd_h\n";
	print OUTFILE "\#define CondFormatsDataRecord_${cmsclass}Rcd_h\n";
	print OUTFILE "#include \"FWCore/Framework/interface/EventSetupRecordImplementation.h\"\n";
	print OUTFILE "class ${cmsclass}Rcd : public edm::eventsetup::EventSetupRecordImplementation<${cmsclass}Rcd> {};\n";
	print OUTFILE "\#endif\n";
	close OUTFILE;

	open OUTFILE, ">CondFormats/DataRecord/src/${cmsclass}Rcd.cc";
	print OUTFILE "// auto-generated file by import_utm.pl \n";
	print OUTFILE "\#include \"FWCore/Framework/interface/eventsetuprecord_registration_macro.h\"\n";
	print OUTFILE "\#include \"CondFormats/DataRecord/interface/${cmsclass}Rcd.h\"\n";
	print OUTFILE "EVENTSETUP_RECORD_REG(${cmsclass}Rcd);\n";
	close OUTFILE;
    } else {
	print "ERROR: unable to guess CMSSW from $fullfile\n";
	exit(0);
    }
}
close FILELIST;

open REPLACELIST, "L1Trigger/L1TGlobal/scripts/replace.txt";
while(<REPLACELIST>){    
    next if (/^\s*\#/);
    #print $_;
    if(/\"(.+)\"\s*\,\s*\"(.+)\"/){
	$a = $1;
	$b = $2;
	print "INFO:  \"$a\" -> \"$b\"\n";
	push @src, $a;
	push @rep, $b;
    }            
}
close REPLACELIST;

print "@src\n";
print "@rep\n";

open FILELIST, "L1Trigger/L1TGlobal/scripts/files.txt";
while($fullfile = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $fullfile;
    print "INFO: processing file:  $fullfile\n";
    $file = basename($fullfile);
    #print "INFO: basename: $file\n";
    if ($file =~ /^es(\S+)\.hh/) {
	$cmsswfile = "CondFormats/L1TObjects/interface/L1TUtm$1.h";
    } else {
	print "ERROR: unable to guess CMSSW from $fullfile\n";
	exit(0);
    }
    print "INFO: CMSSW filename is: $cmsswfile\n";

    open FILENAME, $fullfile;
    open OUTFILE, ">$cmsswfile";


    print OUTFILE "//\n";
    print OUTFILE "// NOTE:  This file was automatically generated from UTM library via import_utm.pl\n"; 
    print OUTFILE "// DIRECT EDITS MIGHT BE LOST.\n";
    print OUTFILE "//\n";

    while (<FILENAME>){
	if (/esTypes/){
	    print "INFO: dropping unneeded include:  $_\n";
	} elsif (/namespace tmeventsetup/){
	    if (! /\}/){
		$_ = <FILENAME>;
	    }
	    print "INFO: dropping namespace.\n";

	} elsif (/^\s*\}\;/){
	    print "INFO: found end of class definition.\n";
	    print OUTFILE "  COND_SERIALIZABLE;\n";
	    print OUTFILE "$_";
	} elsif (/\* headers/){
	    print "INFO: found start of header files\n";
	    print OUTFILE $_;
	    $_ = <FILENAME>;
	    print OUTFILE $_;
	    $_ = <FILENAME>;
	    while(/\#include\s*\</){
		print OUTFILE $_;
		$_ = <FILENAME>;
	    }
	    print OUTFILE "#include \"CondFormats/Serialization/interface/Serializable.h\"\n";
	    print OUTFILE $_;

	} elsif (/\#if defined\(SWIG\)/){
	    print "INFO: dropping SWIG blog.\n";
	    until(/endif/){
		#print "DEBUG:  $_";
		$_ = <FILENAME>;
	    }
	} else {
	    # strip out utm namespace
	    s/tmeventsetup\:\://g;

	    for ($i=0; $i<=$#src; $i++){
		$a = $src[$i];
		$b = $rep[$i];
		#print "INFO: checking for $a --> $b\n";
		s/\b$a\b/$b/g;
		s/\_$a\_/\_$b\_/g;
	    }
	    print OUTFILE $_;
	}
    }
    close FILENAME;
    close OUTFILE;
}
