#! /usr/bin/env perl
use File::Basename;

# get the complete list of class names for replacement:

my @src = ();
my @rep = ();


open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    $rcdname     = "${classname}Rcd";
    print "INFO:  class: \"$classname\" --> record:  \"$rcdname\"\n";
    open INPUT, "template.h";
    open OUTPUT, ">CondFormats/DataFormats/interface/${classname}Rcd.h";
    while(<INPUT>){
	s/XXX/$classname/g;
	print OUTPUT $_;
    }
    close INPUT;
    close OUTPUT;


    open INPUT, "template.cc";
    open OUTPUT, ">CondFormats/DataFormats/src/${classname}Rcd.cc";
    while(<INPUT>){
	s/XXX/$classname/g;
	print OUTPUT $_;
    }
    close INPUT;
    close OUTPUT;
}
close FILELIST;

open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "#include \"CondFormats/L1TObjects/interface/${classname}.h\"\n";
    print "#include \"CondFormats/DataRecord/interface/${classname}Rcd.h\"\n";
    print "REGISTER_PLUGIN(${classname}Rcd, ${classname});\n";
    print "\n";
}
close FILELIST;


open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "FETCH_PAYLOAD_CASE( $classname )\n";
}
close FILELIST;
print "\n";

open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "IMPORT_PAYLOAD_CASE( $classname )\n";
}
close FILELIST;
print "\n";

open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "#include \"CondFormats/L1TObjects/interface/${classname}.h\"\n";
}
close FILELIST;

open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "\<class name=\"std::vector\<${classname}>\"/>\n";
    print "\<class name=\"std::map\<std::string, ${classname}>\"/>\n";
}
close FILELIST;

open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "\<class name=\"${classname}\">\n";
    open CLASSDEF, "CondFormats/L1TObjects/interface/${classname}.h";
    while (<CLASSDEF>){
	#print $_;
	if (/std::(vector|map).* (\S+)\;/){
	    next if ($2 eq "\}");
	    #print "--------------------------------------------------\n";
	    print "  \<field name=\"$2\" mapping=\"blob\" />\n";
	    #print "--------------------------------------------------\n";

	}

    }
    print "\</class>\n";
}
close FILELIST;
