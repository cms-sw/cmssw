#! /usr/bin/env perl
use File::Basename;

# get the complete list of class names for replacement:

my @src = ();
my @rep = ();

print <<EOF;

Generally you'll need to create/edit all of these files to install a new CondFormat.

But by putting your classname into:
#modified:   L1Trigger/L1TCommon/scripts/make_records_files.txt
creating your class header file, and running this script, most of this is automated.

The complete list of files you'll need to update for class L1TXXX is:

# Changes to be committed:
#   (use "git reset HEAD <file>..." to unstage)
#
#modified:   CondCore/Utilities/src/CondDBFetch.cc
#modified:   CondCore/Utilities/src/CondDBImport.cc
#modified:   CondCore/Utilities/src/CondFormats.h
#new file:   CondFormats/DataRecord/interface/L1TXXXRcd.h
#new file:   CondFormats/DataRecord/src/L1TXXXRcd.cc
#new file:   CondFormats/L1TObjects/interface/L1TXXX.h
#new file:   CondFormats/L1TObjects/src/L1TXXX.cc
#modified:   CondFormats/L1TObjects/src/classes.h
#modified:   CondFormats/L1TObjects/src/classes_def.xml
#modified:   CondFormats/L1TObjects/test/testSerializationL1TObjects.cpp
#modified:   L1Trigger/L1TCommon/scripts/make_records.pl
EOF

open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    $rcdname     = "${classname}Rcd";
    print "INFO:  class: \"$classname\" --> record:  \"$rcdname\"\n";
    open INPUT, "L1Trigger/L1TCommon/scripts/template.h";
    open OUTPUT, ">CondFormats/DataRecord/interface/${classname}Rcd.h";
    while(<INPUT>){
	s/XXX/$classname/g;
	print OUTPUT $_;
    }
    close INPUT;
    close OUTPUT;


    open INPUT, "L1Trigger/L1TCommon/scripts/template.cc";
    open OUTPUT, ">CondFormats/DataRecord/src/${classname}Rcd.cc";
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
    $file = "CondFormats/L1TObjects/src/T_EventSetup_${classname}.cc";
    open OUTPUT, "$file";
    print OUTPUT "#include \"CondFormats/L1TObjects/interface/${classname}.h\"\n";
    print OUTPUT "#include \"CondFormats/DataRecord/interface/${classname}Rcd.h\"\n";
    print OUTPUT "REGISTER_PLUGIN(${classname}Rcd, ${classname});\n";
    print OUTPUT "\n";
    close OUTPUT;
}
close FILELIST;

print "MANUAL ADDITION NEEDED to CondCore/Utilities/src/CondDBFetch.cc, add following snippet:\n"; 

open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "FETCH_PAYLOAD_CASE( $classname )\n";
}
close FILELIST;
print "\n";

print "MANUAL ADDITION NEEDED to CondCore/Utilities/src/CondDBImport.cc, add following snippet:\n"; 

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
    open OUTPUT, ">CondFormats/L1TObjects/src/${classname}.cc";
    print OUTPUT "#include \"CondFormats/L1TObjects/interface/${classname}.h\"\n";
    close OUTPUT;
}
close FILELIST;

print "MANUAL ADDITION NEEDED to CondCore/Utilities/src/CondFormats.h, add following snippet:\n"; 
open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
while($classname = <FILELIST>){
    next if ($fullfile =~ /^\s*\#/);
    chomp $classname;
    print "#include \"CondFormats/L1TObjects/interface/${classname}.h\"\n";
}
close FILELIST;

print "MANUAL ADDITION NEEDED to CondFormats/L1TObjects/src/classes_def.xml, add following snippet:\n"; 

#open FILELIST, "L1Trigger/L1TCommon/scripts/make_records_files.txt";
#while($classname = <FILELIST>){
#    next if ($fullfile =~ /^\s*\#/);
#    chomp $classname;
#    print "\<class name=\"std::vector\<${classname}>\"/>\n";
#    print "\<class name=\"std::map\<std::string, ${classname}>\"/>\n";
#}
#close FILELIST;



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
