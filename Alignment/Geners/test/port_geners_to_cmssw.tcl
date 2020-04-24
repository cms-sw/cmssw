#!/bin/sh
# The next line restarts using tclsh \
exec tclsh "$0" ${1+"$@"}

#
# This script ports the "geners" generic binary I/O
# package to CMSSW. The porting consists in
#
# 1. Changing various "#include" CPP definitions so that
#    they pick up the files from the corresponding CMSSW
#    location.
#
# 2. Changing all exceptions so that they inherit from
#    cms::Exception instead of std::exception.
#
# 3. Replacing LOKI_STATIC_CHECK statements by static_assert.
#
# You only need to modify the three main variables defined below,
# the rest will be done by the script.
#
# After porting, check the contents of "IOException.hh" file.
# The current procedure simply hardwires certain line numbers
# to replace in that file, and might fail in the future in case
# file contents are changed.
#
set inputdir "/afs/cern.ch/user/i/igv/local/src/geners-1.3.0/geners"
set dest_package "Alignment/Geners"
set packagedir "/afs/cern.ch/user/i/igv/CMSSW_7_1_0_pre4/src/Alignment/Geners"

# Create the map for changing include statements
set includemap [list "\#include \"geners/static_check.h\"" {}]
lappend includemap LOKI_STATIC_CHECK static_assert
lappend includemap "\#include \"geners/" "\#include \"$dest_package/interface/"

# Take care of exceptions so that they comply with CMSSW guidelines
lappend includemap std::length_error gs::IOLengthError
lappend includemap std::out_of_range gs::IOOutOfRange
lappend includemap std::invalid_argument gs::IOInvalidArgument
lappend includemap "\#include <stdexcept>" "\#include \"$dest_package/interface/IOException.hh\""

proc file_contents {filename} {
    set chan [open $filename "r"]
    set contents [read $chan [file size $filename]]
    close $chan
    set contents
}

proc filemap {infile outfile map} {
    set in_data [file_contents $infile]
    set chan [open $outfile "w"]
    puts -nonewline $chan [string map $map $in_data]
    close $chan
}

# Procedures for reinserting .icc files
proc is_icc_line {line} {
    set trline [string trim $line]
    if {[string first "\#include" $trline] != 0} {
        return 0
    }
    if {[string compare [string range $trline end-4 end] ".icc\""]} {
        return 0
    }
    return 1
}

proc icc_contents {icc_line icc_dir} {
    set trline [string trim $icc_line]
    set fname [string trim [lindex [split $trline /] end] "\""]
    file_contents [file join $icc_dir $fname]
}

proc insert_icc {infile outfile icc_dir} {
    set output [list]
    foreach line [split [file_contents $infile] "\n"] {
        if {[is_icc_line $line]} {
            lappend output [icc_contents $line $icc_dir]
        } else {
            lappend output $line
        }
    }
    set chan [open $outfile "w"]
    puts $chan [join $output "\n"]
    close $chan
}

# Other useful procedures
proc replace_lines {infile outfile from to replacement} {
    set output [list]
    set linenum 1
    foreach line [split [file_contents $infile] "\n"] {
        set skip 0
        set replace 0
        if {$linenum >= $from && $linenum <= $to} {
            set skip 1
        }
        if {$skip} {
            if {$linenum == $to} {
                set replace 1
            }
        }
        if {$replace} {
            lappend output $replacement
        }
        if {!$skip} {
            lappend output $line
        }
        incr linenum
    }
    set chan [open $outfile "w"]
    puts $chan [join $output "\n"]
    close $chan
}

proc tempfile {dir} {
    set chars "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    set nrand_chars 10
    set maxtries 10
    set access [list RDWR CREAT EXCL TRUNC]
    set permission 0600
    set channel ""
    set checked_dir_writable 0
    set mypid [pid]
    for {set i 0} {$i < $maxtries} {incr i} {
	set newname ""
	for {set j 0} {$j < $nrand_chars} {incr j} {
	    append newname [string index $chars \
		    [expr ([clock clicks] ^ $mypid) % 62]]
	}
	set newname [file join $dir $newname]
	if {[file exists $newname]} {
	    after 1
	} else {
	    if {[catch {open $newname $access $permission} channel]} {
		if {!$checked_dir_writable} {
		    set dirname [file dirname $newname]
		    if {![file writable $dirname]} {
			error "Directory $dirname is not writable"
		    }
		    set checked_dir_writable 1
		}
	    } else {
		# Success
                close $channel
		return $newname
	    }
	}
    }
    if {[string compare $channel ""]} {
	error "Failed to open a temporary file: $channel"
    } else {
	error "Failed to find an unused temporary file name"
    }
}

proc process_static_assert {lines} {
    set code [join $lines "\n"]
    set comma [string first "," $code]
    set end [string first ")" $code $comma]
    set statement [string range $code [expr {$comma+1}] [expr {$end-1}]]
    set newstatement [string map {_ { }} [string trim $statement]]
    set newtext "[string range $code 0 $comma] \"$newstatement\");"
}

proc fix_static_assert {infile outfile} {
    set in_assert 0
    set output [list]
    set assert_lines [list]
    foreach line [split [file_contents $infile] "\n"] {
        if {!$in_assert} {
            if {[string first "static_assert" $line] >= 0} {
                set in_assert 1
            }
        }
        if {$in_assert} {
            lappend assert_lines $line
            if {[string first ";" $line] >= 0} {
                lappend output [process_static_assert $assert_lines]
                set assert_lines [list]
                set in_assert 0
            }        
        } else {
            lappend output $line
        }
    }
    set chan [open $outfile "w"]
    puts -nonewline $chan [join $output "\n"]
    close $chan
}

# Now, do the replacements
foreach hh [glob "$inputdir/*.hh"] {
    set tempfile [tempfile /tmp]
    set outfile [file join "$packagedir/interface" [file tail $hh]]
    insert_icc $hh $tempfile $inputdir
    filemap $tempfile $outfile $includemap
    file delete $tempfile
    fix_static_assert $outfile $outfile
}

foreach cc [glob "$inputdir/*.cc"] {
    set outfile [file join "$packagedir/src" [file tail $cc]]
    filemap $cc $outfile $includemap
}

# Update the IOException.hh file
set update {
#include "FWCore/Utilities/interface/Exception.h"

namespace gs {
    /** Base class for the exceptions specific to the Geners I/O library */
    struct IOException : public cms::Exception
    {
        inline IOException() : cms::Exception("gs::IOException") {}

        inline explicit IOException(const std::string& description)
            : cms::Exception(description) {}

        inline explicit IOException(const char* description)
            : cms::Exception(description) {}

        virtual ~IOException() throw() {}
    };

    struct IOLengthError : public IOException
    {
        inline IOLengthError() : IOException("gs::IOLengthError") {}

        inline explicit IOLengthError(const std::string& description)
            : IOException(description) {}

        virtual ~IOLengthError() throw() {}
    };

    struct IOOutOfRange : public IOException
    {
        inline IOOutOfRange() : IOException("gs::IOOutOfRange") {}

        inline explicit IOOutOfRange(const std::string& description)
            : IOException(description) {}

        virtual ~IOOutOfRange() throw() {}
    };

    struct IOInvalidArgument : public IOException
    {
        inline IOInvalidArgument() : IOException("gs::IOInvalidArgument") {}

        inline explicit IOInvalidArgument(const std::string& description)
            : IOException(description) {}

        virtual ~IOInvalidArgument() throw() {}
    };

    /* Automatic replacement end} */
}

set infile "$inputdir/IOException.hh"
set outfile "$packagedir/interface/IOException.hh"
replace_lines $infile $outfile 5 24 $update

# Port the C++11 configuration part
file copy -force "$packagedir/interface/CPP11_config_enable.hh" "$packagedir/interface/CPP11_config.hh"
file delete "$packagedir/interface/CPP11_config_disable.hh" "$packagedir/interface/CPP11_config_enable.hh"

# Fix the programs from the "tools" section of geners
foreach hh [glob "$inputdir/../tools/*.hh"] {
    set outfile [file join "$packagedir/test" [file tail $hh]]
    filemap $hh $outfile $includemap
}

foreach cc [glob "$inputdir/../tools/*.cc"] {
    set outfile [file join "$packagedir/test" [file tail $cc]]
    filemap $cc $outfile $includemap
}
