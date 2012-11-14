#!/bin/sh
# The next line restarts using tclsh \
exec tclsh "$0" ${1+"$@"}

#
# This script ports the "NPStat" package to CMSSW,
# in its part that is used to generate interpolation
# tables for jet corrections
#
set inputdir "/afs/cern.ch/user/i/igv/npstat-1.0.4"
set packagedir "/afs/cern.ch/user/i/igv/FFTJet_corr_v2/CMSSW_6_1_0_pre4/src/JetMETCorrections/InterpolationTables"

# Create the map for changing include statements
set includemap [list "\#include \"geners/" "\#include \"Alignment/Geners/interface/"]
lappend includemap "\#include \"npstat/stat/" "\#include \"JetMETCorrections/InterpolationTables/interface/"
lappend includemap "\#include \"npstat/nm/" "\#include \"JetMETCorrections/InterpolationTables/interface/"

# Take care of exceptions so that they comply with CMSSW guidelines
lappend includemap std::out_of_range npstat::NpstatOutOfRange
lappend includemap std::invalid_argument npstat::NpstatInvalidArgument
lappend includemap std::runtime_error npstat::NpstatRuntimeError
lappend includemap std::domain_error npstat::NpstatDomainError
lappend includemap "\#include <stdexcept>" "\#include \"JetMETCorrections/InterpolationTables/interface/NpstatException.h\""

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

proc find_npstat_file {header {addh 1}} {
    global inputdir
    if {$addh} {
        set hh "${header}h"
    } else {
        set hh $header
    }
    set dirlist [list [file join $inputdir npstat nm] \
                      [file join $inputdir npstat stat]]
    foreach dir $dirlist {
        set htry [file join $dir $hh]
        if {[file readable $htry]} {
            return $htry
        }
    }
    error "Failed to find npstat file $hh"
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

# Remove pieces between "#ifdef SWIG" and corresponding #endif
proc remove_swig {infile outfile} {
    set in_swig 0
    set output [list]
    set ifdef_count 0
    foreach line [split [file_contents $infile] "\n"] {
        set tline [string trim $line]
        if {!$in_swig} {
            if {[string equal $tline "#ifdef SWIG"]} {
                set in_swig 1
            }
        }
        if {$in_swig} {
            if {[string equal -length 6 $tline "#endif"]} {
                incr ifdef_count -1
                if {$ifdef_count == 0} {
                    set in_swig 0
                }
            }
            if {[string equal -length 6 $tline "#ifdef"]} {
                incr ifdef_count
            }
            if {[string equal -length 7 $tline "#ifndef"]} {
                incr ifdef_count
            }
        } else {
            lappend output $line
        }
    }
    set chan [open $outfile "w"]
    puts -nonewline $chan [join $output "\n"]
    close $chan
}

# Redo the #include statements so that they use .h instead of .hh
proc fix_naming_convention {infile outfile} {
    set output [list]
    set target "\#include \"JetMETCorrections/InterpolationTables/interface/"
    set tlen [string length $target]
    foreach line [split [file_contents $infile] "\n"] {
        set trline [string trim $line]
        if {[string equal -length 8 $trline "// \\file"]} {
            set line [string range $trline 0 end-1]
        } elseif {[string equal -length $tlen $trline $target]} {
            set tail [string range $trline end-3 end-1]
            if {[string equal $tail ".hh"]} {
                set line [string range $trline 0 end-2]
                append line "\""
            }
        }
        lappend output $line
    }
    set chan [open $outfile "w"]
    puts -nonewline $chan [join $output "\n"]
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

# Headers we need to include
set header_list [list \
    AbsArrayProjector.h \
    absDifference.h \
    AbsMultivariateFunctor.h \
    AbsVisitor.h \
    allocators.h \
    ArrayND.h \
    ArrayNDScanner.h \
    ArrayRange.h \
    ArrayShape.h \
    BoxND.h \
    BoxNDScanner.h \
    CircularMapper1d.h \
    closeWithinTolerance.h \
    ComplexComparesAbs.h \
    ComplexComparesFalse.h \
    convertAxis.h \
    CoordinateSelector.h \
    DualAxis.h \
    DualHistoAxis.h \
    EquidistantSequence.h \
    GridAxis.h \
    HistoAxis.h \
    HistoNDFunctorInstances.h \
    HistoND.h \
    interpolate.h \
    interpolateHistoND.h \
    InterpolationFunctorInstances.h \
    Interval.h \
    isMonotonous.h \
    LinearMapper1d.h \
    LinInterpolatedTableND.h \
    MultivariateFunctorScanner.h \
    NUHistoAxis.h \
    PreciseType.h \
    ProperDblFromCmpl.h \
    rescanArray.h \
    SimpleFunctors.h \
    StorableHistoNDFunctor.h \
    StorableInterpolationFunctor.h \
    StorableMultivariateFunctor.h \
    StorableMultivariateFunctorReader.h \
    UniformAxis.h]

set cc_list [list \
    ArrayNDScanner.cc \
    ArrayRange.cc \
    ArrayShape.cc \
    convertAxis.cc \
    DualAxis.cc \
    DualHistoAxis.cc \
    EquidistantSequence.cc \
    GridAxis.cc \
    HistoAxis.cc \
    NUHistoAxis.cc \
    StorableMultivariateFunctor.cc \
    StorableMultivariateFunctorReader.cc \
    UniformAxis.cc]

# Now, do the replacements
foreach header $header_list {
    set from_header [find_npstat_file $header]
    set to_header [file join "$packagedir/interface" $header]
    set tempfile [tempfile /tmp]
    insert_icc $from_header $tempfile [file dirname $from_header]
    remove_swig $tempfile $tempfile
    filemap $tempfile $to_header $includemap
    fix_naming_convention $to_header $to_header
    file delete $tempfile
}

foreach cc $cc_list {
    set fromfile [find_npstat_file $cc 0]
    set outfile [file join "$packagedir/src" $cc]
    filemap $fromfile $outfile $includemap
    fix_naming_convention $outfile $outfile
}
