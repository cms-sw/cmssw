#!/bin/sh
# -*- Tcl -*- the next line restarts using tclsh \
exec tclsh "$0" ${1+"$@"}

proc print_usage {} {
    puts stderr ""
    puts stderr "Usage: [file tail [info script]] file_name line_number_1 line_number_2 ..."
    puts stderr ""
    return
}

# Check the input arguments
if {$argc < 2} {
    print_usage
    exit 1
}

# Parse the arguments
set fname [lindex $argv 0]
if {![file readable $fname]} {
    puts stderr "Error: file \"$fname\" does not exist (or unreadable)"
    exit 1
}

proc file_contents {filename} {
    set chan [open $filename "r"]
    set contents [read $chan [file size $filename]]
    close $chan
    return $contents
}

set lines [split [file_contents $fname] "\n"]
set nlines [llength $lines]

set line_numbers [list]
foreach inp [lrange $argv 1 end] {
    if {![string is integer -strict $inp]} {
        puts stderr "Argument \"$inp\" does not represent a valid line number"
        exit 1
    }
    if {$inp <= 0 || $inp > $nlines} {
        puts stderr "Line number $inp is out of range"
        exit 1
    }
    lappend line_numbers [expr {$inp - 1}]
}

foreach l $line_numbers {
    puts [lindex $lines $l]
}

exit 0
