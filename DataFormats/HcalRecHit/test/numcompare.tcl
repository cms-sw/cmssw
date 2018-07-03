#!/bin/sh
# -*- Tcl -*- the next line restarts using tclsh \
exec tclsh "$0" ${1+"$@"}

# Default precision
set eps 1.0e-5

proc print_usage {} {
    puts stderr ""
    puts stderr "Usage: [file tail [info script]] file1 file2 \[epsilon\]"
    puts stderr ""
    return
}

# Check the input arguments
if {$argc != 2 && $argc != 3} {
    print_usage
    exit 1
}

# Parse the arguments
foreach {file1 file2} $argv break
foreach fname [list $file1 $file2] {
    if {![file readable $fname]} {
        puts stderr "Error: file \"$fname\" does not exist (or unreadable)"
        exit 1
    }
}

if {$argc > 2} {
    set eps [lindex $argv 2]
    if {![string is double -strict $eps]} {
        print_usage
        exit 1
    }
    if {$eps < 0.0} {
        puts stderr "Error: comparison precision can not be negative"
        exit 1
    }
}
set eps [expr {1.0 * $eps}]

proc file_contents {filename} {
    set chan [open $filename "r"]
    set contents [read $chan [file size $filename]]
    close $chan
    return $contents
}

proc is_commentline {line} {
    string equal -length 1 $line "\#"
}

proc uncomment {filename} {
    set uncommented [list]
    set linenum 0
    foreach line [split [file_contents $filename] "\n"] {
        incr linenum
        if {![is_commentline $line]} {
            lappend uncommented $linenum $line
        }
    }
    list $linenum $uncommented
}

proc is_same_double {x y eps} {
    set mag [expr {abs(($x + $y)/2.0)}]
    set diff [expr {abs($x - $y)/($mag + 1.0)}]
    if {$diff <= $eps} {
        return 1
    } else {
        return 0
    }
}

proc min {n1 n2} {
    if {$n1 < $n2} {
        return $n1
    } else {
        return $n2
    }
}

proc numequal {line1 line2 eps} {
    if {[string equal $line1 $line2]} {
        return 1
    }
    set words1 [regexp -all -inline {\S+} [string map {= " "} $line1]]
    set words2 [regexp -all -inline {\S+} [string map {= " "} $line2]]
    if {[llength $words1] != [llength $words2]} {
        return 0
    }
    foreach w1 $words1 w2 $words2 {
        if {[string is integer -strict $w1] && \
                [string is integer -strict $w2]} {
            # Compare $w1 and $w2 as integers
            if {$w1 != $w2} {
                 return 0
            }
        } elseif {[string is double -strict $w1] && \
                      [string is double -strict $w2]} {
            # Compare $w1 and $w2 as real numbers
            if {![is_same_double $w1 $w2 $eps]} {
                return 0
            }
        } else {
            # Compare $w1 and $w2 as strings
            if {![string equal $w1 $w2]} {
                return 0
            }
        }
    }
    return 1
}

foreach {n1 lines1} [uncomment $file1] break
foreach {n2 lines2} [uncomment $file2] break

if {$n1 != $n2} {
    puts stderr "Files \"$file1\" and \"$file2\" have different number of lines ($n1 and $n2)"
    exit 1
}

if {[llength $lines1] != [llength $lines2]} {
    puts stderr "Files \"$file1\" and \"$file2\" have different number of comments"
    exit 1
}

set diffs_found 0
foreach {n1 l1} $lines1 {n2 l2} $lines2 {
    if {$n1 != $n2} {
        puts stderr "Different line [min $n1 $n2] in files \"$file1\" and \"$file2\""
        incr diffs_found
    } elseif {![numequal $l1 $l2 $eps]} {
        puts stderr "Different line $n1 in files \"$file1\" and \"$file2\""
        incr diffs_found
    }
}

if {$diffs_found} {
    puts "$diffs_found different lines found"
    exit 1
}

exit 0
