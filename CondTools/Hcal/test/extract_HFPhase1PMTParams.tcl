#!/bin/sh
# -*- Tcl -*- the next line restarts using tclsh \
exec tclsh "$0" ${1+"$@"}

global env
if {![info exists env(CMSSW_BASE)]} {
    puts stderr "Please set up CMSSW environment first"
    exit 1
}

set current_dir [pwd]
set cmssw_base $env(CMSSW_BASE)
set testdir $cmssw_base/src/CondTools/Hcal/test
set pkgdir HFPhase1PMTParams

set pkg_files {
    DataFormats/HcalDetId {
        HcalDetId.h
        HcalDetId.cc
        HcalSubdetector.h
    }
    DataFormats/DetId {
        DetId.h
    }
    CondTools/Hcal {
        CmdLine.h
        make_HFPhase1PMTParams.h
        parseHcalDetId.h
        parseHcalDetId.cc
        visualizeHFPhase1PMTParams.h
    }
    CondFormats/HcalObjects {
        AbsHcalAlgoData.h
        AbsHcalFunctor.h
        HcalChebyshevFunctor.h
        HcalChebyshevFunctor.cc
        HcalConstFunctor.h
        HcalConstFunctor.cc
        HcalCubicInterpolator.h
        HcalCubicInterpolator.cc
        HcalDetIdTransform.h
        HcalDetIdTransform.cc
        HcalIndexLookup.h
        HcalIndexLookup.cc
        HcalInterpolatedTableFunctor.h
        HcalInterpolatedTableFunctor.cc
        HcalItemArrayCollById.h
        HcalItemArrayColl.h
        HcalItemCollById.h
        HcalItemColl.h
        HcalLinearCompositionFunctor.h
        HcalLinearCompositionFunctor.cc
        HcalPiecewiseLinearFunctor.h
        HcalPiecewiseLinearFunctor.cc
        HcalPolynomialFunctor.h
        HcalPolynomialFunctor.cc
        HFPhase1PMTData.h
        HFPhase1PMTParams.h
    }
}

set test_files {
    visualizeHFPhase1PMTParams.cc
    write_HFPhase1PMTParams.cc
    makeHFPhase1PMTParamsPlots.cc
    make_HFPhase1PMTParams_data.cc
    make_HFPhase1PMTParams_dummy.cc
    make_HFPhase1PMTParams_mc.cc
    make_HFPhase1PMTParams_test.cc
    pmtlist_example.txt
}

set copy_files {
    Makefile.1 Makefile.common
    Makefile.2 Makefile
    Makefile.3 Makefile.plots
}

# Fetch all packages
#
# Porting consists in replacing the header file locations
# and changing all cms::Exception into standard exceptions
#
cd [file join $cmssw_base src]
set filelist [list]
set includemap [list "\#include \"FWCore/Utilities/interface/Exception.h\"" \
                    "\#include <stdexcept>"]
lappend includemap cms::Exception std::runtime_error

foreach {pkg files} $pkg_files {
    if {![file isdirectory $pkg]} {
        exec git cms-addpkg $pkg
    }
    foreach f $files {
        if {[string equal [file extension $f] ".h"]} {
            set header "$pkg/interface/$f"
            lappend includemap "\#include \"$header\"" "\#include \"$f\""
            lappend filelist $header
        } else {
            lappend filelist "$pkg/src/$f"
        }
    }
}

if {![file isdirectory CondFormats/Serialization]} {
    exec git cms-addpkg CondFormats/Serialization
}

proc file_contents {filename} {
    set chan [open $filename "r"]
    set contents [read $chan [file size $filename]]
    close $chan
    set contents
}

proc filemap {infile outfile map} {
    set fix [list "(\"Invalid DetId\") <<" "(\"Invalid DetId\"); // <<"]
    set in_data [file_contents $infile]
    set chan [open $outfile "w"]
    puts -nonewline $chan [string map $fix [string map $map $in_data]]
    close $chan
}

set todir "$testdir/$pkgdir"
file delete -force $todir
exec /bin/mkdir -p $todir
exec /bin/mkdir -p $todir/CondFormats/Serialization/interface
file copy -force CondFormats/Serialization/interface/eos \
    $todir/CondFormats/Serialization/interface
file copy -force CondFormats/Serialization/src/templateInstantiations.cc $todir

foreach f $filelist {
    set outfile [file join "$todir" [file tail $f]]
    filemap $f $outfile $includemap
}

foreach f $test_files {
    set infile [file join $testdir $f]
    set outfile [file join $todir $f]
    filemap $infile $outfile $includemap
}

foreach {f1 f2} $copy_files {
    set infile [file join $testdir $f1]
    set outfile [file join $todir $f2]
    file copy -force $infile $outfile
}

set tarfile [file join $current_dir $pkgdir.tar.gz]
file delete -force $tarfile
cd $testdir
exec /bin/tar -czvf $tarfile $pkgdir
file delete -force $todir
puts "Wrote file $pkgdir.tar.gz"

exit 0
