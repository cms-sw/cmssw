#!/bin/tcsh

set num_args = ${#argv}

set myLang = "$LANG"
set tempLang = C
setenv LANG $tempLang

if ( $num_args == 3 ) then
    set input  = $argv[1]
    set bkmrk  = $argv[3]
    set output = $argv[2]
    if ( -e $CMSSW_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl ) then
        $CMSSW_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl $input $output $bkmrk
    else
        $CMSSW_RELEASE_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl $input $output $bkmrk
    endif
else if ( $num_args == 2 ) then 
    set input  = $argv[1]
    set base   = `echo $input | cut -d '.' -f 1`
    set bkmrk  = "$base-bookmark.txt"
    set output = $argv[2]
    if ( -e $CMSSW_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl ) then
        $CMSSW_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl $input $output $bkmrk
    else
        $CMSSW_RELEASE_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl $input $output $bkmrk
    endif
else if ( $num_args == 1 ) then 
    set input  = $argv[1]
    set base   = `echo $input | cut -d '.' -f 1`
    set bkmrk  = "$base-bookmark.txt"
    set output = "$base-bm.pdf"
    if ( -e $CMSSW_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl ) then
        $CMSSW_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl $input $output $bkmrk
    else
        $CMSSW_RELEASE_BASE/src/HLTrigger/Tools/scripts/bookmarkPdf.pl $input $output $bkmrk
    endif
else 
    echo "Usage: addBookmarks.csh <in.pdf> (<out.pdf>) (<bkmrk.txt>)"
    echo " "
    echo "Input PDF <in.pdf> and bookmarks text file <bkmrk.txt> must already exist,"
    echo "and results are output to bookmarked file <out.pdf>"
    echo "If <out.pdf> = <in.pdf>, the input file WILL be overwritten."
    echo "NOTE: <out.pdf> and <bkmrk.txt> are optional.  Defaults are [in]-bm.pdf and [in]-bookmark.txt, respectively"
endif

setenv LANG $myLang

