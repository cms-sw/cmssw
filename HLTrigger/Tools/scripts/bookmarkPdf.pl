#!/usr/bin/env perl

use strict ; 

# Get the input/output file names at the beginning
my $infile ; my $bkmrkfile ; my $outfile ;
if ($#ARGV == 2) {
    $infile = $ARGV[0] ; 
    $outfile = $ARGV[1] ; 
    $bkmrkfile = $ARGV[2] ; 
    print "Adding bookmarks to $infile.  Output file is $outfile.\n" ; 
} elsif ($#ARGV == 1) { 
    $infile = $ARGV[0] ; 
    $outfile = $ARGV[1] ;
    my @inSplitter = split(/\./,$infile) ;
    $bkmrkfile = "$inSplitter[0]-bookmark.txt"
} elsif ($#ARGV == 0) {
    my $arg = $ARGV[0] ; 
    chomp($arg) ; 
    if (($arg eq "-h") || ($arg eq "--help")) {
        die "\nUsage: (addBookmarks.csh/bookmarkPdf.pl) <in.pdf> (<out.pdf>) (<bkmrk.txt>)\n\n" . 
        "Input PDF <in.pdf> and bookmarks text file <bkmrk.txt> must already exist,\n" . 
        "and results are output to bookmarked file <out.pdf>\n\n" . 
        "If <out.pdf> = <in.pdf>, the input file WILL be overwritten.\n" .
        "NOTE: <out.pdf> and <bkmrk.txt> are optional.  Defaults are [in]-bm.pdf and [in]-bookmark.txt, respectively.\n\n" .
        "Format of <bkmrk.txt>: \n" . 
        "[Chapter name]^^[page number]\n" .
        "[Chapter name]^[Section name]^^[page number]\n" .
        "[Chapter name]^[Section name]^[Subsection name]^^[page number] ...\n" .
        "Each line of <bkmrk.txt> must specify a desired bookmark.\n" ; 
    } else {
        $infile = $ARGV[0] ; 
        my @inSplitter = split(/\./,$infile) ;
        # print "insplitter position 0 is $inSplitter[0]\n" ; 
        # print "insplitter position 1 is $inSplitter[1]\n" ; 
        $outfile = "$inSplitter[0]-bm.pdf" ;
        $bkmrkfile = "$inSplitter[0]-bookmark.txt"
#        die "\nInvalid format for input to (addBookmarks.csh/bookmarkPdf.pl).\n" .
#            "\nUsage: (addBookmarks.csh/bookmarkPdf.pl) <in.pdf> (<out.pdf>) (<bkmrk.txt>)\n\n" . 
#            "Input PDF <in.pdf> and bookmarks text file <bkmrk.txt> must already exist,\n" . 
#            "and results are output to bookmarked file <out.pdf>\n" . 
#            "If <out.pdf> = <in.pdf>, the input file WILL be overwritten.\n" .
#            "NOTE: <out.pdf> and <bkmrk.txt> are optional.  Defaults are [in]-bm.pdf and [in]-bookmark.txt, respectively.\n\n" ; 
 
    }
} else {
    die "\nUsage: (addBookmarks.csh/bookmarkPdf.pl) <in.pdf> (<out.pdf>) (<bkmrk.txt>)\n\n" . 
        "Input PDF <in.pdf> and bookmarks text file <bkmrk.txt> must already exist,\n" . 
        "and results are output to bookmarked file <out.pdf>\n" . 
        "If <out.pdf> = <in.pdf>, the input file WILL be overwritten.\n" . 
        "NOTE: <out.pdf> and <bkmrk.txt> are optional.  Defaults are [in]-bm.pdf and [in]-bookmark.txt, respectively.\n\n" ; 
}

# Quick sanity checks
open(INFILE,$infile) || die "Could not open $infile.\n" .
    "\nUsage: (addBookmarks.csh/bookmarkPdf.pl) <in.pdf> (<out.pdf>) (<bkmrk.txt>)\n\n" . 
    "Input PDF <in.pdf> and bookmarks text file <bkmrk.txt> must already exist,\n" . 
    "and results are output to bookmarked file <out.pdf>\n" . 
    "If <out.pdf> = <in.pdf>, the input file WILL be overwritten.\n" . 
    "NOTE: <out.pdf> and <bkmrk.txt> are optional.  Defaults are [in]-bm.pdf and [in]-bookmark.txt, respectively.\n\n" ; 
open(BKMRKS,$bkmrkfile) || die "Could not open $bkmrkfile.\n" .
    "\nUsage: (addBookmarks.csh/bookmarkPdf.pl) <in.pdf> (<out.pdf>) (<bkmrk.txt>)\n\n" . 
    "Input PDF <in.pdf> and bookmarks text file <bkmrk.txt> must already exist,\n" . 
    "and results are output to bookmarked file <out.pdf>\n" . 
    "If <out.pdf> = <in.pdf>, the input file WILL be overwritten.\n" .  
    "NOTE: <out.pdf> and <bkmrk.txt> are optional.  Defaults are [in]-bm.pdf and [in]-bookmark.txt, respectively.\n\n" ; 

# Create an array of bookmarks
my @bookmarkPageLabels ; 
my @bookmarkPages ; 
my @bookmarkRank ; 
my $numBookmarks = 0 ; 
my $maxRank = 0 ; 
while (my $line = <BKMRKS>) {
    $numBookmarks++ ; 
    chomp($line) ; 
    my @splitter = split(/\^\^/,$line) ; 
    $bookmarkPages[($numBookmarks-1)] = $splitter[1] ; 
    my @bkmrks = split(/\^/,$splitter[0]) ; 
    $bookmarkPageLabels[($numBookmarks-1)] = $bkmrks[$#bkmrks] ; 
    $bookmarkRank[($numBookmarks-1)] = $#bkmrks ; 
    ($bookmarkRank[($numBookmarks-1)] > $maxRank) && 
        ($maxRank = $bookmarkRank[($numBookmarks-1)]) ; 
}


my $objCount = 0 ; 
my $keepDumping = 1 ; 

my @objects ; 
my @newPdf ; 
my $lineCtr = 0 ; 

my $outlineLine ; 
my $firstChapter ; 
my $lastChapter ; 

my $inBaseObj = 0 ; 

my $objBase ; my $objOutlines ; my $objPages ; 
my $numPages = 0 ; my $pagesLine ; my $pageCountLine ; 
my $outbaseLine ; 

while (my $line = <INFILE>) {
    my $write_newline = 0 ; 
    my $newline = $line ; 

    #########################################################
    # Sanity check: make sure bookmarks don't already exist #
    #########################################################
    my $pdfStr = "this.pageNum" ; 
    if ($line =~ /$pdfStr/io ) {
        die "File $infile already has bookmarks.  Exiting...\n" ; 
    } 
    
    $pdfStr = " 0 obj" ; 
    if ($line =~ /$pdfStr/io ) {
        my @splitter = split(/ /, $line) ; 
        $objects[$objCount] = $splitter[0] ; 
        $objCount++ ; 
    } 

    $pdfStr = "Catalog" ; 
    if ($line =~ /$pdfStr/i) {
        $objBase = $objCount - 1 ; 
        $inBaseObj = 1 ;
    }
    ($objBase < ($objCount - 1)) && ($inBaseObj = 0) ; 

    if ($inBaseObj) {
        $pdfStr = "Outlines" ; 
        if ($line =~ /$pdfStr/i) {
            $outbaseLine = $lineCtr ; 
            my @splitter = split(/ /,$line) ; 
            $objOutlines = $splitter[1] ; 
        }
        $pdfStr = "Pages" ; 
        if ($line =~ /$pdfStr/i) {
            my @splitter = split(/ /,$line) ; 
            $objPages = $splitter[1] ; 
            $pagesLine = $lineCtr ; 
        }
    }

    if ($objects[$objCount-1] == $objOutlines) {
        $pdfStr = "Count" ; 
        if ($line =~ /$pdfStr/i) {
            $write_newline = 1 ; 
            $newline = "/First TBD 0 R /Last TBD 0 R\n" ; 
        }
    }

    if ($objects[$objCount-1] == $objPages) {
        $pdfStr = "Count" ; 
        if ($line =~ /$pdfStr/i) {
            my @splitter = split(/ /,$line) ; 
            $pageCountLine = $lineCtr ; 
            $numPages = $splitter[1] ; chomp($numPages) ; 
        }
    }

    # Calculate bookmarks at the end
    $pdfStr = "xref\n" ; 
    if ($line eq $pdfStr) {
        $keepDumping = 0 ; 
        $objCount++ ; 

        # Create a duplicate "Pages" object
        $newline = "$objCount 0 obj <</Type/Pages/Kids [$objPages 0 R ]" . 
            "/Count $numPages>> endobj\n" ; 
        $newPdf[$lineCtr] = $newline ; 
        $newPdf[$pagesLine] = "/Pages $objCount 0 R\n" ; 
        $newPdf[$pageCountLine] = "/Count $numPages /Parent $objCount 0 R\n" ; 
        $lineCtr++ ; 

        my @bookmarkLines ; 
        my @parentLoc ; 
        for (my $i=0; $i<=$maxRank; $i++) {
            $parentLoc[$i] = 0 ; 
        }

        my @taskObj ; my @xrefObj ; 
        my @parentObj ; my @firstKid ; my @lastKid ;
        my @prevXref ; my @nextXref ; 
        for (my $i=0; $i<$numBookmarks; $i++) {
            $taskObj[$i] = -1 ; $xrefObj[$i] = -1 ; 
            $parentObj[$i] = -1 ; $firstKid[$i] = -1 ; $lastKid[$i] = -1 ; 
            $prevXref[$i] = -1 ; $nextXref[$i] = -1 ; 
        }

        # Bookmarks filled in rank order
        my $bkCtr = 0 ; 
        for (my $rank=0; $rank<=$maxRank; $rank++) {
            my $nSameRank = 0 ; 
            my $rankLoc ; 
            for (my $i=0; $i<$numBookmarks; $i++) {
                if ($bookmarkRank[$i] == $rank) {
                    $bkCtr++ ; $nSameRank++ ; 
                    $xrefObj[$i] = $bkCtr ; 
                    $rankLoc .= $i . ',' ; 
                    ($rank<$maxRank) && ($parentLoc[($rank+1)] = $xrefObj[$i]) ; 
                } elsif ($bookmarkRank[$i] == ($rank+1)) {
                    $parentObj[$i] = $parentLoc[$bookmarkRank[$i]] ; 
                }
            }

            chop($rankLoc) ; 
            my @splitter = split(/,/, $rankLoc) ; 
            my $parentBkmrk = 0 ; 
            ($rank == 0) && ($firstChapter = $xrefObj[$splitter[0]]) ; 
            for (my $i=0; $i<$nSameRank; $i++) {
                if ($i != 0) {
                    if ($parentObj[$splitter[$i]] == $parentObj[$splitter[($i-1)]]) {
                        $prevXref[$splitter[$i]] = $xrefObj[$splitter[($i-1)]] ;  
                    } else {
                        $parentBkmrk = $splitter[$i] - 1 ;
                        $firstKid[($splitter[$i]-1)] = $xrefObj[$splitter[$i]] ;
                    }
                } else {
                    if ($rank > 0) {
                        $parentBkmrk = $splitter[$i] - 1 ; 
                        $firstKid[($splitter[$i]-1)] = $xrefObj[$splitter[$i]] ;
                    }
                } 
                if ($i!=($nSameRank-1)) {
                    if ($parentObj[$splitter[$i]] == $parentObj[$splitter[($i+1)]]) {
                        $nextXref[$splitter[$i]] = $xrefObj[$splitter[($i+1)]] ;
                    } else {
                        $lastKid[$parentBkmrk] = $xrefObj[$splitter[$i]] ;
                    }
                } else {
                    if ($rank > 0) {
                        $lastKid[$parentBkmrk] = $xrefObj[$splitter[$i]] ; 
                    }
                }
                $taskObj[$splitter[$i]] = $bkCtr + $i + 1; 
                ($rank == 0) && ($lastChapter = $xrefObj[$splitter[$i]]) ; 
            }
            $bkCtr += $nSameRank ; 
        }

        for (my $i=0; $i<$numBookmarks; $i++) {
            my $rank = $bookmarkRank[$i] ; 
            my $name = $bookmarkPageLabels[$i] ; 
            my $page = $bookmarkPages[$i] - 1 ; 

            my $objValTask = $objCount + $taskObj[$i] ; 
            my $objValXref = $objCount + $xrefObj[$i] ; 
            my $objParentVal = $objOutlines ; 
            ($rank > 0) && ($objParentVal = $objCount + $parentObj[$i]) ; 
            my $firstVal = $objCount + $firstKid[$i] ; 
            my $lastVal = $objCount + $lastKid[$i] ; 
            my $prevVal = $objCount + $prevXref[$i] ; 
            my $nextVal = $objCount + $nextXref[$i] ; 

            ($rank == 0) && ($objParentVal = $objOutlines) ; 

            $bookmarkLines[(2*$i)] = "$objValTask 0 obj" . 
                "<</S/JavaScript/JS " . 
                "(this.pageNum = $page; " . 
                "this.scroll\\(40, 200\\);)" . 
                ">>endobj\n"; 
            $bookmarkLines[(2*$i+1)] = "$objValXref 0 obj" . 
                "<</Title ($name) /Parent $objParentVal 0 R" . 
                "/A $objValTask 0 R" ;
            ($firstKid[$i] > 0) && 
                ($bookmarkLines[(2*$i+1)] .= "/First $firstVal 0 R") ; 
            ($lastKid[$i] > 0) && 
                ($bookmarkLines[(2*$i+1)] .= "/Last $lastVal 0 R") ; 
            ($prevXref[$i] > 0) && 
                ($bookmarkLines[(2*$i+1)] .= "/Prev $prevVal 0 R") ; 
            ($nextXref[$i] > 0) && 
                ($bookmarkLines[(2*$i+1)] .= "/Next $nextVal 0 R") ; 
            $bookmarkLines[(2*$i+1)] .= ">>endobj\n" ;  
        }

        for (my $i=0; $i<(2*$numBookmarks); $i++) {
            $newPdf[$lineCtr] = $bookmarkLines[$i] ; 
            $lineCtr++ ; 
        }

        $firstChapter += $objCount ; 
        $lastChapter += $objCount ; 
        $objCount += 2*$numBookmarks ; 
        $lineCtr++ ; 
        $newPdf[$lineCtr] = $line ;
    } elsif ($keepDumping) {
        if (($lineCtr > 0) && ($lineCtr == $outbaseLine)) {
            chomp($line) ; 
            $line .= " /PageMode /UseOutlines\n" ; 
        }
        $newPdf[$lineCtr] = $line ; 
        $lineCtr++ ; 
        if ($write_newline) {
            $newPdf[$lineCtr] = $newline ; 
            $outlineLine = $lineCtr ; 
            $lineCtr++ ;
        }
    }
}
$newPdf[$outlineLine] = "/First $firstChapter 0 R/Last $lastChapter 0 R\n" ; 

###############
# Dump output #
###############
if (-e $outfile) {
    print "Results will overwrite existing file $outfile\n"  ;
    system("rm $outfile") ; 
    (-e $outfile) && print "File not successfully removed\n" ; 
}
open(OUTFILE,">$outfile") || die "Could not open $outfile for writing\n" ; 

my @objChars ; 
for (my $i=0; $i<$objCount; $i++) {
    $objChars[$i] = 0 ; 
}
my $nChars = 0 ; 
foreach my $line (@newPdf) {
    my $pdfStr = " 0 obj" ; 
    if ($line =~ /$pdfStr/i) {
        my @splitter = split(/ /, $line) ; 
        my $objNum = $splitter[0] ; 
        $objChars[$objNum] = $nChars ; 
    }
    $pdfStr = "xref" ; 
    if (!($line =~ /$pdfStr/i)) {
        $nChars += length($line) ; 
    }
    print OUTFILE $line ; 
}

my $endCount = $objCount + 1 ; 
print OUTFILE "0 $endCount\n" ; 
print OUTFILE "0000000000 65535 f\n" ; 

for (my $i=1; $i<$endCount; $i++) {
    my $numZeros = 10 - length($objChars[$i]) ; 
    for (my $j=0; $j<$numZeros; $j++) {
        print OUTFILE "0" ; 
    }
    print OUTFILE "$objChars[$i] 00000 n\n" ;
}

print OUTFILE "trailer\n" ; 
print OUTFILE "<<\n" ; 
print OUTFILE "/Size $endCount\n" ; 
print OUTFILE "/Root 1 0 R\n" ; 
print OUTFILE "/Info 2 0 R\n" ; 
print OUTFILE ">>\n" ; 
print OUTFILE "startxref\n" ; 
print OUTFILE "$nChars\n" ; 
print OUTFILE "%%EOF\n" ; 

close(OUTFILE) || die "Could not close $outfile\n" ; 
