#!/usr/bin/env perl

## Tool to dig out information about the event size in PAT
## 
## Please run this giving as argument the root file, and redirecting the output on an HTML file
## Notes:
##    - you must have a correctly initialized environment, and FWLite auto-loading with ROOT
##    - you must put in the same folder of the html also these three files:
##            http://cern.ch/gpetrucc/patsize.css
##            http://cern.ch/gpetrucc/blue-dot.gif
##            http://cern.ch/gpetrucc/red-dot.gif
##      otherwise you will get an unreadable output file
##    - for small files, compression does not work (as you will read from the output html)
##    - per-event provenance is just the GetZipBytes of EventMetaData, EventHistory.
##    -  

use strict; 
use warnings;
use Data::Dumper;
use File::Temp qw/tempfile/;
use File::stat;

my $filename = shift(@ARGV);

if ((!$filename) || ($filename eq "-h")) {
    print STDERR "Usage: diskSize.pl filename.root > filename.html\n";
    exit(1);
}

my $st = stat($filename);

my ($MACRO, $macrofile) = tempfile( "macroXXXXX", SUFFIX=>'.c' , UNLINK => 1 );
my ($macroname) = ($macrofile =~ m/(macro.....)\.c/);

print STDERR "Getting list of branches ...\n";

print $MACRO "void $macroname(){\nEvents->Print();\n}\n";
close $MACRO;
my $IN = qx(root.exe -b -l $filename $macrofile -q 2> /dev/null);

my %survey = ();
my $obj = undef; my $item = undef;
my $events = 0;
my %arrays = ();

foreach (split(/\n/, $IN)) {
  chomp; #print STDERR "    [$_]\n";
  if (m/\*Branch\s+:((\w+)_(\w+)_(\w*)_(\w+))\./) {
        $item = undef;
  }
  if (m/\*Br\s+\d+\s+:((\w+)_(\w+)_(\w*)_(\w+))\.obj\s/) {
        $survey{$1} = { 'type'=>$2, 'label'=>$3, 'instance'=>$4, 'process'=>$5, 'tot'=>0, 'num'=> 0, 'items'=>{},  };
        $obj = undef; $item = $1;
        #print STDERR "Got item $item\n";
  }
  next unless defined $item;
  if (m/\*Br\s+\d+\s+:((\w+)_(\w+)_(\w*)_(\w+))\.obj\.(\S+) :/) {
        $obj = $6; $item = $1;
        #print STDERR "Got item $item, obj $obj\n";
        die "Product $1 not found" unless defined($survey{$1});
  }
  if ((m/\|\s+\w+\[\S+/) && ($survey{$item}->{'type'} ne 'edmTriggerResults')) { $arrays{$item} = 1;  }
  next unless defined $obj;
  if (m/Entries\s+:\s*(\d+)\s+:\s+Total\s+Size=\s+(\d+)\s+bytes\s+File\s+Size\s+=\s+(\d+)/) {
        die "Mismatching number of events ($events, $1) " unless (($events == 0) || ($events == $1));
        $events = $1;
        $survey{$item}->{'items'}->{$obj} = { 'siz'=>$3/1024.0, 'ok'=>1 };
        $survey{$item}->{'tot'} += $survey{$item}->{'items'}->{$obj}->{'siz'};
  } elsif (m/Entries\s+:\s*(\d+)\s+:\s+Total\s+Size=\s+(\d+)\s+bytes\s+One basket in memory/) {
        die "Mismatching number of events ($events, $1) " unless (($events == 0) || ($events == $1));
        $events = $1;
        $survey{$item}->{'items'}->{$obj} = { 'siz'=>$2/1024.0, 'ok'=>0 };
        $survey{$item}->{'tot'} += $survey{$item}->{'items'}->{$obj}->{'siz'};
  }
}

my ($grandtotal,$provenance) = (0,0);
foreach (keys(%survey)) { $grandtotal += $survey{$_}->{'tot'}; }

print STDERR "Events: $events\n";
open $MACRO, "> $macrofile";
print $MACRO "void $macroname() {\n";
foreach my $coll (sort(keys(%arrays))) {
    print $MACRO "   Events->Draw(\"$coll.\@obj.size()>>htmp\");\n";
    print $MACRO "   std::cout << \"SIZE\t$coll\\t\" << (htmp->GetMean()*htmp->GetEntries()) << std::endl;\n";
}
print $MACRO "   std::cout << \"PROVENANCE\t\" << (EventMetaData->GetZipBytes()+EventHistory->GetZipBytes()) << std::endl;\n";
print $MACRO "}\n";
close $MACRO;

print STDERR "Getting items in the collections (it can take a while) ...\n";

my $root = qx(root.exe -b -l "$filename" -q $macrofile  2> /dev/null);
my @lines = split('\n', $root);
foreach (grep( /^SIZE\s+\S+\s+\S+/, @lines)) {
    my ($item, $total) = (m/SIZE\s+(\w+)\s+(\S+)/);
    $survey{$item}->{'num'} = $total;
}
foreach my $item (keys(%survey)) { $survey{$item}->{'num'} = $events if $survey{$item}->{'num'} == 0; }

foreach (grep( /^PROVENANCE\s+(\S+)/, @lines)) { /^PROVENANCE\s+(\S+)/ and $provenance = $1/1024.0; }

my $allsize = $st->size/1024.0;
my $s_allsize = sprintf("%.3f Mb, \%d events, %.2f kb/event", $allsize/1024.0, $events, $allsize/$events);

print <<_END_;
<html>
<head>
    <title>$filename : PAT Size ($s_allsize)</title>
    <link rel="stylesheet" type="text/css" href="patsize.css" />
</head>
<h1>Summary ($s_allsize)</h1>
<table>
_END_
print "<tr class='header'><th>".join("</th><th>", "Collection", "items/event", "kb/event", "kb/item", "plot", "%") . "</th></tr>\n";
foreach (sort({$survey{$b}->{'tot'} <=> $survey{$a}->{'tot'} }
              keys(%survey))) {
    print "<th><a href='#$_'>$_</a></th>";
    foreach my $val ($survey{$_}->{'num'}/$events, $survey{$_}->{'tot'}/$events, $survey{$_}->{'tot'}/$survey{$_}->{'num'}) {
        print sprintf("<td>%.2f</td>", $val);
    }
    print sprintf("<td class=\"img\"><img src='blue-dot.gif' width='\%d' height='\%d' /></td>",
                            $survey{$_}->{'tot'}/$grandtotal * 200, 10 );
    print sprintf("<td>%.1f%%</td>", $survey{$_}->{'tot'}/$grandtotal * 100.0);
    print "</tr>\n";
}

# all known data
print "<th>All Event data</th>";
print sprintf("<td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>" , $grandtotal/$events);
print sprintf("<td class=\"img\"><img src=\"green-dot.gif\" width='\%d' height='10' />", $grandtotal/$allsize*200.0);
print sprintf("</td><td>%.1f%%<sup>a</sup></td>", $grandtotal/$allsize*100.0);
print "</tr>\n";

# per-event provenance
print "<th>EventMetaData + EventHistory</th>";
print sprintf("<td>&nbsp;</td><td>%.2f</td><td>&nbsp;</td>", $provenance/$events);
print sprintf("<td class=\"img\"><img src='red-dot.gif' width='\%d' height='\%d' /></td>",$provenance/$allsize * 200, 10 );
print sprintf("<td>%.1f%%<sup>a</sup></td>", $provenance/$allsize * 100.0);
print "</tr>\n";

# other, unknown overhead
print "<th>Non per-event data or overhead</th>";
print sprintf("<td>&nbsp;</td><td>%.2f</td><td>&nbsp;</td>", ($allsize-$provenance-$grandtotal)/$events);
print sprintf("<td class=\"img\"><img src='red-dot.gif' width='\%d' height='\%d' /></td>",($allsize-$provenance-$grandtotal)/$allsize * 200, 10 );
print sprintf("<td>%.1f%%<sup>a</sup></td>", ($allsize-$provenance-$grandtotal)/$allsize * 100.0);
print "</tr>\n";


# all file
print "<th>File size</th>";
print sprintf("<td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>" , $allsize/$events);
print "<td>&nbsp;</td><td>&nbsp;</td></tr>\n";

print <<_END_;
</table>
Note: size percentages of individual event products are relative to the total size of Event data only.<br />
Percentages with <sup>a</sup> are instead relative to the full file size.
<h1>Detail</h1>
_END_
foreach (sort(keys(%survey))) {
    my $avg = sprintf("%.1f",$survey{$_}->{'num'}/$events);
    print <<_END_;
<h2><a name="$_" id="$_">$_</a> ($avg items/event)</h2>
<table>
_END_
    print "<tr class='header'><th>".join("</th><th>", "Datamember", "kb/event", "kb/item", "plot", "%", "compressed") . "</th></tr>\n";
    foreach my $it (sort({$survey{$_}->{'items'}->{$b}->{'siz'} <=> $survey{$_}->{'items'}->{$a}->{'siz'}} 
                         keys(%{$survey{$_}->{'items'}}))) {
        print "<th>$it</th>";
        my $IT = $survey{$_}->{'items'}->{$it};
        foreach my $val ($IT->{'siz'}/$events, $IT->{'siz'}/$survey{$_}->{'num'}) {
            print sprintf("<td>%.3f</td>", $val);
        }
        print sprintf("<td class=\"img\"><img src='\%s-dot.gif' width='\%d' height='\%d' /></td>",
                                ($IT->{'ok'} ? 'blue' : 'red'), $IT->{'siz'}/$survey{$_}->{'tot'} * 200, 10 );
        print sprintf("<td>%.1f%%</td>", $IT->{'siz'}/$survey{$_}->{'tot'} * 100.0);
        print "<td>". ($IT->{'ok'} ? 'ok' : 'no') . "</td>";
        print "</tr>\n";
    }
    print <<_END_;
</table>
_END_
}
print <<_END_;
</body></html>
_END_
close;
