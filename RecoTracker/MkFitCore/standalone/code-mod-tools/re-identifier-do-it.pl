#!/ usr / bin / perl

die "Usage: $0 replacement-rules-file" unless - r $ARGV[0];

#$TARGET = "functions";
$TARGET = "data-members";

if ($TARGET eq "functions") {
  $PRE = '(::|\s|\.|->|"|\(|\[)';
  $POST = '(\s*\()';
}
elsif($TARGET eq "data-members") {
  $PRE = '(\W)';
  $POST = '(\W)';
  push @FILES, map{"MkFitCore/src/Mk$_.h"} qw{Base Fitter Finder};
  push @FILES, map{"MkFitCore/src/Mk$_.cc"} qw{Fitter Finder};
}

open(F, "$ARGV[0]");

while (my $l = <F>) {
  next if $l = ~m / ^\s* $ / ;
  next if $l = ~m / ^\s* # / ;
  chomp $l;

  my($from, $to) = $l = ~m / ^\s*(\w +)(?:\s + (\w +)\s*) ? $ / ;

  if (not defined $to) {
    if ($TARGET eq "functions") {
      $to = lcfirst($from);
    }
    elsif($TARGET eq "data-members") { $to = 'm_'.$from; }
  }

#my @matches = `find.- name \*.h - or -name \*.cc | xargs grep - P '${PRE}${from}${POST}'`;
#print "Replace '$from' --> '$to' in\n ", join(" ", @matches), "\n";
#next;

  my @matched_files;
  if ($TARGET eq "functions") {
    @matched_files = split("\n", `find.- name \*.h - or -name \*.cc | xargs grep - l - P '${PRE}${from}${POST}'`);
  }
  elsif($TARGET eq "data-members") { @matched_files = @FILES; }

  next unless @matched_files;

  print "Replace '$from' --> '$to' in ", join(" ", @matched_files), "\n";

    for
      my $fname(@matched_files) {
        my $xxx = $ / ;
        undef $ / ;
        open(X, '<', $fname) or die "Can not open $fname for reading";
        my $file = <X>;
        close(X);
        $ / = $xxx;

        $file = ~s / ${PRE} ${from} ${POST} / $1${to} $2 / msg;

        my @matches = $file = ~m / ^ .*$from.*$ / mg;
        print $fname, "\n  ", join("\n  ", @matches), "\n";

        open(X, '>', $fname) or die "Can not open $fname for writing";
        print X $file;
        close(X);
      }

    print "\n";
}

close(F);
