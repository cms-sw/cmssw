#!/usr/bin/perl

# all files: `find RecoTracker -name \*.h -or -name \*.cc -or -name \*.icc -or -name \*.acc`;

$depth = shift;
$depth = 1 unless defined $depth;

$pref = `pwd`; chomp $pref;
$pref =~ s!.*/(RecoTracker/.*)!$1!;

@headers = map { chomp; s!^./!!; $_; } `find . -maxdepth $depth -name \\*.h`;

print "PREF: $pref\n";
print "HEADERS:\n", join("\n", @headers);
print "\n\n";

local $/;
undef $/;


for my $file (@headers)
{
    open F, $file;
    $f = <F>;
    close F;

    # print $file,"\n",$f,"\n\n";

    my $incguard = "$pref/$file";
    $incguard =~ s!/!_!og;
    $incguard =~ s!\.(h|H)$!_$1!;

    print "$file   --> $incguard\n";

    if ($f =~ m/^#ifndef\s+(.*)\s+#define\s+(.*)/m)
    {
        my $same = $1 eq $2;
        if (not $same) { print "ERRORRORR incguard ifdef/defnot mathcing --- FIXFIXFIX\n"; next; }
        my $correct = $1 eq $incguard;
        print "  found existing include guard $1, $2 -- same $same, correct $correct\n";

        if (not $correct)
        {
            print "  FIXFIXFIX\n";

            $f =~ s/^#ifndef\s+(.*)\s+#define\s+(.*)/#ifndef $incguard\n#define $incguard/m;
        }

        open F, ">$file";
        print F $f;
        close F;
    }
    else
    {
        print "  NONONO include guard\n";
    }
}
